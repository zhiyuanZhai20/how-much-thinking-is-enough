"""Phase 3b: Self-judge truncation experiment.

PRIMARY JUDGE = the reasoning model itself. For each correct trace of
model M with steps (r_1, ..., r_N), for each k in {1, ..., N}:
  - Present M with its own first-k steps and force it to commit to a
    final boxed answer.
  - The smallest k at which M's own self-continuation produces the
    correct answer is k*(r).

CROSS JUDGE = gpt-4o-mini, uniformly across all models.

Invocation differs per model because of API limitations:
  - deepseek_r1 (beta endpoint): chat prefix completion with
    assistant message `<think>\n{prefix}\n</think>\n\n`, stop=</think>
  - qwq_32b (DashScope): stream + partial assistant, same wrapper
  - r1_distill_7b (DashScope): user-prompt approach (the model cannot
    continue cleanly from partial prefill), enable_thinking=False
  - qwen3_30b_thinking (DashScope): user-prompt approach
    (partial blocked by API when enable_thinking=true)

All four strategies implement the same semantic contract: "the same
model that produced the trace is now asked to commit to a final answer
given only its own first-k reasoning steps."
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm as stqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import iter_jsonl, append_jsonl, done_keys
from utils.answer_extraction import extract_answer, answers_equal


MAX_TRUNC_POINTS = 20   # sub-sample long traces to 20 truncation levels
import os
MAX_PRIMARY_CONCURRENCY = int(os.getenv("MAX_CONC", "6"))

DATASET_FILES = {
    "math500": "math500.jsonl",
    "gsm8k": "gsm8k_500.jsonl",
    "svamp": "svamp_100.jsonl",
    "aqua": "aqua_100.jsonl",
}


# ---------------- per-model client builders ---------------- #

def _client(kind: str) -> AsyncOpenAI:
    if kind == "deepseek_beta":
        return AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY,
                           base_url="https://api.deepseek.com/beta")
    elif kind == "dashscope":
        return AsyncOpenAI(api_key=config.DASHSCOPE_API_KEY,
                           base_url=config.DASHSCOPE_BASE_URL)
    elif kind == "openai":
        return AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    raise ValueError(kind)


# ---------------- primary-judge invocations ---------------- #

SELF_INSTR = (
    "Problem: {problem}\n\n"
    "Here is my partial reasoning for this problem:\n\n"
    "{prefix}\n\n"
    "Based solely on my partial reasoning above, output ONLY the final "
    "numerical answer in \\boxed{{}} format. Do not add any additional reasoning."
)


async def _stream_collect(stream):
    c_parts, r_parts = [], []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            r_parts.append(delta.reasoning_content)
        if delta.content:
            c_parts.append(delta.content)
    return "".join(c_parts), "".join(r_parts)


async def judge_deepseek_r1(problem: str, prefix: str) -> str:
    """Chat prefix completion; force exit via </think> stop."""
    cli = _client("deepseek_beta")
    res = await cli.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "user", "content": problem + "\n\nReason step by step and put final answer in \\boxed{}."},
            {"role": "assistant", "content": f"<think>\n{prefix}\n</think>\n\n", "prefix": True},
        ],
        max_tokens=512,
        stop=["</think>"],
    )
    return res.choices[0].message.content or ""


async def judge_qwq(problem: str, prefix: str) -> str:
    """Stream + partial; prefix includes </think>, we take post-think content."""
    cli = _client("dashscope")
    stream = await cli.chat.completions.create(
        model="qwq-32b",
        messages=[
            {"role": "user", "content": problem + "\n\nReason step by step and put final answer in \\boxed{}."},
            {"role": "assistant", "content": f"<think>\n{prefix}\n</think>\n\n", "partial": True},
        ],
        max_tokens=256,
        stream=True,
    )
    c, _ = await _stream_collect(stream)
    return c


async def judge_user_prompt(problem: str, prefix: str, model: str,
                             enable_thinking_false: bool = False) -> str:
    """For models where partial/prefix doesn't work cleanly: ask the same
    model to output the answer given its own partial reasoning."""
    cli = _client("dashscope")
    extra = {}
    if enable_thinking_false:
        extra["enable_thinking"] = False
    stream = await cli.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": SELF_INSTR.format(problem=problem, prefix=prefix)}],
        max_tokens=512,
        stream=True,
        extra_body=extra or None,
    )
    c, _ = await _stream_collect(stream)
    return c


async def judge_r1_distill(problem: str, prefix: str) -> str:
    return await judge_user_prompt(problem, prefix,
                                    "deepseek-r1-distill-qwen-7b",
                                    enable_thinking_false=True)


async def judge_qwen3(problem: str, prefix: str) -> str:
    # Qwen3-30B-Thinking requires enable_thinking=True; it will produce
    # some reasoning_content first but we only care about the final content.
    return await judge_user_prompt(problem, prefix,
                                    "qwen3-30b-a3b-thinking-2507",
                                    enable_thinking_false=False)


async def judge_cross_4omini(problem: str, prefix: str) -> str:
    """Cross-judge: GPT-4o-mini, uniform across all models."""
    cli = _client("openai")
    res = await cli.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": SELF_INSTR.format(problem=problem, prefix=prefix)}],
        temperature=0.0,
        max_tokens=256,
    )
    return res.choices[0].message.content or ""


PRIMARY_JUDGES = {
    "deepseek_r1":        judge_deepseek_r1,
    "qwq_32b":            judge_qwq,
    "r1_distill_7b":      judge_r1_distill,
    "qwen3_30b_thinking": judge_qwen3,
}


# ---------------- truncation sampling ---------------- #

def _ks_to_eval(N: int) -> list[int]:
    if N <= MAX_TRUNC_POINTS:
        return list(range(1, N + 1))
    import numpy as np
    ks = np.linspace(1, N, MAX_TRUNC_POINTS).round().astype(int).tolist()
    return sorted(set(ks))


# ---------------- per-trace evaluation ---------------- #

async def evaluate_trace(seg, problem_text, gt, judge_fn, out_path, sem):
    steps = seg["steps"]
    N = len(steps)
    if N == 0:
        return
    results = []
    for k in _ks_to_eval(N):
        prefix = "\n\n".join(steps[:k])
        async with sem:
            try:
                out = await asyncio.wait_for(judge_fn(problem_text, prefix), timeout=90)
                pred = extract_answer(out)
                ok = bool(answers_equal(pred, gt))
            except asyncio.TimeoutError:
                pred, ok = "ERR:timeout", False
            except Exception as e:
                pred, ok = f"ERR:{type(e).__name__}", False
        results.append({"k": k, "correct": ok, "pred": pred})
    correct_ks = [r["k"] for r in results if r["correct"]]
    cp = min(correct_ks) if correct_ks else None
    rho = (1 - cp / N) if cp else 0.0
    append_jsonl(out_path, {
        "task_id": seg["task_id"],
        "problem_id": seg["problem_id"],
        "sample_idx": seg["sample_idx"],
        "dataset": seg["dataset"],
        "level": seg.get("level"),
        "n_steps": N,
        "step_tokens": seg["step_tokens"],
        "total_tokens_words": seg["total_tokens_words"],
        "truncation_results": results,
        "critical_point": cp,
        "redundancy_step": rho,
    })


# ---------------- pipeline per (model, dataset, judge) ---------------- #

async def run_condition(model_name: str, dataset: str, judge_name: str, judge_fn):
    seg_path = config.segments_path(model_name, dataset)
    if not seg_path.exists():
        print(f"[{model_name}/{dataset}/{judge_name}] no segments, skip")
        return
    out_path = config.TRUNC_DIR / f"v2__{model_name}__{dataset}__judge-{judge_name}.jsonl"
    done = done_keys(out_path, "task_id")

    # problem text
    fname = DATASET_FILES.get(dataset)
    ptext_map = {r["id"]: r["problem"] for r in iter_jsonl(config.DATA_DIR / fname)}

    # ground truths
    gt_map = {r["task_id"]: r["ground_truth"]
              for r in iter_jsonl(config.trace_path(model_name, dataset))}

    # correct traces only
    correct = [r for r in iter_jsonl(seg_path) if r["is_correct"] and r["n_steps"] >= 2]
    todo = [r for r in correct if r["task_id"] not in done]
    if not todo:
        print(f"[{model_name}/{dataset}/{judge_name}] all done ({len(correct)} traces)")
        return
    print(f"[{model_name}/{dataset}/{judge_name}] judging {len(todo)} traces", flush=True)

    sem = asyncio.Semaphore(MAX_PRIMARY_CONCURRENCY)
    CHUNK = MAX_PRIMARY_CONCURRENCY * 2
    pbar = stqdm(total=len(todo), desc=f"{model_name}/{dataset}/{judge_name}")
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        coros = []
        for seg in batch:
            ptext = ptext_map.get(seg["problem_id"])
            gt = gt_map.get(seg["task_id"])
            if ptext is None or gt is None:
                continue
            coros.append(evaluate_trace(seg, ptext, gt, judge_fn, out_path, sem))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        pbar.update(len(batch))
    pbar.close()


async def main():
    import os
    ALL_DS = ["svamp", "gsm8k", "aqua", "math500"]
    MODELS = [m for m, _, _ in config.REASONING_MODELS]
    only_primary = os.getenv("ONLY_PRIMARY") == "1"
    only_cross = os.getenv("ONLY_CROSS") == "1"
    only_model = os.getenv("ONLY_MODEL")  # e.g., "qwq_32b"

    if only_model:
        MODELS = [m for m in MODELS if m == only_model]

    if not only_cross:
        # Process each (model, dataset) sequentially per process; caller runs
        # multiple processes in parallel (one per model) for speedup.
        for model_name in MODELS:
            for ds in ALL_DS:
                try:
                    await run_condition(model_name, ds, "self",
                                         PRIMARY_JUDGES[model_name])
                except Exception as e:
                    print(f"[primary {model_name}/{ds}] FATAL: {e}")

    if not only_primary:
        # Cross judge uses gpt-4o-mini; can be parallelised across models via
        # ONLY_MODEL filter. Here we just iterate (single-process case) OR
        # if ONLY_MODEL is set, it's already filtered above.
        for model_name in MODELS:
            for ds in ALL_DS:
                try:
                    await run_condition(model_name, ds, "cross",
                                         judge_cross_4omini)
                except Exception as e:
                    print(f"[cross {model_name}/{ds}] FATAL: {e}")


if __name__ == "__main__":
    asyncio.run(main())
