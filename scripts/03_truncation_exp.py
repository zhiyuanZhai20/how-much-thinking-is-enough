"""Phase 3: progressive truncation experiment.

For each correct trace with N steps, ask the judge to produce an answer from
the prefix r_1..r_k for k=1..N. Identify the critical point k* (smallest k
that yields the correct answer).

Self-judge uses deepseek-chat (cheap, no reasoning).
Cross-judge uses gpt-4o-mini.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.api_clients import chat_async, ConcurrencyLimiter
from utils.answer_extraction import extract_answer, answers_equal
from utils.io_utils import iter_jsonl, append_jsonl, done_keys


JUDGE_PROMPT = (
    "You are evaluating a partial mathematical reasoning trace. Based ONLY on "
    "the reasoning provided below, determine the final answer to the problem.\n\n"
    "Problem: {problem}\n\n"
    "Partial reasoning:\n{reasoning}\n\n"
    "Based on the above reasoning, output ONLY the final answer in \\boxed{{}} format. "
    "Do not add any additional reasoning."
)


DATASET_FILES = {
    "math500": "math500.jsonl",
    "gsm8k": "gsm8k_500.jsonl",
    "svamp": "svamp_100.jsonl",
    "aqua": "aqua_100.jsonl",
}

def load_problem_text(dataset: str) -> dict[str, str]:
    fname = DATASET_FILES.get(dataset, f"{dataset}.jsonl")
    path = config.DATA_DIR / fname
    return {r["id"]: r["problem"] for r in iter_jsonl(path)}


async def judge_one(provider, model, problem_text, prefix_text, ground_truth):
    prompt = JUDGE_PROMPT.format(problem=problem_text, reasoning=prefix_text)
    res = await chat_async(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=config.MAX_TOKENS_JUDGE,
    )
    pred = extract_answer(res.content)
    return bool(answers_equal(pred, ground_truth)), pred


MAX_TRUNC_POINTS = 30  # for long traces, sample this many evenly-spaced k values


def _ks_to_eval(N: int) -> list[int]:
    if N <= MAX_TRUNC_POINTS:
        return list(range(1, N + 1))
    # evenly spaced, always include k=1 and k=N
    import numpy as np
    ks = np.linspace(1, N, MAX_TRUNC_POINTS).round().astype(int).tolist()
    return sorted(set(ks))


async def evaluate_trace(seg_record, problem_text, judge_provider, judge_model, limiter, out_path, ground_truth):
    task_id = f"{seg_record['task_id']}__judge"
    steps = seg_record["steps"]
    N = len(steps)
    if N == 0:
        return
    truncation_results = []
    for k in _ks_to_eval(N):
        prefix = "\n\n".join(steps[:k])
        async with limiter:
            try:
                ok, pred = await asyncio.wait_for(
                    judge_one(judge_provider, judge_model, problem_text, prefix, ground_truth),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                ok, pred = False, "ERR:timeout"
            except Exception as e:
                ok, pred = False, f"ERR:{type(e).__name__}"
        truncation_results.append({"k": k, "correct": ok, "pred": pred})
    correct_ks = [r["k"] for r in truncation_results if r["correct"]]
    critical_point = min(correct_ks) if correct_ks else None
    redundancy = (1 - critical_point / N) if critical_point else 0.0
    append_jsonl(out_path, {
        "task_id": seg_record["task_id"],
        "problem_id": seg_record["problem_id"],
        "sample_idx": seg_record["sample_idx"],
        "dataset": seg_record["dataset"],
        "level": seg_record.get("level"),
        "n_steps": N,
        "step_tokens": seg_record["step_tokens"],
        "total_tokens_words": seg_record["total_tokens_words"],
        "truncation_results": truncation_results,
        "critical_point": critical_point,
        "redundancy_step": redundancy,
    })


async def run(model_name: str, dataset: str, judge_label: str, judge_provider: str, judge_model: str):
    seg_path = config.segments_path(model_name, dataset)
    out_path = config.truncation_path(model_name, dataset, judge_label)
    if not seg_path.exists():
        return
    problem_texts = load_problem_text(dataset)
    done = done_keys(out_path, "task_id")
    # Only evaluate correct traces; cap to keep compute manageable in pilot.
    correct_traces = [r for r in iter_jsonl(seg_path) if r["is_correct"] and r["n_steps"] >= 2]
    # Optional cap for pilot only
    if config.PILOT_MODE:
        correct_traces = correct_traces[:60]
    todo = [r for r in correct_traces if r["task_id"] not in done]
    if not todo:
        print(f"[{model_name}/{dataset}/{judge_label}] all done ({len(correct_traces)} traces)")
        return
    print(f"[{model_name}/{dataset}/{judge_label}] judging {len(todo)} traces", flush=True)
    limiter = ConcurrencyLimiter(config.MAX_CONCURRENT)
    # Need ground truths
    gt_lookup = {}
    for r in iter_jsonl(config.trace_path(model_name, dataset)):
        gt_lookup[r["task_id"]] = r["ground_truth"]

    # Chunk traces so we never have more than CHUNK traces in flight at once.
    # This avoids scheduler pathologies when thousands of sub-tasks
    # simultaneously contend for a small semaphore.
    CHUNK = max(config.MAX_CONCURRENT * 2, 16)
    from tqdm import tqdm as stqdm
    pbar = stqdm(total=len(todo), desc=f"trunc {model_name}/{dataset}/{judge_label}")
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        coros = []
        for seg in batch:
            ptext = problem_texts.get(seg["problem_id"])
            if ptext is None:
                continue
            gt = gt_lookup.get(seg["task_id"])
            coros.append(evaluate_trace(seg, ptext, judge_provider, judge_model, limiter, out_path, gt))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        pbar.update(len(batch))
    pbar.close()


async def main():
    ALL_DS = ["math500", "gsm8k", "svamp", "aqua"]
    for model_name, _, _ in config.REASONING_MODELS:
        for dataset in ALL_DS:
            await run(model_name, dataset, "self", *config.JUDGE_SELF)
    # cross-judge only for deepseek_r1 on math500/gsm8k (cost bounded)
    for dataset in ["math500", "gsm8k"]:
        await run("deepseek_r1", dataset, "cross", *config.JUDGE_CROSS)


if __name__ == "__main__":
    asyncio.run(main())
