"""Phase 9: Prefix-position ablation.

For each correct trace, compare four prefix strategies at the same k:
  - first-k:   steps[0:k]     (this is what Phase 3 already measured)
  - last-k:    steps[-k:]     (is the TAIL sufficient?)
  - middle-k:  steps around N/2
  - random-k:  a uniform random subset of k steps (sorted to preserve order)

Theory prediction: if redundancy concentrates in the tail (Finding 6 /
Theorem 2), then last-k should be *at least as good* as first-k at the
same k, middle-k should be comparable or worse, and random-k should be in
between. The contrast first-k vs middle-k is the sharpest test.

Output: outputs/ablation/prefix_position__<model>__math500.jsonl
  {task_id, problem_id, n_steps, k_values, strategies: {
     first: [correct per k],
     last:  [correct per k],
     middle:[correct per k],
     random:[correct per k]}}

We restrict to math500 for cost, and sub-sample N_TRACES per model.
"""
from __future__ import annotations
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.api_clients import chat_async, ConcurrencyLimiter
from utils.answer_extraction import extract_answer, answers_equal
from utils.io_utils import iter_jsonl, append_jsonl, done_keys

from tqdm import tqdm as stqdm

N_TRACES_PER_MODEL = 120  # random subset
K_POINTS_PER_TRACE = 8    # evenly spaced k values per strategy
STRATEGIES = ["first", "last", "middle", "random"]

JUDGE_PROMPT = (
    "You are evaluating a partial mathematical reasoning trace. Based ONLY "
    "on the reasoning provided below, determine the final answer to the "
    "problem.\n\n"
    "Problem: {problem}\n\n"
    "Partial reasoning:\n{reasoning}\n\n"
    "Based on the above reasoning, output ONLY the final answer in "
    "\\boxed{{}} format. Do not add any additional reasoning."
)


def _build_prefix(steps: list[str], k: int, strategy: str, seed: int) -> str:
    N = len(steps)
    k = max(1, min(k, N))
    if strategy == "first":
        idx = list(range(0, k))
    elif strategy == "last":
        idx = list(range(N - k, N))
    elif strategy == "middle":
        start = max(0, (N - k) // 2)
        idx = list(range(start, start + k))
    elif strategy == "random":
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(N), k))
    else:
        raise ValueError(strategy)
    return "\n\n".join(steps[i] for i in idx)


async def judge_one(provider, model, problem_text, prefix_text):
    prompt = JUDGE_PROMPT.format(problem=problem_text, reasoning=prefix_text)
    res = await chat_async(
        provider=provider, model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=256,
    )
    return extract_answer(res.content)


async def run_trace(seg, problem_text, gt, judge_provider, judge_model, limiter, out_path):
    task_id = seg["task_id"]
    steps = seg["steps"]
    N = len(steps)
    if N < 4:
        return
    # evenly spaced k values in [1, N]
    import numpy as np
    k_values = sorted(set(
        int(v) for v in np.linspace(1, N, K_POINTS_PER_TRACE).round().astype(int)
    ))
    results = {s: [] for s in STRATEGIES}
    for k in k_values:
        for strat in STRATEGIES:
            prefix = _build_prefix(steps, k, strat, seed=int(hash(task_id)) % (2**31))
            async with limiter:
                try:
                    pred = await asyncio.wait_for(
                        judge_one(judge_provider, judge_model, problem_text, prefix),
                        timeout=60,
                    )
                except Exception as e:
                    pred = f"ERR:{type(e).__name__}"
            ok = bool(answers_equal(pred, gt))
            results[strat].append({"k": k, "correct": ok, "pred": pred})
    append_jsonl(out_path, {
        "task_id": task_id,
        "problem_id": seg["problem_id"],
        "n_steps": N,
        "k_values": k_values,
        "strategies": results,
        "level": seg.get("level"),
    })


def load_problem_text(dataset: str) -> dict[str, str]:
    path = config.DATA_DIR / ("math500.jsonl" if dataset == "math500" else "gsm8k_500.jsonl")
    return {r["id"]: r["problem"] for r in iter_jsonl(path)}


async def run_model(model_name: str):
    seg_path = config.segments_path(model_name, "math500")
    if not seg_path.exists():
        print(f"[{model_name}] no segments, skip")
        return
    out_path = config.ABL_DIR / f"prefix_position__{model_name}__math500.jsonl"
    done = done_keys(out_path, "task_id")
    problem_texts = load_problem_text("math500")
    gt_lookup = {r["task_id"]: r["ground_truth"]
                 for r in iter_jsonl(config.trace_path(model_name, "math500"))}

    # subsample by problem_id (not by task_id) to get diverse coverage
    correct_by_problem: dict[str, list] = {}
    for r in iter_jsonl(seg_path):
        if not r["is_correct"] or r["n_steps"] < 4:
            continue
        pid = r["problem_id"]
        if pid not in correct_by_problem:
            correct_by_problem[pid] = r  # take first correct trace per problem
    candidates = list(correct_by_problem.values())
    rng = random.Random(config.SEED)
    rng.shuffle(candidates)
    selected = candidates[:N_TRACES_PER_MODEL]
    todo = [r for r in selected if r["task_id"] not in done]
    if not todo:
        print(f"[{model_name}] prefix ablation all done ({len(selected)} traces)")
        return
    print(f"[{model_name}] prefix ablation on {len(todo)} traces "
          f"({K_POINTS_PER_TRACE} k x {len(STRATEGIES)} strategies = "
          f"{K_POINTS_PER_TRACE * len(STRATEGIES)} judge calls / trace)")

    limiter = ConcurrencyLimiter(config.MAX_CONCURRENT)
    judge_provider, judge_model = config.JUDGE_SELF  # deepseek-chat
    CHUNK = config.MAX_CONCURRENT * 2
    pbar = stqdm(total=len(todo), desc=f"prefix-pos {model_name}")
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        coros = []
        for seg in batch:
            ptext = problem_texts.get(seg["problem_id"])
            gt = gt_lookup.get(seg["task_id"])
            if ptext is None or gt is None:
                continue
            coros.append(run_trace(seg, ptext, gt, judge_provider, judge_model, limiter, out_path))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)
        pbar.update(len(batch))
    pbar.close()


async def main():
    for model_name, _, _ in config.REASONING_MODELS:
        try:
            await run_model(model_name)
        except Exception as e:
            print(f"[{model_name}] FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
