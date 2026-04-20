"""Phase 4: leave-one-out step ablation.

For each correct trace, remove step i and ask the judge to produce the answer.
Classify each step as critical (removal flips to wrong) or redundant (still right).
"""
from __future__ import annotations
import asyncio
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.api_clients import chat_async, ConcurrencyLimiter
from utils.answer_extraction import extract_answer, answers_equal
from utils.io_utils import iter_jsonl, append_jsonl, done_keys


JUDGE_PROMPT = (
    "You are evaluating a mathematical reasoning trace (one step has been removed). "
    "Based ONLY on the reasoning provided below, determine the final answer.\n\n"
    "Problem: {problem}\n\n"
    "Reasoning:\n{reasoning}\n\n"
    "Output ONLY the final answer in \\boxed{{}} format."
)


def load_problem_text(dataset: str) -> dict[str, str]:
    path = config.DATA_DIR / ("math500.jsonl" if dataset == "math500" else "gsm8k_500.jsonl")
    return {r["id"]: r["problem"] for r in iter_jsonl(path)}


def select_traces(seg_path: Path, dataset: str) -> list[dict]:
    """Pick a manageable subset balanced by difficulty (MATH) or random (GSM8K)."""
    rng = random.Random(config.SEED)
    correct = [r for r in iter_jsonl(seg_path) if r["is_correct"] and r["n_steps"] >= 2]
    # one trace per problem
    by_problem: dict[str, dict] = {}
    for r in correct:
        by_problem.setdefault(r["problem_id"], r)
    cands = list(by_problem.values())
    if dataset == "math500":
        per_level = config.ABLATION_PROBLEMS_PER_LEVEL
        buckets: dict = defaultdict(list)
        for r in cands:
            buckets[r.get("level")].append(r)
        out = []
        for lvl, items in buckets.items():
            rng.shuffle(items)
            out.extend(items[:per_level])
        return out
    else:
        rng.shuffle(cands)
        return cands[: config.ABLATION_PROBLEMS_PER_LEVEL * 5]


async def judge_one(provider, model, problem_text, reasoning_text, ground_truth):
    res = await chat_async(
        provider=provider,
        model=model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(problem=problem_text, reasoning=reasoning_text)}],
        temperature=0.0,
        max_tokens=config.MAX_TOKENS_JUDGE,
    )
    pred = extract_answer(res.content)
    return bool(answers_equal(pred, ground_truth)), pred


async def ablate_trace(seg, problem_text, gt, judge_provider, judge_model, limiter, out_path):
    steps = seg["steps"]
    N = len(steps)
    classifications = []
    for i in range(N):
        ablated = "\n\n".join(steps[:i] + steps[i + 1:])
        async with limiter:
            try:
                ok, pred = await judge_one(judge_provider, judge_model, problem_text, ablated, gt)
            except Exception as e:
                ok, pred = True, f"ERR:{e}"  # conservative: treat error as redundant
        classifications.append({
            "step_index": i,
            "classification": "critical" if not ok else "redundant",
            "pred": pred,
        })
    n_critical = sum(1 for c in classifications if c["classification"] == "critical")
    append_jsonl(out_path, {
        "task_id": seg["task_id"],
        "problem_id": seg["problem_id"],
        "dataset": seg["dataset"],
        "level": seg.get("level"),
        "n_steps": N,
        "step_classifications": classifications,
        "critical_count": n_critical,
        "redundant_count": N - n_critical,
        "critical_fraction": n_critical / N,
        "redundant_fraction": (N - n_critical) / N,
    })


async def run(model_name: str, dataset: str):
    seg_path = config.segments_path(model_name, dataset)
    if not seg_path.exists():
        return
    out_path = config.ablation_path(model_name, dataset)
    done = done_keys(out_path, "task_id")
    selected = select_traces(seg_path, dataset)
    todo = [r for r in selected if r["task_id"] not in done]
    if not todo:
        print(f"[{model_name}/{dataset}] ablation all done ({len(selected)} traces)")
        return
    print(f"[{model_name}/{dataset}] ablating {len(todo)} traces")
    problem_texts = load_problem_text(dataset)
    gt_lookup = {r["task_id"]: r["ground_truth"] for r in iter_jsonl(config.trace_path(model_name, dataset))}
    judge_provider, judge_model = config.JUDGE_SELF
    limiter = ConcurrencyLimiter(config.MAX_CONCURRENT)
    tasks = [
        ablate_trace(seg, problem_texts[seg["problem_id"]], gt_lookup[seg["task_id"]],
                     judge_provider, judge_model, limiter, out_path)
        for seg in todo if seg["problem_id"] in problem_texts and seg["task_id"] in gt_lookup
    ]
    await atqdm.gather(*tasks, desc=f"abl {model_name}/{dataset}")


async def main():
    for model_name, _, _ in config.REASONING_MODELS:
        for dataset in ["math500", "gsm8k"]:
            await run(model_name, dataset)


if __name__ == "__main__":
    asyncio.run(main())
