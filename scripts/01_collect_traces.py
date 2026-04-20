"""Phase 1: collect M reasoning traces per problem from each reasoning model.

Resumable: skips (problem_id, sample_idx) pairs already in the JSONL output.
Concurrent: bounded by config.MAX_CONCURRENT.
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
from utils.io_utils import read_jsonl, append_jsonl, done_keys


PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, then put your final "
    "answer inside \\boxed{{}}.\n\nProblem: {problem}"
)


DATASET_FILES = {
    "math500": ("math500.jsonl", "N_MATH"),
    "gsm8k": ("gsm8k_500.jsonl", "N_GSM8K"),
    "svamp": ("svamp_100.jsonl", 100),
    "aqua": ("aqua_100.jsonl", 100),
}

def load_problems(dataset: str) -> list[dict]:
    info = DATASET_FILES.get(dataset)
    if info is None:
        return []
    fname, n_or_attr = info
    path = config.DATA_DIR / fname
    if not path.exists():
        return []
    n = getattr(config, n_or_attr) if isinstance(n_or_attr, str) else n_or_attr
    return read_jsonl(path)[:n]


async def collect_one(provider, model, problem, sample_idx, limiter, out_path):
    task_id = f"{problem['id']}__s{sample_idx}"
    async with limiter:
        try:
            res = await chat_async(
                provider=provider,
                model=model,
                messages=[{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem["problem"])}],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS_REASONING,
            )
        except Exception as e:
            append_jsonl(out_path, {"task_id": task_id, "error": str(e), "problem_id": problem["id"]})
            return
    pred = extract_answer(res.content) or extract_answer(res.reasoning)
    is_correct = answers_equal(pred, problem["answer"])
    append_jsonl(out_path, {
        "task_id": task_id,
        "problem_id": problem["id"],
        "sample_idx": sample_idx,
        "problem": problem["problem"],
        "ground_truth": problem["answer"],
        "level": problem.get("level"),
        "subject": problem.get("subject"),
        "dataset": problem["dataset"],
        "reasoning_trace": res.reasoning,
        "final_content": res.content,
        "predicted_answer": pred,
        "is_correct": bool(is_correct),
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "reasoning_tokens": res.reasoning_tokens,
    })


async def run_model_dataset(model_name, provider, model_id, dataset):
    out_path = config.trace_path(model_name, dataset)
    done = done_keys(out_path, "task_id")
    problems = load_problems(dataset)
    limiter = ConcurrencyLimiter(config.MAX_CONCURRENT)
    tasks = []
    for p in problems:
        for s in range(config.M_SAMPLES):
            tid = f"{p['id']}__s{s}"
            if tid in done:
                continue
            tasks.append(collect_one(provider, model_id, p, s, limiter, out_path))
    if not tasks:
        print(f"[{model_name}/{dataset}] all {len(problems)*config.M_SAMPLES} samples already done")
        return
    print(f"[{model_name}/{dataset}] running {len(tasks)} tasks (skipping {len(done)} done)")
    await atqdm.gather(*tasks, desc=f"{model_name}/{dataset}")


ALL_DATASETS = ["math500", "gsm8k", "svamp", "aqua"]

async def main():
    for model_name, provider, model_id in config.REASONING_MODELS:
        for dataset in ALL_DATASETS:
            try:
                await run_model_dataset(model_name, provider, model_id, dataset)
            except Exception as e:
                print(f"[{model_name}/{dataset}] FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
