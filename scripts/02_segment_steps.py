"""Phase 2: segment each reasoning trace into discrete steps."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import iter_jsonl, write_jsonl
from utils.segmentation import segment_trace


def process(model_name: str, dataset: str):
    in_path = config.trace_path(model_name, dataset)
    out_path = config.segments_path(model_name, dataset)
    if not in_path.exists():
        return
    out_records = []
    for r in iter_jsonl(in_path):
        if r.get("error"):
            continue
        steps = segment_trace(r.get("reasoning_trace") or r.get("final_content", ""))
        token_counts = [len(s.split()) for s in steps]
        out_records.append({
            "task_id": r["task_id"],
            "problem_id": r["problem_id"],
            "sample_idx": r["sample_idx"],
            "level": r.get("level"),
            "subject": r.get("subject"),
            "dataset": r["dataset"],
            "is_correct": r["is_correct"],
            "n_steps": len(steps),
            "step_tokens": token_counts,
            "total_tokens_words": sum(token_counts),
            "completion_tokens": r.get("completion_tokens", 0),
            "reasoning_tokens": r.get("reasoning_tokens", 0),
            "steps": steps,
        })
    write_jsonl(out_path, out_records)
    print(f"[{model_name}/{dataset}] segmented {len(out_records)} traces -> {out_path.name}")


def main():
    for model_name, _, _ in config.REASONING_MODELS:
        for dataset in ["math500", "gsm8k", "svamp", "aqua"]:
            process(model_name, dataset)


if __name__ == "__main__":
    main()
