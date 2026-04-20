"""Download MATH-500 and a 500-problem GSM8K test subset to data/."""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import write_jsonl


def download_math500() -> list[dict]:
    from datasets import load_dataset
    print("[MATH-500] loading HuggingFaceH4/MATH-500 ...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for i, ex in enumerate(ds):
        out.append({
            "id": f"math500_{i:04d}",
            "problem": ex["problem"],
            "solution": ex.get("solution", ""),
            "answer": ex.get("answer", ""),
            "level": ex.get("level"),
            "subject": ex.get("subject", ""),
            "dataset": "math500",
        })
    return out


def download_gsm8k(n: int = 500) -> list[dict]:
    from datasets import load_dataset
    print("[GSM8K] loading openai/gsm8k ...")
    ds = load_dataset("gsm8k", "main", split="test")
    rng = random.Random(config.SEED)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    idx = idx[:n]
    out = []
    for j, i in enumerate(idx):
        ex = ds[i]
        # answer is after '#### '
        ans = ex["answer"].split("####")[-1].strip()
        out.append({
            "id": f"gsm8k_{j:04d}",
            "problem": ex["question"],
            "solution": ex["answer"],
            "answer": ans,
            "level": None,
            "subject": "gsm8k",
            "dataset": "gsm8k",
        })
    return out


def main():
    math = download_math500()
    write_jsonl(config.DATA_DIR / "math500.jsonl", math)
    print(f"  saved {len(math)} MATH-500 problems")

    gsm = download_gsm8k(500)
    write_jsonl(config.DATA_DIR / "gsm8k_500.jsonl", gsm)
    print(f"  saved {len(gsm)} GSM8K problems")


if __name__ == "__main__":
    main()
