"""JSONL append/read helpers with checkpointing."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Iterator


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def done_keys(path: Path, key: str = "task_id") -> set:
    return {r[key] for r in iter_jsonl(path) if key in r}


def trace_length_words(record: dict) -> int:
    """Uniform length metric in words. Works for both DeepSeek-R1 (which also
    has completion_tokens) and QwQ (which does not). Use this consistently
    in all cross-model length comparisons.
    """
    reasoning = record.get("reasoning_trace") or ""
    content = record.get("final_content") or ""
    return len((reasoning + " " + content).split())
