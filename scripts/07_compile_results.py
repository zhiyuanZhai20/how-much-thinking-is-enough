"""Phase 7: combine intermediate results into the final results.json + case studies.

Also picks 3 representative qualitative case studies from the truncation outputs.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import iter_jsonl


def case_studies():
    """Pick three example traces:
       (1) easy MATH problem with high redundancy
       (2) hard MATH problem with low redundancy
       (3) any trace with very high redundancy (proxy for circular reflection)
    """
    examples = []
    for model_name, _, _ in config.REASONING_MODELS:
        seg = list(iter_jsonl(config.segments_path(model_name, "math500")))
        seg_by_id = {r["task_id"]: r for r in seg}
        trunc = list(iter_jsonl(config.truncation_path(model_name, "math500", "self")))
        if not trunc:
            continue
        easy = [r for r in trunc if r.get("level") in (1, 2) and r.get("critical_point")]
        hard = [r for r in trunc if r.get("level") in (4, 5) and r.get("critical_point")]
        easy.sort(key=lambda r: -(1 - r["critical_point"] / r["n_steps"]))
        hard.sort(key=lambda r: (1 - r["critical_point"] / r["n_steps"]))
        if easy:
            r = easy[0]
            examples.append(("Easy problem over-thinking", model_name, r, seg_by_id.get(r["task_id"])))
        if hard:
            r = hard[0]
            examples.append(("Hard problem efficient reasoning", model_name, r, seg_by_id.get(r["task_id"])))
        # high redundancy = circular reflection proxy
        all_sorted = sorted(trunc, key=lambda r: -(1 - r["critical_point"] / r["n_steps"]) if r.get("critical_point") else 0)
        if len(all_sorted) > 1:
            r = all_sorted[1]
            examples.append(("Circular self-reflection (proxy)", model_name, r, seg_by_id.get(r["task_id"])))
        break  # one model is enough
    return examples


def render_case_study(title, model, t, seg):
    if seg is None:
        return f"### {title} ({model})\n[no segment data]\n\n"
    cp = t["critical_point"]
    N = t["n_steps"]
    rho = 1 - cp / N
    out = [f"### {title} ({model})", f"task_id: {t['task_id']}",
           f"n_steps={N}, critical_point={cp}, redundancy={rho:.2%}", ""]
    for i, s in enumerate(seg["steps"]):
        tag = "[CRITICAL prefix]" if i < cp else "[REDUNDANT]"
        out.append(f"-- step {i+1} {tag} --")
        out.append(s[:600] + ("..." if len(s) > 600 else ""))
        out.append("")
    return "\n".join(out) + "\n"


def _merge_extended_into_figure_data(R):
    """Pull multi-model data from extended_results.json into R['figure_data']."""
    ext_path = config.OUT_DIR / "extended_results.json"
    if not ext_path.exists():
        return R
    E = json.loads(ext_path.read_text(encoding="utf-8"))
    fd = R.setdefault("figure_data", {})

    # Length-accuracy per model from extended.length_vs_correctness
    # (currently computes only on last model processed; we need per-model)
    # Re-compute on the fly for every model here so fig5 has multi-curves.
    from collections import defaultdict
    import numpy as np
    fd["fig5_length_accuracy_multi"] = {}
    for model_name, _, _ in config.REASONING_MODELS:
        tp = config.trace_path(model_name, "math500")
        if not tp.exists():
            continue
        rows = []
        from utils.io_utils import iter_jsonl
        for r in iter_jsonl(tp):
            if r.get("error"):
                continue
            # uniform word-count length (works for both DS and QwQ)
            ln = len(((r.get("reasoning_trace") or "") + " " + (r.get("final_content") or "")).split())
            rows.append((ln, int(r["is_correct"])))
        if not rows:
            continue
        lens = np.array([r[0] for r in rows])
        corr = np.array([r[1] for r in rows])
        order = np.argsort(lens)
        lens_s, corr_s = lens[order], corr[order]
        n = len(rows); b = max(1, n // 10)
        mean_l, accs = [], []
        for i in range(0, n, b):
            cl = lens_s[i:i+b]; cc = corr_s[i:i+b]
            if len(cl) == 0:
                continue
            mean_l.append(float(cl.mean()))
            accs.append(float(cc.mean()))
        fd["fig5_length_accuracy_multi"][model_name] = {
            "mean_length": mean_l,
            "accuracy": accs,
            "pearson_r": float(np.corrcoef(lens, corr)[0, 1]) if n > 1 else 0,
        }

    # Practical strategies
    psb = E.get("practical_strategy_baselines", {})
    if psb:
        fd["fig6_practical_strategies"] = {k: v for k, v in psb.items() if v}

    # Attach the whole extended dictionary too for table generation.
    R["extended"] = E
    return R


def main():
    inter_path = config.OUT_DIR / "results_intermediate.json"
    final_path = config.OUT_DIR / "results.json"
    if not inter_path.exists():
        print("missing results_intermediate.json -- run 05_analysis.py first")
        return
    R = json.loads(inter_path.read_text(encoding="utf-8"))
    R = _merge_extended_into_figure_data(R)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=str)
    print(f"saved {final_path}")

    # case studies
    cs = case_studies()
    cs_path = config.OUT_DIR / "case_studies.txt"
    with open(cs_path, "w", encoding="utf-8") as f:
        for title, model, t, seg in cs:
            f.write(render_case_study(title, model, t, seg))
    print(f"saved {cs_path}")

    # human-readable summary
    print("\n=== summary ===")
    for k, v in R.items():
        if isinstance(v, dict) and v:
            print(f"\n[{k}]")
            print(json.dumps(v, indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()
