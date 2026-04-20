"""Phase 5-6: aggregate stats + practical mitigation strategies.

Reads outputs from phases 1-4 and produces a single results dictionary saved
to outputs/results_intermediate.json. Phase 07 produces the final results.json.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import iter_jsonl, read_jsonl
from utils.answer_extraction import extract_answer, answers_equal


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else None


# ---------------- Truncation aggregates ---------------- #

def aggregate_truncation(model_name: str, dataset: str, judge_label: str):
    path = config.truncation_path(model_name, dataset, judge_label)
    if not path.exists():
        return None
    rhos, rho_Ls, n_steps_list = [], [], []
    by_level = defaultdict(list)
    by_level_lengths = defaultdict(list)
    positional = []  # (relative_pos, is_redundant)
    for r in iter_jsonl(path):
        N = r["n_steps"]
        cp = r.get("critical_point")
        if not cp:
            continue
        rho = 1 - cp / N
        step_tokens = r["step_tokens"]
        nec_tokens = sum(step_tokens[:cp])
        total_tokens = sum(step_tokens)
        rho_L = 1 - nec_tokens / total_tokens if total_tokens else 0
        rhos.append(rho)
        rho_Ls.append(rho_L)
        n_steps_list.append(N)
        if r.get("level") is not None:
            by_level[int(r["level"])].append(rho)
            by_level_lengths[int(r["level"])].append((total_tokens, nec_tokens))
        # positional info: a step is "redundant" if its index >= cp (since cp suffices)
        for i in range(N):
            positional.append((i / max(N - 1, 1), 1 if i >= cp else 0))
    if not rhos:
        return None
    # bootstrap CI helper
    def boot_ci(vals, n_boot=1000, alpha=0.05):
        if len(vals) < 2:
            return (safe_mean(vals), safe_mean(vals))
        arr = np.array(vals, dtype=float)
        rng = np.random.default_rng(config.SEED)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        means = arr[idx].mean(axis=1)
        return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))

    return {
        "rho": safe_mean(rhos),
        "rho_L": safe_mean(rho_Ls),
        "avg_steps": safe_mean(n_steps_list),
        "n_traces": len(rhos),
        "by_level_rho": {lvl: safe_mean(v) for lvl, v in by_level.items()},
        "by_level_ci": {lvl: boot_ci(v) for lvl, v in by_level.items()},
        "by_level_n": {lvl: len(v) for lvl, v in by_level.items()},
        "by_level_lengths": {
            lvl: {
                "avg_total": safe_mean([t for t, _ in v]),
                "avg_necessary": safe_mean([n for _, n in v]),
                "ratio": safe_mean([t / n if n else None for t, n in v]),
            }
            for lvl, v in by_level_lengths.items()
        },
        "positional": positional,
    }


# ---------------- Variance analysis (Phase 5) ---------------- #

def variance_analysis(model_name: str, dataset: str):
    path = config.trace_path(model_name, dataset)
    if not path.exists():
        return None
    by_problem = defaultdict(list)
    for r in iter_jsonl(path):
        if r.get("error"):
            continue
        by_problem[r["problem_id"]].append(r)
    cvs, mean_lengths, ratios = [], [], []
    for pid, samples in by_problem.items():
        correct = [s for s in samples if s["is_correct"]]
        if len(correct) < 2:
            continue
        if len(correct) / len(samples) < 0.8:
            continue
        lengths = [len(((s.get("reasoning_trace") or "") + " " + (s.get("final_content") or "")).split()) for s in correct]
        if len(lengths) < 2:
            continue
        m = mean(lengths)
        s = stdev(lengths)
        cvs.append(s / m if m else 0)
        mean_lengths.append(m)
        ratios.append(min(lengths) / max(lengths) if max(lengths) else 0)
    if not cvs:
        return None
    return {
        "mean_length": safe_mean(mean_lengths),
        "mean_cv": safe_mean(cvs),
        "shortest_longest_ratio": safe_mean(ratios),
        "n_problems": len(cvs),
    }


# ---------------- Practical strategies (Phase 6) ---------------- #

def shortest_correct(model_name: str, dataset: str):
    path = config.trace_path(model_name, dataset)
    if not path.exists():
        return None
    by_problem = defaultdict(list)
    for r in iter_jsonl(path):
        if not r.get("error"):
            by_problem[r["problem_id"]].append(r)
    rand_acc, rand_len, sc_acc, sc_len, n = 0, 0, 0, 0, 0
    for pid, samples in by_problem.items():
        # majority vote answer
        preds = [s["predicted_answer"] for s in samples if s["predicted_answer"]]
        if not preds:
            continue
        vote = Counter(preds).most_common(1)[0][0]
        # samples whose predicted_answer == vote
        agreeing = [s for s in samples if answers_equal(s["predicted_answer"], vote)]
        if not agreeing:
            continue
        # ground truth correctness via vote
        gt = samples[0]["ground_truth"]
        is_correct = answers_equal(vote, gt)
        # random correct trace baseline = mean length over correct samples
        correct_samples = [s for s in samples if s["is_correct"]]
        if not correct_samples:
            continue
        rand_acc += is_correct
        rand_len += mean([len(((s.get("reasoning_trace") or "") + " " + (s.get("final_content") or "")).split()) for s in correct_samples])
        # shortest correct
        shortest = min(correct_samples, key=lambda s: len(((s.get("reasoning_trace") or "") + " " + (s.get("final_content") or "")).split()))
        sc_acc += is_correct
        sc_len += shortest.get("completion_tokens") or len((shortest.get("reasoning_trace") or "").split())
        n += 1
    if n == 0:
        return None
    return {
        "random_correct": {"accuracy": rand_acc / n, "avg_length": rand_len / n},
        "shortest_correct": {"accuracy": sc_acc / n, "avg_length": sc_len / n,
                             "reduction": 1 - (sc_len / rand_len) if rand_len else 0},
        "n_problems": n,
    }


def difficulty_budget(model_name: str):
    path = config.trace_path(model_name, "math500")
    if not path.exists():
        return None
    by_level = defaultdict(list)
    for r in iter_jsonl(path):
        if r.get("error") or r.get("level") is None:
            continue
        ln = len(((r.get("reasoning_trace") or "") + " " + (r.get("final_content") or "")).split())
        by_level[int(r["level"])].append((r, ln))
    medians = {lvl: median([ln for _, ln in v]) for lvl, v in by_level.items() if v}
    out = {"unconstrained": _budget_eval(by_level, lambda lvl: float("inf"))}
    for alpha in [2.0, 1.5, 1.0]:
        out[f"alpha_{alpha}"] = _budget_eval(by_level, lambda lvl, a=alpha: a * medians.get(lvl, 0))
    return out


def _budget_eval(by_level, budget_fn):
    accs, tokens, n = 0, 0, 0
    for lvl, items in by_level.items():
        b = budget_fn(lvl)
        for r, ln in items:
            if ln <= b:
                accs += 1 if r["is_correct"] else 0
                tokens += ln
                n += 1
    return {"accuracy": accs / n if n else 0, "avg_tokens": tokens / n if n else 0, "n": n}


def convergence_detection(model_name: str, dataset: str = "math500"):
    """Heuristic: if two consecutive steps both surface the same numeric answer, stop."""
    seg_path = config.segments_path(model_name, dataset)
    trace_path = config.trace_path(model_name, dataset)
    if not seg_path.exists():
        return None
    gt_lookup = {r["task_id"]: r["ground_truth"] for r in iter_jsonl(trace_path)}
    full_acc, conv_acc, full_tok, conv_tok, n = 0, 0, 0, 0, 0
    num_re = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
    for r in iter_jsonl(seg_path):
        steps = r["steps"]
        token_counts = r["step_tokens"]
        if not steps:
            continue
        full_total = sum(token_counts)
        gt = gt_lookup.get(r["task_id"])
        # full prediction = whatever the model finally said (we approximate as r["is_correct"])
        full_correct = bool(r["is_correct"])
        # convergence stopping
        prev_nums, stop_at = None, None
        for i, s in enumerate(steps):
            nums = num_re.findall(s)
            if prev_nums and nums and prev_nums[-1] == nums[-1]:
                stop_at = i + 1
                break
            prev_nums = nums or prev_nums
        if stop_at is None:
            stop_at = len(steps)
        cand = (num_re.findall("\n".join(steps[:stop_at])) or [None])[-1]
        conv_correct = bool(answers_equal(cand, gt)) if gt else False
        full_acc += full_correct
        conv_acc += conv_correct
        full_tok += full_total
        conv_tok += sum(token_counts[:stop_at])
        n += 1
    if n == 0:
        return None
    return {
        "full": {"accuracy": full_acc / n, "avg_tokens": full_tok / n},
        "convergence": {"accuracy": conv_acc / n, "avg_tokens": conv_tok / n,
                        "savings": 1 - conv_tok / full_tok if full_tok else 0},
        "n_problems": n,
    }


# ---------------- Step taxonomy from ablation ---------------- #

def step_taxonomy(model_name: str):
    path = config.ablation_path(model_name, "math500")
    if not path.exists():
        return None
    by_level = defaultdict(lambda: {"critical": 0, "redundant": 0})
    overall = {"critical": 0, "redundant": 0}
    for r in iter_jsonl(path):
        lvl = r.get("level")
        for c in r["step_classifications"]:
            cls = c["classification"]
            overall[cls] += 1
            if lvl is not None:
                by_level[int(lvl)][cls] += 1
    def fracs(d):
        tot = d["critical"] + d["redundant"]
        return {
            "critical_frac": d["critical"] / tot if tot else 0,
            "redundant_frac": d["redundant"] / tot if tot else 0,
            "facilitative_frac": 0.0,  # leave-one-out cannot detect facilitative
        }
    return {
        "overall": fracs(overall),
        "by_level": {lvl: fracs(d) for lvl, d in sorted(by_level.items())},
    }


# ---------------- Theory fit ---------------- #

def theory_fit(by_level_rho: dict[int, float]):
    """Fit rho(d) = alpha / (d + beta)."""
    from scipy.optimize import curve_fit
    if not by_level_rho or len(by_level_rho) < 3:
        return None
    levels = sorted(by_level_rho.keys())
    xs = np.array(levels, dtype=float)
    ys = np.array([by_level_rho[l] for l in levels], dtype=float)
    def fn(d, a, b):
        return a / (d + b)
    try:
        popt, _ = curve_fit(fn, xs, ys, p0=[1.0, 1.0], maxfev=5000)
        pred = fn(xs, *popt)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else 0
        return {"alpha": float(popt[0]), "beta": float(popt[1]), "r_squared": r2}
    except Exception as e:
        return {"error": str(e)}


# ---------------- Main aggregation ---------------- #

def main():
    summary = {
        "table1_overall_redundancy": {},
        "table2_crossjudge": {},
        "table3_length_by_difficulty": {},
        "table4_variance": {},
        "table5_shortest_correct": {},
        "table6_budget": {},
        "table7_early_stop": {},
        "table8_theory_fit": {},
        "figure_data": {
            "fig2_difficulty_redundancy": {},
            "fig3_step_taxonomy": {},
            "fig4_positional_redundancy": {},
        },
    }
    for model_name, _, _ in config.REASONING_MODELS:
        summary["table1_overall_redundancy"][model_name] = {}
        for dataset in ["math500", "gsm8k"]:
            agg = aggregate_truncation(model_name, dataset, "self")
            if agg:
                summary["table1_overall_redundancy"][model_name][dataset] = {
                    "rho": agg["rho"], "rho_L": agg["rho_L"],
                    "avg_steps": agg["avg_steps"], "n_traces": agg["n_traces"],
                }
                if dataset == "math500":
                    summary["table3_length_by_difficulty"][model_name] = agg["by_level_lengths"]
                    lvls_sorted = sorted(agg["by_level_rho"].keys())
                    summary["figure_data"]["fig2_difficulty_redundancy"][model_name] = {
                        "levels": lvls_sorted,
                        "rho": [agg["by_level_rho"][k] for k in lvls_sorted],
                        "ci_lo": [agg["by_level_ci"][k][0] for k in lvls_sorted],
                        "ci_hi": [agg["by_level_ci"][k][1] for k in lvls_sorted],
                        "n_per_level": [agg["by_level_n"][k] for k in lvls_sorted],
                    }
                    summary["table8_theory_fit"][model_name] = theory_fit(agg["by_level_rho"])
                    if agg["positional"]:
                        bins = np.linspace(0, 1, 11)
                        positions = np.array([p for p, _ in agg["positional"]])
                        labels = np.array([y for _, y in agg["positional"]])
                        digit = np.digitize(positions, bins) - 1
                        digit = np.clip(digit, 0, len(bins) - 2)
                        prob = []
                        for k in range(len(bins) - 1):
                            mask = digit == k
                            prob.append(float(labels[mask].mean()) if mask.any() else 0)
                        pos_data = {
                            "positions": [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)],
                            "redundant_prob": prob,
                        }
                        # keep single-model backward-compat key (overwritten by last model),
                        # and also store per-model in fig4_positional_redundancy_multi
                        summary["figure_data"]["fig4_positional_redundancy"] = pos_data
                        if "fig4_positional_redundancy_multi" not in summary["figure_data"]:
                            summary["figure_data"]["fig4_positional_redundancy_multi"] = {}
                        summary["figure_data"]["fig4_positional_redundancy_multi"][model_name] = pos_data

        # cross-judge (R1 + math500 only)
        if model_name == "deepseek_r1":
            cross = aggregate_truncation(model_name, "math500", "cross")
            self_ = aggregate_truncation(model_name, "math500", "self")
            if cross and self_:
                summary["table2_crossjudge"] = {
                    "self_rho": self_["rho"], "self_rho_L": self_["rho_L"],
                    "cross_rho": cross["rho"], "cross_rho_L": cross["rho_L"],
                    "n_self": self_["n_traces"], "n_cross": cross["n_traces"],
                }
        # variance
        v = variance_analysis(model_name, "math500")
        if v:
            summary["table4_variance"][model_name] = v
        # shortest-correct
        sc = shortest_correct(model_name, "math500")
        if sc:
            summary["table5_shortest_correct"][model_name] = sc
        # budget
        b = difficulty_budget(model_name)
        if b:
            summary["table6_budget"][model_name] = b
        # early stop
        es = convergence_detection(model_name)
        if es:
            summary["table7_early_stop"][model_name] = es
        # taxonomy
        tax = step_taxonomy(model_name)
        if tax:
            summary["figure_data"]["fig3_step_taxonomy"][model_name] = tax

    out_path = config.OUT_DIR / "results_intermediate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
