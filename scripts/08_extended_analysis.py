"""Phase 8: extended offline analyses for appendix / extra experiments.

Everything in this file consumes data already on disk; no new API calls.
Outputs: outputs/extended_results.json
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.io_utils import iter_jsonl
from utils.answer_extraction import answers_equal


def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return float(mean(xs)) if xs else None


def _boot_ci(xs, n_boot=1000, alpha=0.05):
    if len(xs) < 2:
        return (None, None)
    arr = np.array(xs, dtype=float)
    rng = np.random.default_rng(config.SEED)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def load_subject_map():
    """MATH-500 problem_id -> subject."""
    out = {}
    for r in iter_jsonl(config.DATA_DIR / "math500.jsonl"):
        out[r["id"]] = r.get("subject") or r.get("type") or "Unknown"
    return out


def load_level_map():
    out = {}
    for r in iter_jsonl(config.DATA_DIR / "math500.jsonl"):
        out[r["id"]] = r.get("level")
    return out


# ---------------- Per-subject redundancy (per model) ---------------- #

def per_subject_redundancy_single(model_name: str):
    trunc_path = config.truncation_path(model_name, "math500", "self")
    if not trunc_path.exists():
        return None
    subj = load_subject_map()
    by_subj = defaultdict(list)
    by_subj_tok = defaultdict(list)
    by_subj_len = defaultdict(list)
    for r in iter_jsonl(trunc_path):
        cp = r.get("critical_point")
        if not cp:
            continue
        N = r["n_steps"]
        rho = 1 - cp / N
        s = subj.get(r["problem_id"], "Unknown")
        by_subj[s].append(rho)
        total_tok = sum(r["step_tokens"])
        nec_tok = sum(r["step_tokens"][:cp])
        by_subj_tok[s].append(1 - nec_tok / total_tok if total_tok else 0)
        by_subj_len[s].append(total_tok)
    out = {}
    for s in by_subj:
        lo, hi = _boot_ci(by_subj[s])
        out[s] = {
            "rho": _safe_mean(by_subj[s]),
            "rho_L": _safe_mean(by_subj_tok[s]),
            "avg_total_tokens": _safe_mean(by_subj_len[s]),
            "n": len(by_subj[s]),
            "ci_lo": lo, "ci_hi": hi,
        }
    return out


def per_subject_redundancy():
    return {
        m: per_subject_redundancy_single(m)
        for m, _, _ in config.REASONING_MODELS
    }


# ---------------- k* distribution (per model) ---------------- #

def k_star_distribution_single(model_name: str):
    trunc_path = config.truncation_path(model_name, "math500", "self")
    if not trunc_path.exists():
        return None
    ks, Ns, ratios = [], [], []
    for r in iter_jsonl(trunc_path):
        cp = r.get("critical_point")
        if not cp:
            continue
        ks.append(cp)
        Ns.append(r["n_steps"])
        ratios.append(cp / r["n_steps"])
    if not ks:
        return None
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(ratios, bins=bins)
    return {
        "mean_k_star": _safe_mean(ks),
        "median_k_star": float(median(ks)),
        "mean_k_star_over_N": _safe_mean(ratios),
        "bins": bins.tolist(),
        "hist": hist.tolist(),
        "n": len(ks),
    }


def k_star_distribution():
    return {m: k_star_distribution_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- Length vs correctness ---------------- #

def length_vs_correctness_single(model_name: str):
    trace_path = config.trace_path(model_name, "math500")
    if not trace_path.exists():
        return None
    rows = []
    for r in iter_jsonl(trace_path):
        if r.get("error"):
            continue
        ln = len(((r.get("reasoning_trace") or "") + " " + (r.get("final_content") or "")).split())
        rows.append((ln, int(r["is_correct"])))
    if not rows:
        return None
    lens = np.array([r[0] for r in rows])
    corr = np.array([r[1] for r in rows])
    order = np.argsort(lens)
    lens_s = lens[order]
    corr_s = corr[order]
    n = len(rows)
    bucket_size = max(1, n // 10)
    out = []
    for i in range(0, n, bucket_size):
        chunk_c = corr_s[i:i + bucket_size]
        chunk_l = lens_s[i:i + bucket_size]
        if len(chunk_c) == 0:
            continue
        out.append({
            "bucket": i // bucket_size,
            "mean_length": float(chunk_l.mean()),
            "accuracy": float(chunk_c.mean()),
            "n": int(len(chunk_c)),
        })
    # overall correlation
    corrcoef = float(np.corrcoef(lens, corr)[0, 1]) if n > 1 else 0.0
    return {"buckets": out, "pearson_r": corrcoef, "n": n}


def length_vs_correctness():
    return {m: length_vs_correctness_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- Judge agreement ---------------- #

def judge_agreement_single(model_name: str):
    self_p = config.truncation_path(model_name, "math500", "self")
    cross_p = config.truncation_path(model_name, "math500", "cross")
    if not (self_p.exists() and cross_p.exists()):
        return None
    self_cp = {r["task_id"]: r.get("critical_point") for r in iter_jsonl(self_p) if r.get("critical_point")}
    cross_cp = {r["task_id"]: r.get("critical_point") for r in iter_jsonl(cross_p) if r.get("critical_point")}
    common = set(self_cp) & set(cross_cp)
    if not common:
        return None
    diffs = [self_cp[t] - cross_cp[t] for t in common]
    exact = sum(1 for d in diffs if d == 0)
    within1 = sum(1 for d in diffs if abs(d) <= 1)
    self_earlier = sum(1 for d in diffs if d < 0)
    cross_earlier = sum(1 for d in diffs if d > 0)
    return {
        "n": len(common),
        "mean_diff_self_minus_cross": _safe_mean(diffs),
        "exact_match_frac": exact / len(common),
        "within_1_frac": within1 / len(common),
        "self_earlier_frac": self_earlier / len(common),
        "cross_earlier_frac": cross_earlier / len(common),
    }


def judge_agreement():
    return {m: judge_agreement_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- Step length over position ---------------- #

def step_length_by_position_single(model_name: str):
    seg_path = config.segments_path(model_name, "math500")
    if not seg_path.exists():
        return None
    buckets = [[] for _ in range(10)]
    for r in iter_jsonl(seg_path):
        if not r["is_correct"]:
            continue
        toks = r["step_tokens"]
        N = len(toks)
        if N < 2:
            continue
        for i, t in enumerate(toks):
            p = i / (N - 1)
            b = min(int(p * 10), 9)
            buckets[b].append(t)
    return {
        "bins": [(i / 10, (i + 1) / 10) for i in range(10)],
        "mean_tokens": [_safe_mean(b) for b in buckets],
        "counts": [len(b) for b in buckets],
    }


def step_length_by_position():
    return {m: step_length_by_position_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- Critical step positions (from ablation) ---------------- #

def critical_step_positions_single(model_name: str):
    abl_path = config.ablation_path(model_name, "math500")
    if not abl_path.exists():
        return {"positions": [], "histogram": [0] * 10, "n": 0}
    positions = []  # relative positions of critical steps
    for r in iter_jsonl(abl_path):
        N = r["n_steps"]
        if N < 2:
            continue
        for c in r["step_classifications"]:
            if c["classification"] == "critical":
                positions.append(c["step_index"] / (N - 1))
    if not positions:
        return {"positions": [], "histogram": [0] * 10, "n": 0}
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(positions, bins=bins)
    return {
        "n": len(positions),
        "histogram": hist.tolist(),
        "mean_position": _safe_mean(positions),
        "median_position": float(median(positions)),
    }


def critical_step_positions():
    return {m: critical_step_positions_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- Shortest-correct at M ∈ {2, 3} ---------------- #

def shortest_correct_varying_M():
    trace_path = config.trace_path("deepseek_r1", "math500")
    by_problem = defaultdict(list)
    for r in iter_jsonl(trace_path):
        if not r.get("error"):
            by_problem[r["problem_id"]].append(r)
    rng = np.random.default_rng(config.SEED)
    results = {}
    for M in [2, 3]:
        rand_len_t = sc_len_t = n = rand_acc = sc_acc = 0
        for pid, samples in by_problem.items():
            if len(samples) < M:
                continue
            # subsample M
            idxs = rng.choice(len(samples), size=M, replace=False)
            sub = [samples[i] for i in idxs]
            preds = [s["predicted_answer"] for s in sub if s["predicted_answer"]]
            if not preds:
                continue
            vote = Counter(preds).most_common(1)[0][0]
            correct_sub = [s for s in sub if s["is_correct"]]
            if not correct_sub:
                continue
            gt = sub[0]["ground_truth"]
            is_c = answers_equal(vote, gt)
            rand_len_t += mean([s.get("completion_tokens") or 0 for s in correct_sub])
            rand_acc += int(is_c)
            shortest = min(correct_sub, key=lambda s: s.get("completion_tokens") or 0)
            sc_len_t += shortest.get("completion_tokens") or 0
            sc_acc += int(is_c)
            n += 1
        if n:
            results[f"M={M}"] = {
                "random_acc": rand_acc / n,
                "random_len": rand_len_t / n,
                "shortest_acc": sc_acc / n,
                "shortest_len": sc_len_t / n,
                "reduction": 1 - sc_len_t / rand_len_t if rand_len_t else 0,
                "n": n,
            }
    return results


# ---------------- Budget at more alphas ---------------- #

def budget_fine_grained():
    trace_path = config.trace_path("deepseek_r1", "math500")
    by_level = defaultdict(list)
    for r in iter_jsonl(trace_path):
        if r.get("error") or r.get("level") is None:
            continue
        ln = len(((r.get("reasoning_trace") or "") + " " + (r.get("final_content") or "")).split())
        by_level[int(r["level"])].append((r, ln))
    medians = {lvl: median([ln for _, ln in v]) for lvl, v in by_level.items() if v}
    out = {}
    for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, float("inf")]:
        accs = 0
        tokens = 0
        n = 0
        for lvl, items in by_level.items():
            b = alpha * medians[lvl] if alpha != float("inf") else float("inf")
            for r, ln in items:
                if ln <= b:
                    accs += int(r["is_correct"])
                    tokens += ln
                    n += 1
        key = "inf" if alpha == float("inf") else f"{alpha:.2f}"
        out[key] = {
            "accuracy": accs / n if n else 0,
            "avg_tokens": tokens / n if n else 0,
            "n": n,
        }
    return out


# ---------------- Per-problem redundancy variance across samples ---------------- #

def per_problem_rho_variance_single(model_name: str):
    trunc_path = config.truncation_path(model_name, "math500", "self")
    if not trunc_path.exists():
        return None
    by_problem = defaultdict(list)
    for r in iter_jsonl(trunc_path):
        cp = r.get("critical_point")
        if not cp:
            continue
        rho = 1 - cp / r["n_steps"]
        by_problem[r["problem_id"]].append(rho)
    multi = {pid: v for pid, v in by_problem.items() if len(v) >= 2}
    if not multi:
        return None
    stds = [pstdev(v) for v in multi.values()]
    return {
        "n_multi_sample_problems": len(multi),
        "mean_within_problem_rho_std": _safe_mean(stds),
        "mean_within_problem_rho_range": _safe_mean([max(v) - min(v) for v in multi.values()]),
    }


def per_problem_rho_variance():
    return {m: per_problem_rho_variance_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- k* depends on K ---------------- #

def critical_length_vs_difficulty_single(model_name: str):
    """Avg k* (in steps) by difficulty level."""
    trunc_path = config.truncation_path(model_name, "math500", "self")
    if not trunc_path.exists():
        return None
    lvls = load_level_map()
    by_lvl = defaultdict(list)
    by_lvl_n = defaultdict(list)
    for r in iter_jsonl(trunc_path):
        cp = r.get("critical_point")
        if cp is None:
            continue
        l = lvls.get(r["problem_id"])
        if l is None:
            continue
        by_lvl[int(l)].append(cp)
        by_lvl_n[int(l)].append(r["n_steps"])
    return {
        lvl: {
            "avg_k_star_steps": _safe_mean(by_lvl[lvl]),
            "avg_N_steps": _safe_mean(by_lvl_n[lvl]),
            "n": len(by_lvl[lvl]),
        }
        for lvl in sorted(by_lvl)
    }


def critical_length_vs_difficulty():
    return {m: critical_length_vs_difficulty_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- GSM8K per-subject (trivial: 1 subject) but include overall ---------------- #

def practical_strategy_baselines(model_name: str, dataset: str = "math500"):
    """Compare realistic deployment strategies (no ground-truth leakage).

    Strategies (all operate on M samples per problem):
      - single: return first generated sample, ignore others (M=1 baseline)
      - shortest_any: return shortest sample (length picked blind)
      - longest_any: return longest sample (anti-baseline)
      - median_any: return median-length sample
      - majority_vote: emit majority-vote answer; length = mean of all samples
      - majority_shortest (ours): majority vote + shortest-agreeing
      - majority_first (ablation): majority vote + first-agreeing
      - majority_median (ablation): majority vote + median-agreeing

    Accuracy is: does the returned answer match ground truth?
    Length is: tokens in the returned trace (or mean for strategies that
               emit only an answer without a specific trace, i.e. majority_vote).
    """
    from utils.answer_extraction import answers_equal
    trace_path = config.trace_path(model_name, dataset)
    if not trace_path.exists():
        return None
    by_problem = defaultdict(list)
    for r in iter_jsonl(trace_path):
        if not r.get("error"):
            by_problem[r["problem_id"]].append(r)

    def ln(s):
        return len(((s.get("reasoning_trace") or "") + " " + (s.get("final_content") or "")).split())

    acc = defaultdict(int)
    length = defaultdict(float)
    n = 0
    for pid, samples in by_problem.items():
        if len(samples) < 1:
            continue
        samples = sorted(samples, key=lambda s: s.get("sample_idx", 0))
        gt = samples[0]["ground_truth"]
        preds = [s.get("predicted_answer") for s in samples]
        non_null_preds = [p for p in preds if p]
        vote = Counter(non_null_preds).most_common(1)[0][0] if non_null_preds else None

        # single (first)
        s = samples[0]
        acc["single"] += int(s["is_correct"])
        length["single"] += ln(s)
        # shortest_any
        s = min(samples, key=ln)
        acc["shortest_any"] += int(answers_equal(s.get("predicted_answer"), gt))
        length["shortest_any"] += ln(s)
        # longest_any
        s = max(samples, key=ln)
        acc["longest_any"] += int(answers_equal(s.get("predicted_answer"), gt))
        length["longest_any"] += ln(s)
        # median_any
        sorted_by_len = sorted(samples, key=ln)
        s = sorted_by_len[len(sorted_by_len) // 2]
        acc["median_any"] += int(answers_equal(s.get("predicted_answer"), gt))
        length["median_any"] += ln(s)
        # majority vote (length = mean of all)
        if vote is not None:
            acc["majority_vote"] += int(answers_equal(vote, gt))
            length["majority_vote"] += sum(ln(s) for s in samples) / len(samples)
        # majority + shortest (OUR method)
        agreeing = [s for s in samples if answers_equal(s.get("predicted_answer"), vote)]
        if agreeing:
            chosen = min(agreeing, key=ln)
            acc["majority_shortest"] += int(answers_equal(vote, gt))
            length["majority_shortest"] += ln(chosen)
            # majority + first
            chosen = min(agreeing, key=lambda s: s.get("sample_idx", 0))
            acc["majority_first"] += int(answers_equal(vote, gt))
            length["majority_first"] += ln(chosen)
            # majority + median
            agreeing_sorted = sorted(agreeing, key=ln)
            chosen = agreeing_sorted[len(agreeing_sorted) // 2]
            acc["majority_median"] += int(answers_equal(vote, gt))
            length["majority_median"] += ln(chosen)
        n += 1
    if n == 0:
        return None
    return {
        k: {"accuracy": acc[k] / n, "avg_length": length[k] / n, "n": n}
        for k in acc
    }


def gsm8k_summary_single(model_name: str):
    trunc_path = config.truncation_path(model_name, "gsm8k", "self")
    if not trunc_path.exists():
        return None
    rhos, cps, ns = [], [], []
    for r in iter_jsonl(trunc_path):
        cp = r.get("critical_point")
        if cp is None:
            continue
        rhos.append(1 - cp / r["n_steps"])
        cps.append(cp)
        ns.append(r["n_steps"])
    if not rhos:
        return None
    lo, hi = _boot_ci(rhos)
    return {
        "rho": _safe_mean(rhos),
        "avg_k_star": _safe_mean(cps),
        "avg_N": _safe_mean(ns),
        "n": len(rhos),
        "ci_lo": lo, "ci_hi": hi,
    }


def gsm8k_summary():
    return {m: gsm8k_summary_single(m) for m, _, _ in config.REASONING_MODELS}


# ---------------- main ---------------- #

def prefix_position_aggregate():
    """Aggregate Phase 9 prefix-position ablation output into a multi-curve
    dataset: for each (model, strategy), report accuracy at 10 relative k
    buckets.
    """
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        path = config.ABL_DIR / f"prefix_position__{model_name}__math500.jsonl"
        if not path.exists():
            continue
        records = list(iter_jsonl(path))
        if not records:
            continue
        # Accumulate per (strategy, relative_bucket)
        from collections import defaultdict as dd
        bucket_accs = dd(lambda: dd(list))  # [strategy][bucket_idx] -> list of 0/1
        for r in records:
            N = r["n_steps"]
            for strat, arr in r["strategies"].items():
                for item in arr:
                    k = item["k"]
                    rel = (k - 1) / max(1, N - 1)
                    b = min(int(rel * 10), 9)
                    bucket_accs[strat][b].append(int(item["correct"]))
        strat_data = {}
        for strat, buckets in bucket_accs.items():
            xs = sorted(buckets.keys())
            rel_positions = [(x + 0.5) / 10 for x in xs]
            accs = [_safe_mean(buckets[x]) for x in xs]
            counts = [len(buckets[x]) for x in xs]
            strat_data[strat] = {
                "rel_positions": rel_positions,
                "accuracy": accs,
                "counts": counts,
            }
        out[model_name] = strat_data
    return out


def dense_main_table():
    """Wide main table: for each (model × dataset × judge), compute many metrics.

    Columns: rho, rho_L, avg_steps, avg_length, avg_k_star, median_k_star,
    critical_frac_loo (from ablation), n_traces.
    """
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        out[model_name] = {}
        for ds in ["math500", "gsm8k"]:
            out[model_name][ds] = {}
            for judge in ["self", "cross"]:
                trunc_path = config.truncation_path(model_name, ds, judge)
                if not trunc_path.exists():
                    continue
                rhos, rho_Ls, steps_list, tok_list, k_list = [], [], [], [], []
                for r in iter_jsonl(trunc_path):
                    cp = r.get("critical_point")
                    if not cp:
                        continue
                    N = r["n_steps"]
                    rhos.append(1 - cp / N)
                    tok = r["step_tokens"]
                    rho_Ls.append(1 - sum(tok[:cp]) / sum(tok) if sum(tok) else 0)
                    steps_list.append(N)
                    tok_list.append(sum(tok))
                    k_list.append(cp)
                if not rhos:
                    continue
                lo, hi = _boot_ci(rhos)
                # Pull LOO critical fraction from the separate ablation file
                crit_frac = None
                abl_path = config.ablation_path(model_name, ds)
                if abl_path.exists():
                    total, crit = 0, 0
                    for r in iter_jsonl(abl_path):
                        total += r["n_steps"]
                        crit += r["critical_count"]
                    if total:
                        crit_frac = crit / total
                out[model_name][ds][judge] = {
                    "rho_pct": 100 * _safe_mean(rhos),
                    "rho_L_pct": 100 * _safe_mean(rho_Ls),
                    "ci_lo_pct": 100 * lo if lo is not None else None,
                    "ci_hi_pct": 100 * hi if hi is not None else None,
                    "avg_steps": _safe_mean(steps_list),
                    "avg_length_words": _safe_mean(tok_list),
                    "avg_k_star": _safe_mean(k_list),
                    "median_k_star": float(median(k_list)),
                    "critical_frac_loo_pct": 100 * crit_frac if crit_frac is not None else None,
                    "n": len(rhos),
                }
    return out


def multi_model_main_table():
    """Headline multi-model × multi-benchmark redundancy."""
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        out[model_name] = {}
        for ds in ["math500", "gsm8k"]:
            trunc = config.truncation_path(model_name, ds, "self")
            if not trunc.exists():
                continue
            rhos, rho_Ls, ns = [], [], []
            for r in iter_jsonl(trunc):
                cp = r.get("critical_point")
                if not cp:
                    continue
                N = r["n_steps"]
                rhos.append(1 - cp / N)
                tok = r["step_tokens"]
                rho_Ls.append(1 - sum(tok[:cp]) / sum(tok) if sum(tok) else 0)
                ns.append(N)
            if not rhos:
                continue
            lo, hi = _boot_ci(rhos)
            out[model_name][ds] = {
                "rho": _safe_mean(rhos),
                "rho_L": _safe_mean(rho_Ls),
                "avg_steps": _safe_mean(ns),
                "n": len(rhos),
                "ci_lo": lo, "ci_hi": hi,
            }
    return out


def multi_model_by_level():
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        trunc = config.truncation_path(model_name, "math500", "self")
        if not trunc.exists():
            continue
        by_level = defaultdict(list)
        for r in iter_jsonl(trunc):
            cp = r.get("critical_point")
            if not cp or r.get("level") is None:
                continue
            by_level[int(r["level"])].append(1 - cp / r["n_steps"])
        out[model_name] = {
            lvl: {
                "rho": _safe_mean(v),
                "ci": _boot_ci(v),
                "n": len(v),
            } for lvl, v in sorted(by_level.items())
        }
    return out


def length_accuracy_per_level(model_name: str):
    """Decile analysis stratified by MATH-500 difficulty level."""
    trace_path = config.trace_path(model_name, "math500")
    if not trace_path.exists():
        return None
    rows_per_level = defaultdict(list)
    for r in iter_jsonl(trace_path):
        if r.get("error") or r.get("level") is None:
            continue
        ln = len(((r.get("reasoning_trace") or "") + " " + (r.get("final_content") or "")).split())
        rows_per_level[int(r["level"])].append((ln, int(r["is_correct"])))
    out = {}
    for lvl, rows in rows_per_level.items():
        if len(rows) < 5:
            continue
        lens = np.array([r[0] for r in rows])
        corr = np.array([r[1] for r in rows])
        order = np.argsort(lens)
        lens_s, corr_s = lens[order], corr[order]
        n = len(rows); b = max(1, n // 5)  # 5 buckets per level
        mean_l, accs = [], []
        for i in range(0, n, b):
            cl = lens_s[i:i + b]; cc = corr_s[i:i + b]
            if len(cl) == 0:
                continue
            mean_l.append(float(cl.mean()))
            accs.append(float(cc.mean()))
        out[lvl] = {
            "mean_length": mean_l,
            "accuracy": accs,
            "n": n,
            "pearson_r": float(np.corrcoef(lens, corr)[0, 1]) if n > 1 and lens.std() > 0 else None,
        }
    return out


def k_star_ecdf_data():
    """ECDF of k*/N per (model, dataset). Also report P50 and P90."""
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        out[model_name] = {}
        for ds in ["math500", "gsm8k"]:
            trunc_path = config.truncation_path(model_name, ds, "self")
            if not trunc_path.exists():
                continue
            ratios = []
            for r in iter_jsonl(trunc_path):
                cp = r.get("critical_point")
                if not cp:
                    continue
                ratios.append(cp / r["n_steps"])
            if not ratios:
                continue
            arr = np.array(sorted(ratios))
            x = np.concatenate([[0], arr, [1]])
            y = np.concatenate([[0], np.linspace(1 / len(arr), 1, len(arr)), [1]])
            out[model_name][ds] = {
                "x": x.tolist(),
                "y": y.tolist(),
                "p50": float(np.quantile(arr, 0.5)),
                "p90": float(np.quantile(arr, 0.9)),
                "n": len(arr),
            }
    return out


def positional_redundancy_by_level():
    """Per-level positional redundancy curves (one per difficulty level)."""
    out = {}
    for model_name, _, _ in config.REASONING_MODELS:
        trunc_path = config.truncation_path(model_name, "math500", "self")
        if not trunc_path.exists():
            continue
        by_level_positions = defaultdict(list)  # level -> list of (rel_pos, is_redundant)
        for r in iter_jsonl(trunc_path):
            cp = r.get("critical_point")
            if not cp or r.get("level") is None:
                continue
            lvl = int(r["level"])
            N = r["n_steps"]
            for i in range(N):
                by_level_positions[lvl].append((i / max(N - 1, 1), 1 if i >= cp else 0))
        model_out = {}
        bins = np.linspace(0, 1, 11)
        for lvl, pts in by_level_positions.items():
            positions = np.array([p for p, _ in pts])
            labels = np.array([y for _, y in pts])
            digit = np.clip(np.digitize(positions, bins) - 1, 0, len(bins) - 2)
            prob = []
            for k in range(len(bins) - 1):
                mask = digit == k
                prob.append(float(labels[mask].mean()) if mask.any() else 0)
            model_out[lvl] = {
                "positions": [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)],
                "redundant_prob": prob,
                "n_steps_total": len(pts),
            }
        out[model_name] = model_out
    return out


def subject_length_rho_scatter():
    """Per-subject scatter data: (avg_length, rho, rho_ci) for each subject per model."""
    per_subj = per_subject_redundancy()
    out = {}
    for model_name, subj_data in per_subj.items():
        if not subj_data:
            continue
        rows = []
        for subject, d in subj_data.items():
            rows.append({
                "subject": subject,
                "avg_length": d["avg_total_tokens"],
                "rho": d["rho"],
                "rho_pct": 100 * d["rho"],
                "ci_lo_pct": 100 * d["ci_lo"],
                "ci_hi_pct": 100 * d["ci_hi"],
                "n": d["n"],
            })
        out[model_name] = rows
    return out


def main():
    out = {
        "length_accuracy_per_level": {
            m: length_accuracy_per_level(m) for m, _, _ in config.REASONING_MODELS
        },
        "k_star_ecdf": k_star_ecdf_data(),
        "positional_redundancy_by_level": positional_redundancy_by_level(),
        "subject_length_rho_scatter": subject_length_rho_scatter(),
        "dense_main_table": dense_main_table(),
        "multi_model_main_table": multi_model_main_table(),
        "multi_model_by_level": multi_model_by_level(),
        "per_subject_redundancy": per_subject_redundancy(),
        "k_star_distribution": k_star_distribution(),
        "length_vs_correctness": length_vs_correctness(),
        "judge_agreement": judge_agreement(),
        "step_length_by_position": step_length_by_position(),
        "critical_step_positions": critical_step_positions(),
        "shortest_correct_varying_M": shortest_correct_varying_M(),
        "budget_fine_grained": budget_fine_grained(),
        "per_problem_rho_variance": per_problem_rho_variance(),
        "critical_length_vs_difficulty": critical_length_vs_difficulty(),
        "practical_strategy_baselines": {
            m: practical_strategy_baselines(m, "math500") for m, _, _ in config.REASONING_MODELS
        },
        "gsm8k_summary": gsm8k_summary(),
        "prefix_position_aggregate": prefix_position_aggregate(),
    }
    path = config.OUT_DIR / "extended_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    main()
