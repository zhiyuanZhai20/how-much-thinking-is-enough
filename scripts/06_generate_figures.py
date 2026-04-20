"""Generate the three PDF figures for the paper."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def load_results():
    path = config.OUT_DIR / "results.json"
    if not path.exists():
        path = config.OUT_DIR / "results_intermediate.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fig_difficulty_redundancy(R):
    """2 panels: (left) rho by difficulty with CI band + theoretical fits;
    (right) avg k* (steps) by difficulty (log scale), showing absolute
    growth of the critical prefix with difficulty."""
    data = R["figure_data"]["fig2_difficulty_redundancy"]
    ext = R.get("extended", {})
    cp_by_lvl = ext.get("critical_length_vs_difficulty", {})
    if not data:
        print("[fig2] no data")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left panel: rho(d) with CIs and fit curves
    for i, (model, d) in enumerate(data.items()):
        levels = np.array(d["levels"])
        rho = np.array([r * 100 for r in d["rho"]])
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        if "ci_lo" in d and "ci_hi" in d:
            lo = np.array([r * 100 for r in d["ci_lo"]])
            hi = np.array([r * 100 for r in d["ci_hi"]])
            ax1.fill_between(levels, lo, hi, color=color, alpha=0.15)
        ax1.plot(levels, rho, marker=marker, linestyle="-",
                 label=model.replace("_", "-"), color=color, lw=2.2, ms=9)
        # annotate n
        if "n_per_level" in d:
            for x, y, n in zip(levels, rho, d["n_per_level"]):
                ax1.annotate("$n{=}" + str(n) + "$", (x, y), textcoords="offset points",
                             xytext=(6, -10 if i == 0 else 10),
                             fontsize=7, color=color)
    ax1.set_xlabel("MATH-500 Difficulty Level $d$")
    ax1.set_ylabel("Redundancy Ratio $\\rho$ (\\%)")
    ax1.set_title("Step-Level Redundancy vs.\\ Difficulty")
    ax1.set_ylim(35, 102)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=8)

    # Right panel: avg k* by difficulty (log scale)
    for i, (model, d) in enumerate(cp_by_lvl.items()):
        if not d:
            continue
        levels = sorted([int(l) for l in d.keys()])
        cps = [d[str(l) if str(l) in d else l]["avg_k_star_steps"] for l in levels]
        Ns = [d[str(l) if str(l) in d else l]["avg_N_steps"] for l in levels]
        color = COLORS[i]
        marker = MARKERS[i]
        ax2.plot(levels, cps, marker=marker, linestyle="-", color=color,
                 lw=2.2, ms=9, label=f"{model.replace('_','-')} $\\bar{{k}}^\\star$")
        ax2.plot(levels, Ns, marker=marker, linestyle="--", color=color,
                 lw=1.4, ms=7, alpha=0.6, label=f"{model.replace('_','-')} $\\bar N$")
    ax2.set_xlabel("MATH-500 Difficulty Level $d$")
    ax2.set_ylabel("Steps (log scale)")
    ax2.set_yscale("log")
    ax2.set_title("Trace Length and Critical Prefix Length")
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8, ncol=1)

    fig.suptitle("Step-Level Redundancy and Critical-Prefix Length vs.\\ Difficulty", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_difficulty_redundancy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig2] saved")


def fig_step_taxonomy(R):
    data = R["figure_data"]["fig3_step_taxonomy"]
    if not data:
        print("[fig3] no data")
        return
    models = list(data.keys())
    # side-by-side stacked bars for up to 2 models
    fig, axes = plt.subplots(1, max(1, len(models)), figsize=(5 * max(1, len(models)), 4), squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        by_level = data[model].get("by_level", {})
        if not by_level:
            ax.set_visible(False)
            continue
        levels = sorted(int(l) for l in by_level.keys())
        def get(l, k):
            return by_level[str(l) if str(l) in by_level else l][k] * 100
        crit = [get(l, "critical_frac") for l in levels]
        redu = [get(l, "redundant_frac") for l in levels]
        ax.bar(levels, crit, label="Critical", color="#d62728")
        ax.bar(levels, redu, bottom=crit, label="Redundant", color="#2ca02c")
        ax.set_xlabel("MATH-500 Difficulty Level")
        ax.set_ylabel("Fraction of Steps (\\%)")
        ax.set_title(model.replace("_", "-"))
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_step_taxonomy.pdf")
    plt.close(fig)
    print("[fig3] saved")


def fig_positional_redundancy(R):
    """2 panels (one per model), each with 5 per-difficulty-level curves
    plus overall curve overlaid."""
    ext = R.get("extended", {})
    by_level = ext.get("positional_redundancy_by_level", {})
    multi_overall = R["figure_data"].get("fig4_positional_redundancy_multi", {})
    if not by_level and not multi_overall:
        return
    models = list(by_level.keys()) if by_level else list(multi_overall.keys())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    level_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for ax, model in zip(axes, models):
        lvl_data = (by_level.get(model) or {})
        for lvl in sorted(lvl_data.keys(), key=lambda x: int(x)):
            d = lvl_data[lvl]
            ax.plot(d["positions"], [p * 100 for p in d["redundant_prob"]],
                    marker="o", linestyle="-", lw=1.3, ms=5,
                    color=level_colors[int(lvl) - 1],
                    label=f"Level {lvl}", alpha=0.8)
        # Overall overlay
        ov = multi_overall.get(model)
        if ov:
            ax.plot(ov["positions"], [p * 100 for p in ov["redundant_prob"]],
                    marker="s", linestyle="-", lw=2.6, ms=8,
                    color="black", label="Overall")
        ax.set_xlabel("Relative position in trace $p = i/N$")
        ax.set_ylabel("P(step $i$ is redundant) (\\%)" if model == models[0] else "")
        ax.set_title(model.replace("_", "-"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("Positional Redundancy Stratified by Difficulty Level", y=1.02)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_positional_redundancy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig4] saved")


def fig_length_accuracy(R):
    """Length-accuracy: 2 panels (one per model), each with 5 per-level curves
    plus the overall curve. Shows that the anti-correlation is driven mostly
    by mixing difficulty levels."""
    ext = R.get("extended", {})
    per_level = ext.get("length_accuracy_per_level", {})
    overall = R["figure_data"].get("fig5_length_accuracy_multi", {})
    if not per_level or not overall:
        return
    models = [m for m in per_level.keys() if per_level[m]]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    axes = axes[0]
    level_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for ax, model in zip(axes, models):
        # Per-level curves
        plot_data = per_level[model]
        for lvl in sorted(plot_data.keys(), key=lambda x: int(x)):
            d = plot_data[lvl]
            ax.plot(d["mean_length"], [a * 100 for a in d["accuracy"]],
                    marker="o", linestyle="-", lw=1.3, ms=5,
                    color=level_colors[int(lvl) - 1],
                    label="Level " + str(lvl) + " ($n{=}" + str(d['n']) + "$)",
                    alpha=0.8)
        # Overall curve (thick, black)
        ov = overall.get(model, {})
        if ov:
            ax.plot(ov["mean_length"], [a * 100 for a in ov["accuracy"]],
                    marker="s", linestyle="-", lw=2.8, ms=8,
                    color="black",
                    label="Overall ($r{=}" + f"{ov.get('pearson_r', 0):+.2f}" + "$)")
        ax.set_xlabel("Mean length in decile (words)")
        ax.set_ylabel("Accuracy (\\%)" if model == models[0] else "")
        ax.set_title(model.replace("_", "-"))
        ax.set_xscale("log")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    fig.suptitle("Length-Accuracy Relationship Stratified by Difficulty", y=1.02)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_length_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig5] saved")


def fig_practical_strategies(R):
    """Pareto scatter: length vs accuracy, 8 strategies x 2 models = 16 points.
    Connect same-model points with a dashed line; mark Pareto-optimal points."""
    data = R["figure_data"].get("fig6_practical_strategies", {})
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    strategies = ["single", "shortest_any", "median_any", "longest_any",
                  "majority_vote", "majority_first", "majority_median",
                  "majority_shortest"]
    strat_short = {"single": "single", "shortest_any": "short",
                   "median_any": "med", "longest_any": "long",
                   "majority_vote": "maj-vote", "majority_first": "maj-first",
                   "majority_median": "maj-med", "majority_shortest": "maj-short"}
    for ax, (model, d) in zip(axes, data.items()):
        xs = [d.get(s, {}).get("avg_length", 0) for s in strategies]
        ys = [d.get(s, {}).get("accuracy", 0) * 100 for s in strategies]
        # Compute Pareto frontier (lower x, higher y is better)
        pts = sorted(zip(xs, ys, strategies), key=lambda t: t[0])
        frontier = []
        best_y = -1
        for x, y, s in pts:
            if y > best_y:
                frontier.append((x, y, s))
                best_y = y
        # Plot all points
        for x, y, s in zip(xs, ys, strategies):
            is_ours = "shortest" in s
            color = "#d62728" if is_ours else "#1f77b4"
            size = 140 if is_ours else 90
            marker = "*" if is_ours else "o"
            ax.scatter(x, y, s=size, color=color, marker=marker, edgecolor="black",
                       lw=0.8, zorder=3)
            # label each point
            dy = 1.2 if not is_ours else -2.5
            ax.annotate(strat_short.get(s, s), (x, y),
                        textcoords="offset points", xytext=(4, dy),
                        fontsize=8, ha="left")
        # Draw frontier
        if len(frontier) >= 2:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, "--", color="gray", lw=1.5, alpha=0.6, zorder=2,
                    label="Pareto frontier")
        ax.set_xlabel("Avg.\\ length (words, lower is better)")
        ax.set_ylabel("Accuracy (\\%, higher is better)")
        ax.set_title(model.replace("_", "-"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Accuracy--Length Pareto Frontier: Our Methods (red $\\star$) Dominate", y=1.02)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_practical_strategies.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig6] saved")


def fig_prefix_position(R):
    """4-strategy x 2-model accuracy vs relative prefix length."""
    ext = R.get("extended", {})
    data = ext.get("prefix_position_aggregate", {})
    if not data:
        return
    models = list(data.keys())
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    axes = axes[0]
    strat_colors = {"first": "#1f77b4", "last": "#d62728",
                    "middle": "#2ca02c", "random": "#ff7f0e"}
    strat_markers = {"first": "o", "last": "s", "middle": "^", "random": "D"}
    for ax, model in zip(axes, models):
        strat_data = data[model]
        for strat, d in strat_data.items():
            ax.plot(d["rel_positions"], [a * 100 for a in d["accuracy"]],
                    marker=strat_markers.get(strat, "o"),
                    color=strat_colors.get(strat, "black"),
                    linestyle="-", lw=2, ms=7,
                    label=f"{strat}-$k$")
        ax.set_xlabel("Relative prefix length $k / N$")
        ax.set_ylabel("Judge accuracy (\\%)")
        ax.set_title(model.replace("_", "-"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Prefix-Position Ablation")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_prefix_position.pdf")
    plt.close(fig)
    print("[fig_prefix_position] saved")


def fig_per_subject(R):
    """Two panels: (left) per-subject rho with CI error bars for both models;
    (right) scatter plot of (avg length, rho) with subject labels and both
    models connected by lines."""
    ext = R.get("extended", {})
    data = ext.get("per_subject_redundancy", {})
    scatter = ext.get("subject_length_rho_scatter", {})
    if not data:
        return
    subjects = set()
    for m, d in data.items():
        if d:
            subjects.update(d.keys())
    # Sort subjects by R1 rho
    r1_data = data.get("deepseek_r1", {}) or {}
    subjects = sorted(subjects, key=lambda s: (r1_data.get(s, {}) or {}).get("rho", 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: grouped bars with CI error bars
    xs = np.arange(len(subjects))
    width = 0.38
    for i, (model, d) in enumerate(data.items()):
        if not d:
            continue
        rhos, lo_err, hi_err = [], [], []
        for s in subjects:
            v = d.get(s, {}) or {}
            rho = v.get("rho", 0) * 100
            lo = v.get("ci_lo", rho / 100) * 100
            hi = v.get("ci_hi", rho / 100) * 100
            rhos.append(rho)
            lo_err.append(rho - lo)
            hi_err.append(hi - rho)
        offset = (i - 0.5) * width
        ax1.bar(xs + offset, rhos, width, label=model.replace("_", "-"),
                color=COLORS[i], edgecolor="black", lw=0.5,
                yerr=[lo_err, hi_err], capsize=3, error_kw={"lw": 1, "alpha": 0.9})
    ax1.set_xticks(xs)
    ax1.set_xticklabels([s.replace(" & ", "\\&").replace(" ", "\n") for s in subjects],
                        fontsize=8)
    ax1.set_ylabel("Redundancy $\\rho$ (\\%)")
    ax1.set_ylim(50, 100)
    ax1.set_title("Per-Subject Redundancy with 95\\% CI")
    ax1.axhline(76.3, color=COLORS[0], linestyle=":", lw=1, alpha=0.5, label="R1 mean")
    ax1.axhline(86.0, color=COLORS[1], linestyle=":", lw=1, alpha=0.5, label="QwQ mean")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: length vs rho scatter with subject labels
    all_lengths = []
    for i, model in enumerate(scatter.keys()):
        rows = scatter[model]
        lens = [r["avg_length"] for r in rows]
        rhos = [r["rho_pct"] for r in rows]
        all_lengths.extend(lens)
        lo_err = [r["rho_pct"] - r["ci_lo_pct"] for r in rows]
        hi_err = [r["ci_hi_pct"] - r["rho_pct"] for r in rows]
        ax2.errorbar(lens, rhos, yerr=[lo_err, hi_err], fmt="none",
                     ecolor=COLORS[i], alpha=0.6, capsize=3)
        ax2.scatter(lens, rhos, s=130, color=COLORS[i],
                    edgecolor="black", lw=0.8, zorder=3,
                    label=model.replace("_", "-"), marker=MARKERS[i])
        # Label subjects
        for r in rows:
            ax2.annotate(r["subject"].replace(" & ", "\\&")[:8],
                         (r["avg_length"], r["rho_pct"]),
                         textcoords="offset points", xytext=(5, 2),
                         fontsize=7, alpha=0.7)
    # Connect same-subject points with line
    if len(scatter) == 2:
        m0, m1 = list(scatter.keys())
        rows0 = {r["subject"]: r for r in scatter[m0]}
        rows1 = {r["subject"]: r for r in scatter[m1]}
        for s in rows0:
            if s in rows1:
                ax2.plot([rows0[s]["avg_length"], rows1[s]["avg_length"]],
                         [rows0[s]["rho_pct"], rows1[s]["rho_pct"]],
                         color="gray", lw=0.8, alpha=0.3, zorder=1)
    ax2.set_xlabel("Avg.\\ trace length (words, log scale)")
    ax2.set_ylabel("Redundancy $\\rho$ (\\%)")
    ax2.set_xscale("log")
    ax2.set_ylim(50, 100)
    ax2.set_title("Length vs.\\ Redundancy per Subject")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_per_subject.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig_per_subject] saved")


def fig_k_star_distribution(R):
    """ECDF of k*/N for 4 (model, dataset) conditions with P50, P90 marker lines."""
    ext = R.get("extended", {})
    data = ext.get("k_star_ecdf", {})
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    i = 0
    styles = {"deepseek_r1": {"math500": ("-", "o"), "gsm8k": ("--", "s")},
              "qwq_32b":     {"math500": ("-", "^"), "gsm8k": ("--", "D")}}
    for model, ds_data in data.items():
        for ds, d in ds_data.items():
            xs, ys = d["x"], d["y"]
            ls, mk = styles.get(model, {}).get(ds, ("-", "o"))
            color = COLORS[i % len(COLORS)]
            label = model.replace("_","-") + "/" + ds + f" (P50={d['p50']:.2f}, P90={d['p90']:.2f}, n={d['n']})"
            ax.plot(xs, ys, linestyle=ls, color=color, lw=2.0,
                    label=label)
            # P50 marker
            ax.plot([d["p50"]], [0.5], marker=mk, color=color, ms=9, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.6)
            # P90 marker
            ax.plot([d["p90"]], [0.9], marker=mk, color=color, ms=9, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.6, alpha=0.7)
            i += 1
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, alpha=0.6)
    ax.axhline(0.9, color="gray", linestyle=":", lw=0.8, alpha=0.6)
    ax.annotate("P50", (0.98, 0.51), fontsize=8, color="gray", ha="right")
    ax.annotate("P90", (0.98, 0.91), fontsize=8, color="gray", ha="right")
    ax.set_xlabel("Relative critical-point position $k^\\star / N$")
    ax.set_ylabel("Cumulative fraction of traces")
    ax.set_title("ECDF of Critical-Point Position: Most Traces Have $k^\\star < 0.1 N$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig_k_star_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[fig_k_star_distribution] saved")


def main():
    import sys, traceback
    R = load_results()
    print("loaded results", flush=True)
    for name, fn in [
        ("fig2", fig_difficulty_redundancy),
        ("fig3", fig_step_taxonomy),
        ("fig4", fig_positional_redundancy),
        ("fig5", fig_length_accuracy),
        ("fig6", fig_practical_strategies),
        ("fig_prefix_position", fig_prefix_position),
        ("fig_per_subject", fig_per_subject),
        ("fig_k_star_distribution", fig_k_star_distribution),
    ]:
        print(f"[{name}] starting...", flush=True)
        try:
            fn(R)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    main()
