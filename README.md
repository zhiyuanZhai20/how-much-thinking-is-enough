# How Much Thinking is Enough?
### Quantifying and Understanding Redundancy in LLM Reasoning

Code, data, and aggregated results for the paper:

> **How Much Thinking is Enough? Quantifying and Understanding Redundancy in LLM Reasoning.**
> Zhiyuan Zhai, Xinkai You, Wenjing Yan, Xin Wang. 2026.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)  <!-- fill after arXiv assigns an ID -->

We formalise *reasoning redundancy* directly in terms of the reasoning model itself: for a correct trace, redundancy is the largest fraction of trailing segmented steps that can be truncated while the model, forced to terminate thinking and emit a final answer, still produces the correct answer. Across 4 frontier reasoning models × 2 math benchmarks × 2 judges, step-level redundancy ρ is consistently **61 %–93 %**; the median critical prefix is a *single* segmented step in 6 of 8 conditions. We then prove that this is a structural consequence of length-agnostic outcome rewards — not a model-specific artefact.

## Highlights

- **C1. Definition.** ρ(r) = 1 − k\*(r)/N, where k\*(r) is the smallest prefix at which the model — forced to emit an answer — is still correct.
- **C2. Quantification at scale.** 4 models (DeepSeek-R1, QwQ-32B, R1-Distill-Qwen-7B, Qwen3-30B-A3B-Thinking) × 2 benchmarks (GSM8K, MATH-500) × 2 judges (π-as-own-decoder + `gpt-4o-mini`). 1,880 correct traces.
- **C3. Theory.** Under any outcome-only (length-agnostic) reward, no finite expected stopping time is optimal — so over-thinking is structural, independent of RL algorithm, base model, or training recipe.

## Repository layout

```
config.py                       # models, endpoints, SCALE knob, paths
.env.example                    # template — copy to `.env` and fill keys
requirements.txt
run_pilot.sh                    # convenience end-to-end runner

utils/
  api_clients.py                # async OpenAI-compatible client (DeepSeek / DashScope / OpenAI)
  answer_extraction.py          # \boxed{} parsing and answer comparison
  segmentation.py               # paragraph-level step segmentation
  io_utils.py                   # JSONL append/read with checkpointing

scripts/
  00_download_data.py           # MATH-500 + 500-problem GSM8K subset
  01_collect_traces.py          # Phase 1: collect M reasoning traces per problem
  02_segment_steps.py           # Phase 2: split traces into discrete steps
  03_truncation_exp.py          # Phase 3: external-judge truncation
  03b_truncation_self_judge.py  # Phase 3b: π-as-own-decoder truncation (paper's primary)
  04_ablation_exp.py            # Phase 4: leave-one-out step ablation
  05_analysis.py                # Phase 5-6: variance + practical strategies
  06_generate_figures.py        # generate paper figures
  07_compile_results.py         # aggregate everything into results.json
  08_extended_analysis.py       # appendix analyses
  09_prefix_position_ablation.py# prefix-position ablation
  10_finalize.py                # assemble final deliverable
  99_watchdog.py                # self-healing driver

data/
  math500.jsonl                 # MATH-500
  gsm8k_500.jsonl               # 500-problem GSM8K test subset
  aqua_100.jsonl                # AQuA-RAT (extras for robustness)
  svamp_100.jsonl               # SVAMP (extras for robustness)

outputs/
  figures/                      # paper figures (8 PDFs)
  results.json                  # aggregated tables 1–8 + figure data
  extended_results.json         # appendix-level aggregates
  # traces/, truncation/, ablation/ are gitignored — regenerate via scripts above
```

## Quick start

### 1. Install

```bash
git clone https://github.com/zhiyuanZhai20/how-much-thinking-is-enough.git
cd how-much-thinking-is-enough
python -m venv .venv && source .venv/bin/activate   # or your env manager
pip install -r requirements.txt
```

### 2. Configure keys

Copy `.env.example` to `.env` and fill in the providers you want to query:

- `DEEPSEEK_API_KEY` — for `deepseek-reasoner` and `deepseek-chat`
- `DASHSCOPE_API_KEY` — for `qwq-32b`, `deepseek-r1-distill-qwen-7b`, `qwen3-30b-a3b-thinking-2507`
- `OPENAI_API_KEY` — optional, only needed for the `gpt-4o-mini` external-judge robustness check

All three providers speak the OpenAI Chat Completions schema, so a single client wrapper handles them (`utils/api_clients.py`).

### 3. Pick a scale

`config.py` reads `SCALE` from the environment:

| `SCALE` | # MATH | # GSM8K | M samples | Models | Use for |
|--------|-------:|--------:|----------:|:-------|:--------|
| `pilot` | 20 | 20 | 2 | DeepSeek-R1 only | smoke test (≈ minutes) |
| `A` (default, paper's config) | 150 | 60 | 3 | all 4 | reproduction |
| `full` | 500 | 500 | 10 | all 4 | ablation budget-permitting |

### 4. Run

```bash
# Full reproduction pipeline (each phase is checkpoint-resumable):
export SCALE=A
python scripts/00_download_data.py
python scripts/01_collect_traces.py            # Phase 1: reasoning traces
python scripts/02_segment_steps.py             # Phase 2: segmentation
python scripts/03b_truncation_self_judge.py    # Phase 3b: π-as-own-decoder (PRIMARY)
python scripts/03_truncation_exp.py            # Phase 3: gpt-4o-mini external judge (robustness)
python scripts/04_ablation_exp.py              # Phase 4: leave-one-out
python scripts/05_analysis.py                  # Phase 5-6: variance + strategies
python scripts/06_generate_figures.py          # regenerate figures
python scripts/07_compile_results.py           # results.json
python scripts/08_extended_analysis.py         # appendix extras
python scripts/09_prefix_position_ablation.py  # prefix-position ablation
```

All scripts write JSONL line-by-line and resume on restart, so you can Ctrl-C safely.

## Reproducing the paper's tables and figures from aggregates

All headline numbers are stored in `outputs/results.json`:

| Key | Contents |
|-----|----------|
| `table1_overall_redundancy` | ρ_π per (model, benchmark), median k\*, mean N |
| `table2_crossjudge` | ρ_ext via `gpt-4o-mini` (robustness check) |
| `table3_length_by_difficulty` | ρ(d) stratified by MATH-500 difficulty |
| `table4_variance` | within-problem σ of ρ |
| `table5_shortest_correct` | shortest-correct-prefix baseline |
| `table6_budget` | budget-filter monotonicity |
| `table7_early_stop` | early-stopping performance |
| `table8_theory_fit` | empirical fit of the inverted-U length–accuracy prediction |
| `figure_data` | raw data backing the 8 figures in `outputs/figures/` |

The 8 figures in `outputs/figures/` are the exact PDFs embedded in the paper. To rebuild them from `figure_data`, run `scripts/06_generate_figures.py`.

## What's *not* in this repo (by design)

- **Raw reasoning traces** (~64 MB) and **truncation-experiment outputs** (~13 MB) are omitted for size. They are fully regenerable from `scripts/01_…` onwards; for a drop-in copy please contact the authors.
- **Logs** (`phase*.log`, `watchdog*.log`) are local debugging artefacts.

## Models queried

All models are accessed via their public APIs; no local inference.

| Display name | Provider | Model ID |
|--------------|----------|----------|
| `deepseek_r1` | DeepSeek | `deepseek-reasoner` |
| `qwq_32b` | DashScope | `qwq-32b` |
| `r1_distill_7b` | DashScope | `deepseek-r1-distill-qwen-7b` |
| `qwen3_30b_thinking` | DashScope | `qwen3-30b-a3b-thinking-2507` |
| external judge | OpenAI | `gpt-4o-mini-2024-07-18` |

All reasoning generations at T = 0.7; forced-termination decoding at T = 0 with `max_tokens=64`; concurrency 8; all seeds fixed at 42.

## Citation

```bibtex
@article{zhai2026howmuch,
  title        = {How Much Thinking is Enough? Quantifying and Understanding Redundancy in {LLM} Reasoning},
  author       = {Zhai, Zhiyuan and You, Xinkai and Yan, Wenjing and Wang, Xin},
  year         = {2026},
  eprint       = {XXXX.XXXXX},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG}
}
```

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

- Zhiyuan Zhai (Fudan University) — `22110720067@m.fudan.edu.cn`
- Issues and PRs on GitHub are welcome.
