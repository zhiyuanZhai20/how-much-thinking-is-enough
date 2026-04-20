#!/usr/bin/env bash
# End-to-end pilot runner. Each phase resumes from its checkpoint.
set -e
cd "$(dirname "$0")"
export SCALE=pilot

PYTHON="${PYTHON:-python}"

$PYTHON scripts/00_download_data.py
$PYTHON scripts/01_collect_traces.py
$PYTHON scripts/02_segment_steps.py
$PYTHON scripts/03_truncation_exp.py
$PYTHON scripts/04_ablation_exp.py
$PYTHON scripts/05_analysis.py
$PYTHON scripts/06_generate_figures.py
$PYTHON scripts/07_compile_results.py
