"""Phase 10: Orchestrate the final deliverable generation.

Run after Phase 3b primary + cross judges complete:
  1. Run 05 analysis (reads v2 data automatically via config.truncation_path)
  2. Run 08 extended analysis (same)
  3. Run 07 compile results (merges extended into results.json)
  4. Run 06 generate figures (reads merged results.json)
  5. Compile rho_results_report.tex into PDF
  6. Copy everything into outputs/deliverable/ and zip
"""
from __future__ import annotations
import subprocess
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable  # current Python interpreter

def run(cmd, cwd=ROOT, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd, **kw)
    if r.returncode != 0:
        print(f"  returned {r.returncode}", flush=True)

def main():
    # Make sure SCALE=A is in env (so REASONING_MODELS has all 4)
    import os
    env = os.environ.copy()
    env["SCALE"] = "A"

    # 1. 05 analysis
    run([PY, "scripts/05_analysis.py"], env=env)
    # 2. 08 extended
    run([PY, "scripts/08_extended_analysis.py"], env=env)
    # 3. 07 compile
    run([PY, "scripts/07_compile_results.py"], env=env)
    # 4. 06 figures (unbuffered for visibility)
    run([PY, "-u", "scripts/06_generate_figures.py"], env=env)

    # 5. Compile rho_results_report.pdf (two passes for cross-refs)
    report_dir = ROOT / "outputs"
    run(["pdflatex", "-interaction=nonstopmode", "rho_results_report.tex"], cwd=report_dir)
    run(["pdflatex", "-interaction=nonstopmode", "rho_results_report.tex"], cwd=report_dir)

    # 6. Assemble deliverable
    deliv = ROOT / "outputs" / "deliverable"
    if deliv.exists():
        shutil.rmtree(deliv)
    deliv.mkdir(parents=True)

    # Files to include (same list as old deliverable for schema compatibility)
    outputs = ROOT / "outputs"
    figs = outputs / "figures"
    to_copy = [
        outputs / "rho_results_report.pdf",
        outputs / "rho_results_report.tex",
        outputs / "results.json",
        outputs / "extended_results.json",
        outputs / "case_studies.txt",
        figs / "fig_difficulty_redundancy.pdf",
        figs / "fig_k_star_distribution.pdf",
        figs / "fig_length_accuracy.pdf",
        figs / "fig_per_subject.pdf",
        figs / "fig_positional_redundancy.pdf",
        figs / "fig_practical_strategies.pdf",
        figs / "fig_prefix_position.pdf",
        figs / "fig_step_taxonomy.pdf",
    ]
    for src in to_copy:
        if src.exists():
            shutil.copy2(src, deliv / src.name)
            print(f"  + {src.name}")
        else:
            print(f"  [missing] {src.name}")

    # 7. Zip it
    zip_path = outputs / "deliverable.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in deliv.iterdir():
            zf.write(f, arcname=f"deliverable/{f.name}")
    print(f"\nSaved: {zip_path}")

if __name__ == "__main__":
    main()
