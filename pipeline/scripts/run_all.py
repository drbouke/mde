"""
Top-level orchestrator — runs all pipeline stages in order.

Usage:
  python pipeline/scripts/run_all.py

Each stage is run as a subprocess so stdout is streamed live.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PYTHON = sys.executable

STAGES = [
    ("Ablation (5-fold CV, 4 datasets)",                ROOT / "pipeline/experiments/run_ablation.py"),
    ("Temporal generalization (CICIDS-2017)",            ROOT / "pipeline/experiments/run_timesplit.py"),
    ("Pseudo-live temporal replay (CICIDS-2017)",        ROOT / "pipeline/experiments/run_temporal_replay.py"),
    ("Classifier comparison (LGB/RF/XGB/CatBoost/MLP/TabNet/FTT)", ROOT / "pipeline/experiments/run_baselines.py"),
    ("Unseen attack families",                           ROOT / "pipeline/experiments/run_unseen.py"),
    ("Per-class detection rates",                        ROOT / "pipeline/experiments/run_perclass.py"),
    ("ROC / CM / JSD figures",                           ROOT / "pipeline/evaluation/run_figures.py"),
    ("Class distribution / heatmap / model comparison",  ROOT / "pipeline/evaluation/run_viz.py"),
    ("SHAP figures + fold stability",                    ROOT / "pipeline/shap/run_shap.py"),
]


def run_stage(label, script_path):
    print(f"\n{'='*70}", flush=True)
    print(f"  STAGE: {label}", flush=True)
    print(f"  Script: {script_path.relative_to(ROOT)}", flush=True)
    print(f"{'='*70}", flush=True)
    result = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n  ERROR: stage '{label}' exited with code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    print(f"\n  DONE: {label}", flush=True)


if __name__ == "__main__":
    print("MDE Pipeline — running all stages", flush=True)
    for label, script in STAGES:
        run_stage(label, script)
    print("\n" + "="*70, flush=True)
    print("  ALL STAGES COMPLETE", flush=True)
    print("="*70, flush=True)
