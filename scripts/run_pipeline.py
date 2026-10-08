"""Ordered core pipeline from a prepared count matrix. Dry-run by default.

No raw-data preparation is implied. Run prepare_data.py or the original COMBAT
preparer first. --execute explicitly enables potentially expensive training.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def plan(cohort, seeds):
    steps = []
    if cohort == "stephenson":
        for seed in seeds:
            steps.extend([("stephenson/stephenson_donor_calibration.py", seed),
                          ("stephenson/ridge_pred.py", seed), ("stephenson/lowrank.py", seed)])
    else:
        steps.extend([(f"combat/combat_{s}.py", None) for s in ["totalvi", "ridge", "lowrank"]])
    steps.extend([(s, None) for s in [
        "analysis/baselines_calib.py",  # creates sentinels_fit*.json before it is consumed
        "analysis/ridge_sentinels.py",
        "analysis/baselines_calib.py",  # now includes ridge + sentinels
        "analysis/proteoguard_metrics.py", "analysis/anchor_budget.py",
        "analysis/marker_calls_all.py", "analysis/risk_coverage_b.py"]])
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["stephenson", "combat"], required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds")
    work = args.work.resolve()
    steps = plan(args.cohort, args.seeds)
    print(json.dumps({"cohort": args.cohort, "work": str(work), "seeds": args.seeds,
                      "execute": args.execute, "steps": steps}, indent=2))
    if not args.execute:
        return
    if not (work / "stephenson_sub1000.h5ad").is_file():
        parser.error("Missing stephenson_sub1000.h5ad; see docs/DATA_PREPARATION.md")
    if args.cohort == "stephenson" and not (work / "haniffa21.processed.h5ad").is_file():
        parser.error("Stephenson lowrank.py also needs haniffa21.processed.h5ad for metadata")
    subprocess.run([sys.executable, str(ROOT / "prepare_data.py"), "validate",
                    str(work / "stephenson_sub1000.h5ad")], check=True)
    env = os.environ.copy()
    env.update(STEPH_DIR=str(work), STEPH_FITS=",".join(map(str, args.seeds)), MPLBACKEND="Agg")
    env.setdefault("STEPH_CTS", "B_cell,CD14,CD4,CD8,NK_16hi" if args.cohort == "stephenson" else "cMono,CD4,CD8,NK,B")
    manifest = work / "pipeline_run.json"
    if manifest.exists():
        parser.error("pipeline_run.json already exists; archive it deliberately before a new run")
    record = dict(cohort=args.cohort, seeds=args.seeds, completed=[], status="running")
    manifest.write_text(json.dumps(record, indent=2))
    try:
        for script, seed in steps:
            step_env = env.copy()
            if seed is not None:
                step_env["STEPH_FIT"] = str(seed)
            print(f"Running {script} (seed={seed})", flush=True)
            subprocess.run([sys.executable, str(ROOT / script)], env=step_env, check=True)
            record["completed"].append([script, seed]); manifest.write_text(json.dumps(record, indent=2))
        subprocess.run([sys.executable, str(ROOT / "figures/fig_budget.py"),
                        "--input", str(work / "anchor_budget.csv"),
                        "--output", str(work / "figures/f7.png")], env=env, check=True)
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"; record["error"] = str(error)
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
