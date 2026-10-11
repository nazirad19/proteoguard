"""CPU end-to-end demonstration using bundled synthetic RNA/ADT counts.

Fits an RNA ridge predictor on training participants, estimates its interval
width on separate validation participants, and runs the actual research
baselines_calib.py on design and evaluation participants. Then fits a second
ridge predictor using the selected five measured proteins as extra inputs.

No totalVI/scVAEIT training, downloads, GPU or patient data are needed.
Synthetic results are not the cohort results in the report.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ridge_fit(x, y, penalty=10.0):
    """Train-only centering/scaling; fixed penalty, unpenalized intercept."""
    mx, sx, my = x.mean(0), x.std(0), y.mean(0)
    sx = np.where(sx > 1e-8, sx, 1.0)
    z = (x - mx) / sx
    weights = np.linalg.solve(z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (y - my))
    return mx, sx, my, weights


def ridge_predict(model, x):
    mx, sx, my, weights = model
    return np.maximum(my + ((x - mx) / sx) @ weights, 0.0)


def validate(frame):
    """Reject malformed data and participant leakage across fixed roles."""
    required = {"cell_id", "patient", "role", "site", "celltype"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError("Missing metadata")
    if frame.cell_id.duplicated().any() or (frame.groupby("patient").role.nunique() != 1).any():
        raise ValueError("Duplicate cells or participant leakage across roles")
    counts = frame.filter(regex=r"^(RNA_|AB_)").to_numpy()
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.rint(counts)).all():
        raise ValueError("Expected finite nonnegative integer counts")
    if set(frame.role) != {"train", "width_validation", "design", "evaluation_pool"}:
        raise ValueError("Expected four predefined participant roles")
    if frame.loc[frame.role == "evaluation_pool", "patient"].nunique() <= 15:
        raise ValueError("Need 15 calibration participants and at least one test participant")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "examples/toy_data/synthetic_counts.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/toy_demo")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error("Output exists; use a new --output directory to preserve earlier runs")
    started = time.time()
    frame = pd.read_csv(args.data).sort_values(["patient", "cell_id"]).reset_index(drop=True)
    validate(frame)
    out.mkdir(parents=True)
    proteins = frame.filter(regex=r"^AB_").columns.to_numpy(dtype=str)
    counts = frame.filter(regex=r"^RNA_").to_numpy(float)
    x = np.log1p(1e4 * counts / np.maximum(counts.sum(1, keepdims=True), 1))
    y = np.log1p(frame[proteins].to_numpy(float))
    train = (frame.role == "train").to_numpy()
    width = (frame.role == "width_validation").to_numpy()
    held = frame.role.isin(["design", "evaluation_pool"]).to_numpy()
    sub = frame.loc[held].reset_index(drop=True)
    pd.DataFrame({"patient": sub.loc[sub.role == "design", "patient"].unique()}).to_csv(
        out / "design_split_fit2026.csv", index=False)

    def fit_export(features, filename):
        model = ridge_fit(features[train], y[train])
        mean = ridge_predict(model, features)
        # Width from participants excluded from ridge fitting, not test labels.
        half = np.quantile(np.abs(y[width] - mean[width]), .90, axis=0, method="higher")
        np.savez_compressed(out / filename, y=y[held], mean=mean[held],
            lo=mean[held] - half, hi=mean[held] + half,
            patient=sub.patient.to_numpy(dtype=str), site=sub.site.to_numpy(dtype=str),
            celltype=sub.celltype.to_numpy(dtype=str), protein=proteins)
        return mean

    base = fit_export(x, "pred_ridge_fit2026.npz")
    env = os.environ.copy()
    # Remove inherited research settings so this command cannot read another run.
    for key in list(env):
        if key.startswith("STEPH_"):
            del env[key]
    env.update(STEPH_DIR=str(out), STEPH_FITS="2026", STEPH_NCAL="15", STEPH_REPS="3",
               PYTHONIOENCODING="utf-8")

    def calibrate(predictors, label):
        env["STEPH_PREDICTORS"] = json.dumps(predictors)
        run = subprocess.run([sys.executable, str(ROOT / "scripts/analysis/baselines_calib.py")],
            env=env, capture_output=True, text=True, encoding="utf-8")
        (out / f"{label}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
        if run.returncode:
            raise RuntimeError(f"Calibration failed: see {out / (label + '.log')}")

    calibrate({"ridge": "pred_ridge_fit{seed}.npz"}, "anchor_selection")
    anchors = json.loads((out / "sentinels_fit2026.json").read_text())["ridge"]
    anchor_indices = [int(np.where(proteins == a)[0][0]) for a in anchors]
    augmented = fit_export(np.c_[x, y[:, anchor_indices]], "pred_ridgeanchor_fit2026.npz")
    calibrate({"ridge": "pred_ridge_fit{seed}.npz",
               "ridge + sentinels": "pred_ridgeanchor_fit{seed}.npz"}, "calibration")
    results = pd.read_csv(out / "baselines_calibration.csv")
    metrics = ["coverage", "frac_participants_ge90", "interval_score", "width"]
    summary = results.groupby(["predictor", "correction", "calibration"])[metrics].mean().reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    # Recover the evaluator's three allocations, solely for reproducibility.
    pool = sub.loc[sub.role == "evaluation_pool", "patient"].unique()
    rng = np.random.default_rng(7)
    split_rows = []
    test_masks = []
    for rep in range(3):
        order = rng.permutation(pool)
        split_rows.extend(dict(rep=rep, patient=p, role="calibration" if i < 15 else "test")
                          for i, p in enumerate(order))
        test_masks.append(frame.patient.isin(order[15:]).to_numpy())
    pd.DataFrame(split_rows).to_csv(out / "evaluation_splits.csv", index=False)
    targets = [j for j in range(len(proteins)) if j not in anchor_indices and "Isotype" not in proteins[j]]
    point = []
    for rep, mask in enumerate(test_masks):
        for name, pred in [("ridge", base), ("ridge + sentinels", augmented)]:
            point.append(dict(rep=rep, predictor=name,
                test_rmsle=float(np.sqrt(np.mean((y[mask][:, targets] - pred[mask][:, targets]) ** 2)))))
    pd.DataFrame(point).to_csv(out / "point_metrics.csv", index=False)
    manifest = dict(synthetic=True, seed=20261011, fit_label=2026,
        base_predictor="numpy ridge; no totalVI or scVAEIT was trained",
        input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        selected_anchors=anchors, evaluation_targets=list(proteins[targets]),
        role_participants=frame.groupby("role").patient.nunique().to_dict(),
        calibration_participants_per_repeat=15, test_participants_per_repeat=len(pool)-15,
        repeats=3, repeats_are_independent=False, seconds=round(time.time()-started, 2))
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert len(anchors) == 5 and len(results) == 63
    assert np.isfinite(results[metrics].to_numpy()).all()
    assert results.coverage.between(0, 1).all()
    print("PASS: synthetic CPU demo completed; no cohort results were reproduced.")
    print(f"Output: {out}\nCalibration rows: {len(results)}; five anchors: {anchors}")
    print(summary.to_string(index=False, float_format=lambda n: f"{n:.3f}"))


if __name__ == "__main__":
    main()
