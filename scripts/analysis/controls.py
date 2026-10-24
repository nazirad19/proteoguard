# Controls: (a) test-only rescoring — headline RMSE recomputed on test participants only (design excluded),
# identical protein set across methods; (b) modality decomposition — anchors-alone ridge (5 measured sentinels
# as the ONLY inputs, fit on design participants) vs RNA alone vs RNA + anchors.
# Pre-registered: evaluation proteins = all minus both sentinel sets minus isotypes; test participants =
# held-out, non-design, non-Sanger; anchors-alone is fit on design cells only (same role as map fitting).
# Run per cohort: STEPH_DIR=~/projects/stephenson python controls.py  (then ~/projects/combat)
import os, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge (RNA alone)": "pred_ridge_fit{seed}.npz",
         "ridge + sentinels (RNA + anchors)": "pred_ridgeanchor_fit{seed}.npz",
         "scVAEIT": "pred_scvaeit_fit{seed}.npz"}
ISO = re.compile(r"(?i)(isotyp|igg1|igg2|igg_|iggfc|mouse_?ig|rat_?ig|hamster)")

rows = []
for seed in FITS:
    design = set(pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str))
    sj = json.load(open(WORK / f"sentinels_fit{seed}.json"))
    anchors_all = set(sj.get("ridge", [])) | set(sj.get("totalVI", []))
    ref = None
    data = {}
    for nm, f in PREDS.items():
        fp = WORK / f.format(seed=seed)
        if fp.exists(): data[nm] = np.load(fp, allow_pickle=True)
    if not data: continue
    d0 = next(iter(data.values()))
    prot = d0["protein"].astype(str)
    E = np.array([j for j, p in enumerate(prot) if p not in anchors_all and not ISO.search(p)])
    A = [list(prot).index(a) for a in sj["ridge"] if a in list(prot)]

    for nm, d in data.items():
        y, mid = d["y"], d["mean"]; pats = d["patient"].astype(str); site = d["site"].astype(str)
        dz_c = np.isin(pats, list(design))
        test_c = ~dz_c & (site != "Sanger")
        rmse_all = float(np.sqrt(((mid[:, E] - y[:, E]) ** 2).mean(0)).mean())
        rmse_test = float(np.sqrt(((mid[test_c][:, E] - y[test_c][:, E]) ** 2).mean(0)).mean())
        rows.append(dict(fit=seed, predictor=nm, rmse_all_held=round(rmse_all, 4), rmse_test_only=round(rmse_test, 4)))

    # anchors-alone ridge: X = [cell sentinel values, participant-median sentinels], fit on design cells
    d = data.get("ridge (RNA alone)") or d0
    y = d["y"]; pats = d["patient"].astype(str); site = d["site"].astype(str)
    med = pd.DataFrame(y[:, A], index=pats).groupby(level=0).median()
    X = np.c_[y[:, A], med.loc[pats].to_numpy()]
    dz_c = np.isin(pats, list(design)); test_c = ~dz_c & (site != "Sanger")
    m = Ridge(alpha=1.0).fit(X[dz_c], y[dz_c][:, E])
    pr = m.predict(X[test_c])
    rmse_test = float(np.sqrt(((pr - y[test_c][:, E]) ** 2).mean(0)).mean())
    rows.append(dict(fit=seed, predictor="anchors alone (5 sentinels, no RNA)", rmse_all_held=np.nan,
                     rmse_test_only=round(rmse_test, 4)))
    # design-mean baseline: predict each protein by its design-cell mean (no inputs at all)
    base = y[dz_c][:, E].mean(0)
    rmse_test = float(np.sqrt(((base - y[test_c][:, E]) ** 2).mean(0)).mean())
    rows.append(dict(fit=seed, predictor="no inputs (design mean per protein)", rmse_all_held=np.nan,
                     rmse_test_only=round(rmse_test, 4)))
    print(f"fit {seed} done ({len(E)} evaluation proteins)", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "controls_rmse.csv", index=False)
pd.set_option("display.width", 200)
print("\nTest-only rescoring and modality decomposition (common evaluation proteins; per fit):")
print(R.to_string(index=False))
print("\nmean over fits:")
print(R.groupby("predictor")[["rmse_all_held", "rmse_test_only"]].mean().round(4).to_string())
print("\nsaved controls_rmse.csv")
