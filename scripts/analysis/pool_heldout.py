# PH1: processing-pool-held-out COMBAT experiment — the boundary of the guarantee.
# For each held-out pool P: design/map and calibration participants come from OTHER pools only; test = P.
# Reports per-pool frac>=90 / below-85 / width for participant-block calibration (totalVI and ridge+sentinels),
# next to the random-split baseline. Failures are the point: do NOT expect the guarantee to extend here.
# Run: STEPH_DIR=~/projects/combat python pool_heldout.py
import os, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "combat"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge + sentinels": "pred_ridgeanchor_fit{seed}.npz"}
GAMMA, N_CAL, N_REPS, MIN_POOL = 0.90, 15, 20, 3
RNG = np.random.default_rng(13)

obs = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad", backed="r").obs
pool_col = next((c for c in obs.columns if "pool" in c.lower()), None)
assert pool_col, f"no pool column found; obs columns: {list(obs.columns)}"
p2pool = obs.groupby("patient_id", observed=True)[pool_col].first().astype(str)
print(f"pool column: {pool_col} | {p2pool.nunique()} pools", flush=True)

rows = []
for seed in FITS:
    design = set(pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str))
    for pname, f in PREDS.items():
        fp = WORK / f.format(seed=seed)
        if not fp.exists(): print(f"skip {pname} {seed}", flush=True); continue
        d = np.load(fp, allow_pickle=True)
        y, lo, hi = d["y"], d["lo"], d["hi"]; pats = d["patient"].astype(str); prot = d["protein"].astype(str)
        A = set(json.load(open(WORK / f"sentinels_fit{seed}.json")).get("ridge", []))
        E = np.array([j for j, p in enumerate(prot) if p not in A])
        order = np.argsort(pats, kind="stable"); y, lo, hi, pats = y[order], lo[order], hi[order], pats[order]
        upat, starts = np.unique(pats, return_index=True)
        bounds = list(starts) + [len(pats)]; sl = {p: slice(bounds[i], bounds[i + 1]) for i, p in enumerate(upat)}
        pool = np.array([p2pool.get(p, "NA") for p in upat])
        held = np.array([p for p in upat])                                          # all in npz are held-out
        # per-participant cell-score quantiles (uncorrected intervals; correction optional and omitted —
        # the question here is the calibration boundary, not the correction)
        sc = np.stack([np.quantile(np.maximum(lo[sl[p]][:, E] - y[sl[p]][:, E], y[sl[p]][:, E] - hi[sl[p]][:, E]),
                                   GAMMA, axis=0, method="higher") for p in upat])
        for P in [q for q in np.unique(pool) if q != "NA"]:
            test_i = np.where((pool == P) & ~np.isin(upat, list(design)))[0]
            other_i = np.where((pool != P) & ~np.isin(upat, list(design)))[0]
            if len(test_i) < MIN_POOL or len(other_i) < N_CAL: continue
            ge90 = b85 = wid = 0.
            for r in range(N_REPS):
                cal = RNG.choice(other_i, N_CAL, replace=False)
                k = int(np.ceil((N_CAL + 1) * (1 - 0.2)))
                lam = np.sort(sc[cal], 0)[k - 1]
                cov = np.stack([((y[sl[upat[t]]][:, E] >= lo[sl[upat[t]]][:, E] - lam) &
                                 (y[sl[upat[t]]][:, E] <= hi[sl[upat[t]]][:, E] + lam)).mean(0) for t in test_i])
                ge90 += (cov >= GAMMA).mean() / N_REPS
                b85 += (cov < 0.85).mean() / N_REPS
                wid += np.median((hi[sl[upat[test_i[0]]]][:, E] + lam) - (lo[sl[upat[test_i[0]]]][:, E] - lam)) / N_REPS
            rows.append(dict(fit=seed, predictor=pname, held_out_pool=P, n_test=len(test_i),
                             frac_ge90=round(ge90, 3), below_85=round(b85, 3), median_width=round(wid, 2)))
            print(f"  {pname} {seed} pool {P}: n={len(test_i)} ge90={ge90:.3f} below85={b85:.3f}", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "pool_heldout.csv", index=False)
pd.set_option("display.width", 200)
print("\nPool-held-out calibration (participant-block, target 0.81 under random splits — NOT guaranteed here):")
print(R.to_string(index=False))
print("\nmean over pools and fits:")
print(R.groupby("predictor")[["frac_ge90", "below_85"]].mean().round(3).to_string())
print("\nsaved pool_heldout.csv")
