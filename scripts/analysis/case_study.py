# CS1: biological case study — activated CD8 T cells (HLA-DR, CD38) from RNA + a small measured panel.
# Per held-out participant: measured vs predicted marker-positive fraction in CD8 cells, with a conformal
# interval and high/low/inconclusive calls; joint HLA-DR+CD38+ evaluated separately; biology-preservation
# checks (Spearman rho, SD ratio) for the shared-map correction, with a within-CD8 offset variant.
# Frozen before running: thresholds from design GMM; tau = design median fraction; metrics MAE, coverage,
# width, call confusion; markers are not anchors. Run: STEPH_DIR=~/projects/stephenson python case_study.py
import os, json, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
MARKERS = [m.strip() for m in os.environ.get("CS_MARKERS", "HLA-DR,CD38").split(",")]
CT = os.environ.get("CS_CELLTYPE", "CD8")
PREDS = {"totalVI": "pred_fit{seed}.npz", "ridge": "pred_ridge_fit{seed}.npz",
         "ridge + sentinels": "pred_ridgeanchor_fit{seed}.npz"}
MIN_CELLS = 20; N_CAL = 15; N_REPS = 50; RNG = np.random.default_rng(11)

def gmm_threshold(v):
    v = v[np.isfinite(v)]
    if len(v) < 50: return np.nan
    g = GaussianMixture(2, random_state=0).fit(v.reshape(-1, 1))
    if abs(np.diff(g.means_.ravel())[0]) < 0.3: return np.nan                    # not bimodal
    grid = np.linspace(v.min(), v.max(), 400).reshape(-1, 1)
    up = np.argmax(g.means_.ravel())
    post = g.predict_proba(grid)[:, up]
    j = np.argmax(post > 0.5)
    return float(grid[j, 0]) if post.max() > 0.5 else np.nan

rows, pr_rows = [], []
for seed in FITS:
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    A_names = json.load(open(WORK / f"sentinels_fit{seed}.json"))["ridge"]
    for pname, f in PREDS.items():
        fp = WORK / f.format(seed=seed)
        if not fp.exists(): print(f"skip {pname} {seed}: no file", flush=True); continue
        d = np.load(fp, allow_pickle=True)
        y, mid = d["y"], d["mean"]; pats = d["patient"].astype(str); ct = d["celltype"].astype(str)
        site = d["site"].astype(str); prot = d["protein"].astype(str)
        mcols = []
        for m in MARKERS:
            hits = [j for j, p in enumerate(prot) if m.upper().replace("-", "") in p.upper().replace("-", "").replace("_", "")]
            if not hits: print(f"  marker {m} not found for {pname} {seed}", flush=True)
            else: mcols.append(hits[0])
        if len(mcols) < len(MARKERS): continue
        assert not set(np.array(prot)[mcols]) & set(A_names), "marker overlaps anchor panel"

        # participant-level offsets and shared map (anchors -> per-protein shift), design-fit only
        order = np.argsort(pats, kind="stable"); y, mid, pats, ct, site = y[order], mid[order], pats[order], ct[order], site[order]
        upat, starts = np.unique(pats, return_index=True); sl = {}
        bounds = list(starts) + [len(pats)]
        for i, p in enumerate(upat): sl[p] = slice(bounds[i], bounds[i + 1])
        off = np.stack([np.median(y[sl[p]] - mid[sl[p]], 0) for p in upat])        # participant x protein
        ct_mask = {p: ct[sl[p]] == CT for p in upat}
        offW = np.stack([np.median((y[sl[p]] - mid[sl[p]])[ct_mask[p]], 0) if ct_mask[p].sum() >= MIN_CELLS
                         else np.full(y.shape[1], np.nan) for p in upat])          # within-CD8 offsets
        aidx = [list(prot).index(a) for a in A_names if a in list(prot)]
        dz = np.isin(upat, design); x = off[:, aidx].mean(1)
        alpha, beta = {}, {}
        for tag, O in (("shared map", off), ("within-CT map", offW)):
            al = np.zeros(len(prot)); be = np.zeros(len(prot))
            okd = dz & np.isfinite(O[:, mcols[0]])
            if okd.sum() >= 5:
                X = np.c_[np.ones(okd.sum()), x[okd]]
                for j in mcols:
                    co, *_ = np.linalg.lstsq(X, O[okd, j], rcond=None); al[j], be[j] = co
            alpha[tag], beta[tag] = al, be

        # thresholds and tau from design CD8 cells only
        dcells = np.isin(pats, design) & (ct == CT)
        thr = {j: gmm_threshold(y[dcells, j]) for j in mcols}
        if any(not np.isfinite(thr[j]) for j in mcols):
            print(f"  no bimodal threshold for a marker ({pname} {seed}); using design median per cell", flush=True)
            for j in mcols:
                if not np.isfinite(thr[j]): thr[j] = float(np.median(y[dcells, j]))

        # per-participant measured and predicted fractions (single markers + joint)
        rest = [p for p in upat if p not in design and site[sl[p]][0] != "Sanger" and ct_mask[p].sum() >= MIN_CELLS]
        def fracs(p, corr_tag=None):
            m = ct_mask[p]; Y = y[sl[p]][m]; P = mid[sl[p]][m].copy()
            if corr_tag:
                i = list(upat).index(p)
                for j in mcols: P[:, j] += alpha[corr_tag][j] + beta[corr_tag][j] * x[i]
            out = {}
            for k, j in enumerate(mcols):
                out[f"meas_{MARKERS[k]}"] = (Y[:, j] > thr[j]).mean()
                out[f"pred_{MARKERS[k]}"] = (P[:, j] > thr[j]).mean()
            out["meas_joint"] = ((Y[:, mcols[0]] > thr[mcols[0]]) & (Y[:, mcols[1]] > thr[mcols[1]])).mean()
            out["pred_joint"] = ((P[:, mcols[0]] > thr[mcols[0]]) & (P[:, mcols[1]] > thr[mcols[1]])).mean()
            out["n_cells"] = int(m.sum())
            return out
        for corr in (None, "shared map", "within-CT map"):
            F = pd.DataFrame({p: fracs(p, corr) for p in rest}).T
            dF = pd.DataFrame({p: fracs(p, corr) for p in upat[dz] if ct_mask[p].sum() >= MIN_CELLS}).T
            targets = [m for m in MARKERS] + ["joint"]
            for t in targets:
                tau = float(dF[f"meas_{t}"].median()); err = (F[f"pred_{t}"] - F[f"meas_{t}"]).to_numpy()
                meas, pred = F[f"meas_{t}"].to_numpy(), F[f"pred_{t}"].to_numpy()
                from scipy.stats import spearmanr
                rho = spearmanr(pred, meas).statistic if len(meas) > 4 else np.nan
                sdr = pred.std() / meas.std() if meas.std() > 0 else np.nan
                cov = wid = 0.; conf = np.zeros(3)                                  # correct, wrong, inconclusive
                for r in range(N_REPS):
                    perm = RNG.permutation(len(rest)); cal, test = perm[:N_CAL], perm[N_CAL:]
                    k = min(int(np.ceil((len(cal) + 1) * 0.9)), len(cal))
                    q = np.sort(np.abs(err[cal]))[k - 1]
                    cov += ((meas[test] >= pred[test] - q) & (meas[test] <= pred[test] + q)).mean() / N_REPS
                    wid += 2 * q / N_REPS
                    hi, lo = pred[test] - q > tau, pred[test] + q < tau
                    truth = meas[test] > tau
                    called = hi | lo
                    conf[0] += ((hi & truth) | (lo & ~truth)).sum() / N_REPS
                    conf[1] += ((hi & ~truth) | (lo & truth)).sum() / N_REPS
                    conf[2] += (~called).sum() / N_REPS
                rows.append(dict(fit=seed, predictor=pname, correction=corr or "none", target=t,
                                 n_participants=len(rest), mae=float(np.abs(err).mean()),
                                 coverage=round(cov, 3), width=round(wid, 3), spearman_rho=round(float(rho), 3),
                                 sd_ratio=round(float(sdr), 3), tau=round(tau, 3),
                                 correct=round(conf[0], 1), wrong=round(conf[1], 1), inconclusive=round(conf[2], 1)))
            if corr == "shared map":                                               # per-participant table, once
                for p in rest:
                    fr = fracs(p, corr)
                    pr_rows.append(dict(fit=seed, predictor=pname, participant=p, n_cells=fr["n_cells"],
                                        **{k: round(v, 3) for k, v in fr.items() if k != "n_cells"}))
        print(f"  {pname} {seed} done", flush=True)

R = pd.DataFrame(rows); R.to_csv(WORK / "case_study.csv", index=False)
pd.DataFrame(pr_rows).to_csv(WORK / "case_study_participants.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print("\nCase study: activated-CD8 markers, per held-out participant (mean over fits and resamples):")
print(R.groupby(["predictor", "correction", "target"])[["mae", "coverage", "width", "spearman_rho", "sd_ratio",
      "correct", "wrong", "inconclusive"]].mean().round(3).to_string())
print("\nsaved case_study.csv + case_study_participants.csv")
