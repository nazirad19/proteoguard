# %% Cell F1 - recompute site-level results for every saved fit, then plot
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
CALIB = os.environ.get("STEPH_CALIB", "calib_fit{seed}.csv")
K_SITE = int(os.environ.get("STEPH_KSITE", "10"))
ALPHA, DELTA = 0.10, 0.20

def tables(pred_file):
    D = np.load(pred_file, allow_pickle=True)
    y, lo, hi = D["y"], D["lo"], D["hi"]
    S = np.maximum(lo - y, y - hi); W0 = hi - lo
    donors, start = np.unique(D["patient"], return_index=True)
    o = np.argsort(start); donors, start = donors[o], start[o]
    stop = np.r_[start[1:], len(y)]; ncell = stop - start; nD, nP = len(donors), y.shape[1]
    GRID = np.sort(np.vstack([np.quantile(S, np.linspace(0, 1, 1601), axis=0), np.zeros((1, nP))]), axis=0)
    F = np.zeros((nD, GRID.shape[0], nP), np.float32)
    for i, (s0, e0) in enumerate(zip(start, stop)):
        for j in range(nP):
            F[i, :, j] = np.searchsorted(np.sort(S[s0:e0, j]), GRID[:, j], side="right") / (e0 - s0)
    DQ = np.stack([np.quantile(S[s0:e0], 1 - ALPHA, axis=0, method="higher") for s0, e0 in zip(start, stop)])
    MEDW = np.stack([np.median(W0[s0:e0], 0) for s0, e0 in zip(start, stop)])
    site = pd.Series(D["site"]).groupby(D["patient"]).first().reindex(donors).to_numpy()
    return dict(F=F, GRID=GRID, DQ=DQ, MEDW=MEDW, ncell=ncell, site=site, nD=nD, nP=nP)

def lam_for(T, method, cal):
    F, GRID, nP = T["F"], T["GRID"], T["nP"]; cols = np.arange(nP)
    if method == "raw":
        return np.zeros(nP)
    if method == "cellwise":
        n = T["ncell"][cal]; pooled = (F[cal] * n[:, None, None]).sum(0) / n.sum()
        level = min(np.ceil((n.sum() + 1) * (1 - ALPHA)) / n.sum(), 1.0)
        return GRID[np.argmax(pooled >= level, 0), cols]
    K = len(cal); k = int(np.ceil((K + 1) * (1 - DELTA)))       # donor_quantile
    return np.sort(T["DQ"][cal], 0)[k - 1] if k <= K else np.full(nP, np.inf)

def coverage(T, lam, test):
    cols = np.arange(T["nP"])
    g = np.array([max(np.searchsorted(T["GRID"][:, j], lam[j], side="right") - 1, 0) for j in cols])
    return T["F"][test][:, g, cols]

SCHEMES = [("raw", "pooled"), ("cellwise", "pooled"), ("donor_quantile", "pooled"), ("donor_quantile", "same_site")]
rows = []
for seed in FITS:
    T = tables(WORK / PRED.format(seed=seed)); rng = np.random.default_rng(2)
    for rep in range(300):
        for site in ["Ncl", "Cambridge"]:
            pool = rng.permutation(np.where(T["site"] == site)[0])
            cal_site, test = pool[:K_SITE], pool[K_SITE:]
            cal_pooled = rng.choice(np.setdiff1d(np.arange(T["nD"]), test), K_SITE, replace=False)
            for meth, scheme in SCHEMES:
                lam = lam_for(T, meth, cal_site if scheme == "same_site" else cal_pooled)
                cov = coverage(T, lam, test)
                rows.append(dict(fit=seed, site=site, label=f"{meth} | {scheme}", below_85=(cov < 0.85).mean(),
                                 coverage=cov.mean(), width=np.median(T["MEDW"][test] + 2 * lam)))
site_res = pd.DataFrame(rows).groupby(["fit", "site", "label"]).mean().reset_index()
site_res.to_csv(WORK / "site_results.csv", index=False)
print(site_res.round(3).to_string(index=False))

# %% Cell F2 - figures
LABELS = ["raw | pooled", "cellwise | pooled", "donor_quantile | pooled", "donor_quantile | same_site"]
NAMES = {"raw | pooled": "Raw totalVI", "cellwise | pooled": "Cell-wise conformal",
         "donor_quantile | pooled": "Donor-level, pooled", "donor_quantile | same_site": "Donor-level, same site"}
COL = {"raw | pooled": "#1f5fff", "cellwise | pooled": "#ff8c00",
       "donor_quantile | pooled": "#00b050", "donor_quantile | same_site": "#7b2cff"}
MARK = {f: mk for f, mk in zip(FITS, ["o", "s", "^", "D"])}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric, ylab in [(axes[0], "below_85", "Fraction of donor x protein pairs below 85% coverage"),
                         (axes[1], "width", "Median interval width (log1p counts)")]:
    for s_i, site in enumerate(["Cambridge", "Ncl"]):
        for l_i, lab in enumerate(LABELS):
            x = s_i * (len(LABELS) + 1) + l_i
            v = site_res[(site_res["site"] == site) & (site_res["label"] == lab)]
            ax.bar(x, v[metric].mean(), color=COL[lab], width=0.8, label=NAMES[lab] if s_i == 0 else None)
            for _, r in v.iterrows():
                ax.scatter(x, r[metric], color="black", marker=MARK[r["fit"]], s=18, zorder=3)
    ax.set_xticks([1.5, 1.5 + len(LABELS) + 1], ["Cambridge", "Ncl"]); ax.set_ylabel(ylab)
axes[0].set_title("Coverage failures by site and calibration method")
axes[1].set_title("Interval width by site and calibration method")
axes[0].legend(fontsize=8, frameon=False)
fig.text(0.5, -0.02, "Bars: mean over fits; markers: individual fits (" + ", ".join(f"seed {f}" for f in FITS) + ")",
         ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(WORK / "fig_site_calibration.png", dpi=200, bbox_inches="tight"); plt.show()

C = pd.concat([pd.read_csv(WORK / CALIB.format(seed=s)).assign(fit=s) for s in FITS])
C.loc[C["infinite_frac"] > 0, "below_85"] = np.nan          # infinite intervals: not a usable method at that K
CK = {"raw": "#1f5fff", "cellwise": "#ff8c00", "donor_crc": "#00b0f0", "donor_quantile": "#7b2cff"}
NK = {"raw": "Raw totalVI", "cellwise": "Cell-wise conformal", "donor_crc": "Donor-level (mean risk)",
      "donor_quantile": "Donor-level (quantile)"}
fig, ax = plt.subplots(figsize=(5.5, 4))
for meth, col in CK.items():
    g = C[C["method"] == meth].groupby(["K", "fit"])["below_85"].mean().reset_index()
    for f in FITS:
        gf = g[g["fit"] == f]; ax.plot(gf["K"], gf["below_85"], color=col, marker=MARK[f], ms=4, lw=1.5,
                                       label=NK[meth] if f == FITS[0] else None)
ax.set_xlabel("Number of calibration donors"); ax.set_ylabel("Fraction of donor x protein pairs below 85% coverage")
ax.set_title("Coverage failures vs number of calibration donors"); ax.legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(WORK / "fig_failures_vs_K.png", dpi=200); plt.show()
