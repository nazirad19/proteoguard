# %% Cell H3 - Hao: is sample-level imputation error one technical factor, and do 5 anchors predict it?
import os
from pathlib import Path
import numpy as np, pandas as pd

WORK = Path(os.environ.get("HAO_DIR", Path.home() / "projects" / "hao"))
PRED = os.environ.get("HAO_PRED", "pred_fit2026.npz")
CTS = os.environ.get("HAO_CTS", "Mono,CD4 T,CD8 T,NK,B").split(",")
MIN_CELLS = int(os.environ.get("HAO_MINCELLS", "20"))
EXCLUDE = ["CD38", "HLA", "PDCD1", "PD-1", "PD1", "TIGIT", "IL2RA", "CD25", "CD69", "CD274", "PD-L1", "CTLA4", "ISO", "IGG", "IGM", "IGA"]

D = np.load(WORK / PRED, allow_pickle=True)
y, mu = D["y"].astype(np.float64), D["mean"].astype(np.float64)
prot = D["protein"].astype(str); samp = D["patient"].astype(str); ct = D["celltype"].astype(str)
df = pd.DataFrame(y - mu, columns=prot); df["s"] = samp; df["ct"] = ct
size = df.groupby(["s", "ct"]).size(); med = df.groupby(["s", "ct"]).median()[size >= MIN_CELLS]
med = med[med.index.get_level_values("ct").isin(CTS)]
R = med.groupby(level="s").mean(); R = R.loc[:, R.std(axis=0) > 1e-6]
samples = R.index.to_numpy(); donor = np.array([s.split("_d")[0] for s in samples])
day = np.array([float(s.split("_d")[1]) if "_d" in s else np.nan for s in samples])
X = R.values; P = X.shape[1]
print(f"held-out samples: {len(samples)} from {len(np.unique(donor))} donors | proteins: {P}")

# 1) low rank? (all held-out samples; permutation null)
def share(M):
    s = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2; return s / s.sum()
sh = share(X); rng = np.random.default_rng(0)
null = np.array([share(np.column_stack([rng.permutation(c) for c in X.T]))[:4] for _ in range(500)])
print("\nVariance explained (PC1-4):", np.round(sh[:4], 3), "| permutation 95th pct:", np.round(np.quantile(null, 0.95, 0), 3))
V = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
print("PC1: fraction of proteins with same sign:", round(float((np.sign(V[0]) == np.sign(V[0].mean())).mean()), 3))

# 2) what does PC1 track? ADT depth vs vaccination day
pc1 = (X - X.mean(0)) @ V[0]
adt = pd.Series(np.expm1(D["y"]).sum(1)).groupby(samp).median().reindex(samples).to_numpy()
from scipy.stats import spearmanr
print("PC1 vs log ADT depth: rho =", round(spearmanr(pc1, np.log10(adt)).statistic, 3),
      "| PC1 vs day: rho =", round(spearmanr(pc1, day).statistic, 3) if np.isfinite(day).all() else "n/a")

# 3) anchors predict unseen donors' errors (leave-one-donor-out; random anchors, shared-offset map)
ok = np.array([not any(e in p.upper() for e in EXCLUDE) for p in R.columns])
detected = (D["y"] > 0).mean(0)[[list(prot).index(p) for p in R.columns]] >= 0.5
cand = np.where(ok & detected)[0]
rows = []
for n_a in [1, 3, 5, 10]:
    for rep in range(100):
        A = rng.choice(cand, n_a, replace=False); O = np.setdiff1d(np.arange(P), A)
        err = base = 0.0
        for d in np.unique(donor):
            te, tr = donor == d, donor != d
            x_tr, x_te = X[tr][:, A].mean(1), X[te][:, A].mean(1)
            B = np.linalg.lstsq(np.c_[np.ones(tr.sum()), x_tr], X[tr][:, O], rcond=None)[0]
            pred = B[0] + np.outer(x_te, B[1])
            err += ((X[te][:, O] - pred) ** 2).sum(); base += ((X[te][:, O] - X[tr][:, O].mean(0)) ** 2).sum()
        rows.append(dict(n_anchor=n_a, R2=1 - err / base))
L = pd.DataFrame(rows)
print("\nLeave-one-donor-out R2 of non-anchor errors from random anchors (median, 10th pct):")
print(L.groupby("n_anchor")["R2"].agg(["median", lambda r: np.quantile(r, 0.1)]).round(3))
