# %% Cell T1 - settings and data
import os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, anndata as ad
import scvi
from scvi.model import TOTALVI
from scvi.distributions import NegativeBinomialMixture
from sklearn.model_selection import train_test_split

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
DATA = WORK / "stephenson_sub1000.h5ad"
SMOKE = os.environ.get("STEPH_SMOKE") == "1"
FIT_SEED = int(os.environ.get("STEPH_FIT", "2026"))
TRAIN = dict(max_epochs=200, lr=4e-3, batch_size=512, early_stopping_patience=20)
N_DRAWS, ALPHA, DELTA = 100, 0.10, 0.20       # 90% intervals; donor_quantile: P(new donor >= 90%) >= 80%
if SMOKE:
    TRAIN.update(max_epochs=2); N_DRAWS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

a = ad.read_h5ad(DATA)
a = a[a.obs["adt_total"].to_numpy() > 0].copy()                 # drop cells without ADT signal
truth = a.obsm["protein_counts"].copy()
pat = a.obs["patient_id"].astype(str).to_numpy()

# donors: half train (RNA + protein), half held out (protein hidden), stratified by site
p_site = a.obs.groupby("patient_id", observed=True)["Site"].first().astype(str)
train_p, held_p = train_test_split(p_site.index.to_numpy(), test_size=0.5, random_state=FIT_SEED, stratify=p_site.to_numpy())
role = np.where(np.isin(pat, train_p), "TRAIN", "HELD")
a.obs["role"] = pd.Categorical(role, categories=["TRAIN", "HELD"])
counts = truth.copy(); counts.loc[role == "HELD", :] = 0.0
a.obsm["protein_counts"] = counts
assert float(counts.loc[role == "HELD"].to_numpy().sum()) == 0.0
print("device:", DEVICE, "| cells:", a.n_obs, "| proteins:", truth.shape[1],
      "| train donors:", len(train_p), "| held donors:", len(held_p))
print(pd.crosstab(a.obs["Site"], a.obs["role"]))

# %% Cell T2 - train totalVI once (held donors contribute RNA only, as in the paper's setup)
model_dir = WORK / ("model_smoke" if SMOKE else f"model_fit{FIT_SEED}")
TOTALVI.setup_anndata(a, batch_key="role", categorical_covariate_keys=["Site"], layer="counts",
                      protein_expression_obsm_key="protein_counts")
if (model_dir / "model.pt").exists():
    m = TOTALVI.load(str(model_dir), adata=a)
else:
    scvi.settings.seed = FIT_SEED; t0 = time.time()
    m = TOTALVI(a, n_latent=20, n_hidden=256, n_layers_encoder=2, n_layers_decoder=1, gene_likelihood="nb",
                latent_distribution="normal", empirical_protein_background_prior=False)
    m.train(max_epochs=TRAIN["max_epochs"], lr=TRAIN["lr"], batch_size=TRAIN["batch_size"],
            accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1, train_size=0.9, validation_size=0.1,
            early_stopping=True, early_stopping_patience=TRAIN["early_stopping_patience"],
            check_val_every_n_epoch=1, reduce_lr_on_plateau=True, adversarial_classifier=True,
            enable_progress_bar=False)
    m.save(str(model_dir), overwrite=True)
    print(f"trained {m.history['elbo_train'].shape[0]} epochs in {(time.time() - t0) / 60:.1f} min")

# %% Cell T3 - posterior-predictive intervals for held-out donors (py_norm_, decoded in TRAIN batch)
pred_file = WORK / ("pred_smoke.npz" if SMOKE else f"pred_fit{FIT_SEED}.npz")
held = np.where(role == "HELD")[0]
held = held[np.argsort(pat[held], kind="stable")]              # cells grouped by donor

@torch.no_grad()
def predictive(m, adata, idx, n_draws):
    m.module.eval(); out = {k: [] for k in ["lo", "hi", "mean", "sd"]}
    for t in m._make_data_loader(adata=adata, indices=idx, batch_size=128, shuffle=False):
        _, gen = m.module.forward(t, inference_kwargs={"n_samples": n_draws},
                                  generative_kwargs={"transform_batch": 0}, compute_loss=False)
        pn = gen["py_norm_"]
        y = torch.log1p(NegativeBinomialMixture(mu1=pn["rate_back"], mu2=pn["rate_fore"], theta1=pn["r"],
                                                mixture_logits=pn["mixing"]).sample())
        q = torch.quantile(y, torch.tensor([ALPHA / 2, 1 - ALPHA / 2], device=y.device), dim=0)
        mix = torch.sigmoid(pn["mixing"])
        out["lo"].append(q[0].cpu().numpy()); out["hi"].append(q[1].cpu().numpy())
        out["sd"].append(y.std(0).cpu().numpy())
        out["mean"].append(torch.log1p((pn["rate_back"] * mix + pn["rate_fore"] * (1 - mix)).mean(0)).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}

if not pred_file.exists():
    t0 = time.time(); P = predictive(m, a, held, N_DRAWS)
    y = np.log1p(truth.to_numpy(np.float32)[held])
    np.savez_compressed(pred_file, y=y, patient=pat[held], site=a.obs["Site"].astype(str).to_numpy()[held],
                        celltype=a.obs["initial_clustering"].astype(str).to_numpy()[held],
                        protein=truth.columns.to_numpy(), **P)
    print(f"predictions for {len(held)} held-out cells in {(time.time() - t0) / 60:.1f} min")

# %% Cell T4 - calibration experiment: resample which held-out donors calibrate and which are tested
D = np.load(pred_file, allow_pickle=True)
y, lo, hi, mu = D["y"], D["lo"], D["hi"], D["mean"]
S = np.maximum(lo - y, y - hi)                                  # >0 = outside the raw interval (log1p units)
W0 = hi - lo
donors, start = np.unique(D["patient"], return_index=True)
order = np.argsort(start); donors, start = donors[order], start[order]
stop = np.r_[start[1:], len(y)]; ncell = stop - start
nD, nP = len(donors), y.shape[1]
print("held-out donors:", nD, "| proteins:", nP, "| cells per donor (median):", int(np.median(ncell)))
print("RMSLE (likelihood mean):", round(float(np.sqrt(((mu - y) ** 2).mean(0)).mean()), 4))

# Precompute, per donor and protein, on a fine per-protein grid of interval adjustments lam:
#   F[i,g,j] = fraction of donor i's cells covered when the raw interval is widened by GRID[g,j]
#   E[i,g,j] = mean amount by which donor i's cells fall outside that interval (for the interval score)
GRID = np.sort(np.vstack([np.quantile(S, np.linspace(0, 1, 1601), axis=0), np.zeros((1, nP))]), axis=0)
G = GRID.shape[0]
F = np.zeros((nD, G, nP), np.float32); E = np.zeros((nD, G, nP), np.float32)
for i, (s0, e0) in enumerate(zip(start, stop)):
    for j in range(nP):
        v = np.sort(S[s0:e0, j]); n = len(v); tail = np.r_[np.cumsum(v[::-1])[::-1], 0.0]
        k = np.searchsorted(v, GRID[:, j], side="right")
        F[i, :, j] = k / n
        E[i, :, j] = (tail[k] - (n - k) * GRID[:, j]) / n
DQ = np.stack([np.quantile(S[s0:e0], 1 - ALPHA, axis=0, method="higher") for s0, e0 in zip(start, stop)])
MEDW = np.stack([np.median(W0[s0:e0], 0) for s0, e0 in zip(start, stop)])
MEANW = np.stack([W0[s0:e0].mean(0) for s0, e0 in zip(start, stop)])
cols = np.arange(nP)

def g_index(lam):                       # largest grid point <= lam (slightly conservative for coverage)
    return np.array([max(np.searchsorted(GRID[:, j], lam[j], side="right") - 1, 0) for j in cols])

def lam_cellwise(cal):                  # pooled cells, as if independent
    pooled = (F[cal] * ncell[cal, None, None]).sum(0) / ncell[cal].sum()
    N = ncell[cal].sum(); level = min(np.ceil((N + 1) * (1 - ALPHA)) / N, 1.0)
    return GRID[np.argmax(pooled >= level, 0), cols]

def lam_donor_crc(cal):                 # E[new donor miscoverage] <= ALPHA
    risk = ((1 - F[cal]).sum(0) + 1) / (len(cal) + 1)
    ok = risk <= ALPHA
    return np.where(ok.any(0), GRID[np.argmax(ok, 0), cols], np.inf)

def lam_donor_quantile(cal):            # P(new donor coverage >= 1-ALPHA) >= 1-DELTA
    K = len(cal); k = int(np.ceil((K + 1) * (1 - DELTA)))
    return np.sort(DQ[cal], 0)[k - 1] if k <= K else np.full(nP, np.inf)

METHODS = {"raw": lambda cal: np.zeros(nP), "cellwise": lam_cellwise,
           "donor_crc": lam_donor_crc, "donor_quantile": lam_donor_quantile}
K_LIST = [k for k in [5, 10, 20, 30] if k <= nD - 10]
N_REPS = 20 if SMOKE else 500
MAXK = max(K_LIST)
rng = np.random.default_rng(0); rows = []; t0 = time.time()
for rep in range(N_REPS):
    perm = rng.permutation(nD); pool, test = perm[:MAXK], perm[MAXK:]
    for K in K_LIST:
        cal = pool[:K]
        for meth, fn in METHODS.items():
            lam = fn(cal); fin = np.isfinite(lam)
            g = g_index(np.where(fin, lam, 0))
            cov = np.where(fin, F[test][:, g, cols], 1.0)                       # (test donors, proteins)
            isc = np.where(fin, MEANW[test] + 2 * lam + (2 / ALPHA) * E[test][:, g, cols], np.nan)
            wid = np.where(fin, MEDW[test] + 2 * lam, np.nan)
            rows.append(dict(rep=rep, K=K, method=meth, mean_coverage=cov.mean(),
                             below_90=(cov < 0.90).mean(), below_85=(cov < 0.85).mean(), below_80=(cov < 0.80).mean(),
                             donor_mean_below_85=(cov.mean(1) < 0.85).mean(),
                             median_width=np.nanmedian(wid) if fin.any() else np.nan,
                             interval_score=np.nanmean(isc) if fin.any() else np.nan,
                             infinite_frac=1 - fin.mean()))
R = pd.DataFrame(rows)
R.to_csv(WORK / ("calib_smoke.csv" if SMOKE else f"calib_fit{FIT_SEED}.csv"), index=False)
print(f"{N_REPS} resamples done in {(time.time() - t0) / 60:.1f} min")

# %% Cell T5 - summary (averaged over resamples; each row = new-donor x protein outcomes)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)
print(R.groupby(["K", "method"]).mean(numeric_only=True).drop(columns="rep").round(3))
