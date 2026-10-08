# %% [markdown]
# # Gate 0: are totalVI's protein prediction intervals calibrated? (SLN111, paper setup)
# Kernel: **totalVI (scvi 1.4.2)** (same environment as the modality-dropout sweep).
#
# Setup (the paper's missing-panel benchmark, plus a calibration set):
# * **SOURCE** = 80% of mouse D1, RNA + proteins used for training.
# * **CAL** = the other 20% of D1, proteins hidden (same mouse, never seen with protein).
# * **TEST** = mouse D2, proteins hidden (the paper's held-out batch).
#
# Prediction target: observed `log1p(ADT count)`. Intervals come from posterior-predictive count
# draws using `py_norm_` (includes totalVI's learned protein efficiency), decoded in the SOURCE batch.
#
# Methods compared, all at 90%:
# 1. **raw**: totalVI's own 5–95% posterior-predictive interval.
# 2. **conformal (cell-wise)**: raw interval widened/shrunk per protein using CAL cells.
#    Checked on (a) the other half of CAL (same mouse) and (b) TEST (new mouse).
#
# What to look at: if (2) gives ~90% on same-mouse cells but drops on D2, cell-wise calibration
# does not carry to a new animal/donor; that is the problem the paper solves.
#
# Quick check first: run `import os; os.environ["GATE0_SMOKE"] = "1"` in a cell before Cell 1.

# %% Cell 1 - settings
import os, hashlib, urllib.request, time
from pathlib import Path
import numpy as np, pandas as pd, torch, anndata as ad
import scvi
from scvi.model import TOTALVI
from scvi.distributions import NegativeBinomialMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

WORK = Path(os.environ.get("GATE0_DIR", Path.home() / "projects" / "totalvi_gate0"))
WORK.mkdir(parents=True, exist_ok=True)
DATA_URL = "https://raw.githubusercontent.com/YosefLab/totalVI_reproducibility/master/data/spleen_lymph_111.h5ad"
DATA_SHA = "bb7abfe9808ac4180a9551b7f6478312d0c2080d26b69503369bc3a90d1e4235"

SEEDS = [2026, 2027]
TRAIN = dict(max_epochs=500, lr=4e-3, batch_size=256, early_stopping_patience=45)
N_LATENT, N_HIDDEN, N_DRAWS = 20, 256, 100       # N_DRAWS posterior-predictive draws per cell
ALPHA = 0.10                                     # 90% intervals
SMOKE = os.environ.get("GATE0_SMOKE") == "1"
if SMOKE:
    TRAIN.update(max_epochs=2); SEEDS = [1]; N_DRAWS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| scvi", scvi.__version__, "| work dir:", WORK)

# %% Cell 2 - data (checksum-verified; reuses the sweep's download if present)
raw = WORK / "spleen_lymph_111.h5ad"
for alt in [Path.home() / "projects" / "totalvi_moddrop" / "spleen_lymph_111.h5ad"]:
    if not raw.exists() and alt.exists():
        raw.symlink_to(alt)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
if not raw.exists() or sha(raw) != DATA_SHA:
    urllib.request.urlretrieve(DATA_URL, raw)
assert sha(raw) == DATA_SHA, "checksum mismatch"

src = ad.read_h5ad(raw)
a = src[:, src.var["hvg_encode"].astype(bool).to_numpy()].copy()
names = np.asarray(src.uns["protein_names"], dtype=str); keep = ~np.char.startswith(names, "HTO")
truth = pd.DataFrame(np.asarray(src.obsm["protein_expression"][:, keep], dtype=np.float32),
                     index=a.obs_names, columns=names[keep])
ct = src.obs["cell_types"].astype(str).to_numpy()
tissue = src.obs["hash_id"].astype(str).to_numpy()
d1 = src.obs["batch_indices"].astype(int).to_numpy() == 0

strat = pd.Series(ct[d1]); strat = np.where(strat.map(strat.value_counts()) >= 10, strat, "rare")
_, cal_idx = train_test_split(np.where(d1)[0], test_size=0.2, random_state=2026, stratify=strat)
group = np.where(d1, "SOURCE", "TEST").astype(object); group[cal_idx] = "CAL"
counts = truth.copy(); counts.loc[group != "SOURCE", :] = 0.0   # CAL and TEST proteins never enter training
a.obs = pd.DataFrame({"batch": pd.Categorical(group, categories=["SOURCE", "CAL", "TEST"]),
                      "cell_type": ct, "tissue": tissue}, index=a.obs_names)
a.layers["counts"] = a.X.copy()
a.obsm["protein_counts"] = counts
assert float(counts.loc[group != "SOURCE"].to_numpy().sum()) == 0.0
print(a.obs["batch"].value_counts().to_dict(), "| proteins:", truth.shape[1])

# CAL is split in half once: CAL_A calibrates, CAL_B checks calibration on the same mouse.
rng = np.random.default_rng(0)
cal_cells = np.where(group == "CAL")[0]; rng.shuffle(cal_cells)
CAL_A, CAL_B = np.sort(cal_cells[: len(cal_cells) // 2]), np.sort(cal_cells[len(cal_cells) // 2:])
TEST = np.where(group == "TEST")[0]
SOURCE = np.where(group == "SOURCE")[0]

# %% Cell 3 - posterior-predictive draws (py_norm_, decoded in SOURCE batch)
@torch.no_grad()
def predictive(m, adata, idx, n_draws):
    """Returns (mean_log1p_expected, draw_quantiles dict, draw_sd) on the log1p scale."""
    m.module.eval()
    qs = [ALPHA / 2, 0.5, 1 - ALPHA / 2]
    lo, med, hi, sd, mean = [], [], [], [], []
    for t in m._make_data_loader(adata=adata, indices=idx, batch_size=128, shuffle=False):
        _, gen = m.module.forward(t, inference_kwargs={"n_samples": n_draws},
                                  generative_kwargs={"transform_batch": 0}, compute_loss=False)
        pn = gen["py_norm_"]
        dist = NegativeBinomialMixture(mu1=pn["rate_back"], mu2=pn["rate_fore"], theta1=pn["r"],
                                       mixture_logits=pn["mixing"])
        y = torch.log1p(dist.sample())                       # (n_draws, cells, proteins)
        q = torch.quantile(y, torch.tensor(qs, device=y.device), dim=0)
        lo.append(q[0].cpu().numpy()); med.append(q[1].cpu().numpy()); hi.append(q[2].cpu().numpy())
        sd.append(y.std(0).cpu().numpy())
        mix = torch.sigmoid(pn["mixing"])
        mean.append(torch.log1p((pn["rate_back"] * mix + pn["rate_fore"] * (1 - mix)).mean(0)).cpu().numpy())
    cat = lambda x: np.concatenate(x)
    return cat(mean), cat(lo), cat(med), cat(hi), cat(sd)

def conformal_q(lo, hi, y, alpha=ALPHA):
    """Per-protein CQR adjustment from calibration cells (cell-wise exchangeability assumed)."""
    s = np.maximum(lo - y, y - hi)                           # (cells, proteins)
    n = s.shape[0]; k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return np.sort(s, axis=0)[k - 1]                          # (proteins,)

def interval_metrics(lo, hi, y, alpha=ALPHA):
    cov = ((y >= lo) & (y <= hi))
    width = hi - lo
    score = width + (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))   # Winkler interval score
    return cov, width, score

# %% Cell 4 - train and evaluate (resumable per seed)
rows_csv = WORK / ("gate0_protein_rows_smoke.csv" if SMOKE else "gate0_protein_rows.csv")
group_csv = WORK / ("gate0_group_rows_smoke.csv" if SMOKE else "gate0_group_rows.csv")
abst_csv = WORK / ("gate0_abstention_smoke.csv" if SMOKE else "gate0_abstention.csv")
done = set(pd.read_csv(rows_csv)["seed"]) if rows_csv.exists() else set()
y_all = np.log1p(truth.to_numpy(np.float64))
prot = truth.columns.to_numpy()

for seed in SEEDS:
    if seed in done:
        continue
    t0 = time.time(); scvi.settings.seed = seed
    b = a.copy()
    TOTALVI.setup_anndata(b, batch_key="batch", layer="counts", protein_expression_obsm_key="protein_counts")
    m = TOTALVI(b, n_latent=N_LATENT, n_hidden=N_HIDDEN, n_layers_encoder=2, n_layers_decoder=1,
                gene_likelihood="nb", latent_distribution="normal", empirical_protein_background_prior=False)
    m.train(max_epochs=TRAIN["max_epochs"], lr=TRAIN["lr"], batch_size=TRAIN["batch_size"],
            accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1, train_size=0.9, validation_size=0.1,
            early_stopping=True, early_stopping_patience=TRAIN["early_stopping_patience"],
            check_val_every_n_epoch=1, reduce_lr_on_plateau=True, adversarial_classifier=True,
            enable_progress_bar=False)
    P = {name: predictive(m, b, idx, N_DRAWS) for name, idx in [("CAL_A", CAL_A), ("CAL_B", CAL_B), ("TEST", TEST)]}
    q = conformal_q(P["CAL_A"][1], P["CAL_A"][3], y_all[CAL_A])

    prow, grow = [], []
    for ev, idx in [("CAL_B", CAL_B), ("TEST", TEST)]:
        mean, lo, med, hi, sd = P[ev]; y = y_all[idx]
        rmsle = np.sqrt(((mean - y) ** 2).mean(0))
        for method, (L, H) in {"raw": (lo, hi), "conformal_cellwise": (lo - q, hi + q)}.items():
            cov, width, score = interval_metrics(L, H, y)
            for j, p in enumerate(prot):
                prow.append(dict(seed=seed, eval=ev, method=method, protein=p, coverage=cov[:, j].mean(),
                                 median_width=np.median(width[:, j]), interval_score=score[:, j].mean(),
                                 rmsle=rmsle[j]))
            # coverage by cell type x protein and by tissue x protein (groups with >= 30 cells)
            for key in ["cell_type", "tissue"]:
                lab = a.obs[key].to_numpy()[idx]
                for g in np.unique(lab):
                    sel = lab == g
                    if sel.sum() < 30: continue
                    for j, p in enumerate(prot):
                        grow.append(dict(seed=seed, eval=ev, method=method, key=key, group=g, protein=p,
                                         n=int(sel.sum()), coverage=cov[sel, j].mean()))

    # abstention on TEST: reject 20% of cells by random / posterior SD / latent distance to SOURCE
    mean, lo, med, hi, sd = P["TEST"]; y = y_all[TEST]
    cell_err = np.sqrt(((mean - y) ** 2).mean(1))
    cov_c = ((y >= lo - q) & (y <= hi + q)).mean(1)
    lat = m.get_latent_representation(b)
    nn = NearestNeighbors(n_neighbors=15).fit(lat[SOURCE])
    knn = nn.kneighbors(lat[TEST])[0].mean(1)
    scores = {"random": np.random.default_rng(seed).random(len(TEST)), "posterior_sd": sd.mean(1), "latent_knn": knn}
    arow = []
    for name, s in scores.items():
        for keep_frac in [1.0, 0.9, 0.8, 0.7, 0.5]:
            kept = s <= np.quantile(s, keep_frac)
            arow.append(dict(seed=seed, score=name, keep_frac=keep_frac, cell_rmsle=cell_err[kept].mean(),
                             conformal_coverage=cov_c[kept].mean()))

    for path, rows in [(rows_csv, prow), (group_csv, grow), (abst_csv, arow)]:
        df = pd.DataFrame(rows); df.to_csv(path, mode="a", header=not path.exists(), index=False)
    eff = float(torch.exp(m.module.log_per_batch_efficiency[:, 0]).median())
    print(f"seed {seed}: done in {(time.time() - t0) / 60:.1f} min | source efficiency median {eff:.3f}", flush=True)

# %% Cell 5 - summary
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
R = pd.read_csv(rows_csv); G = pd.read_csv(group_csv); A = pd.read_csv(abst_csv)
print("RMSLE sanity check (paper benchmark, likelihood_mean; earlier runs ~0.565 on all-D1 training):",
      R[(R["eval"] == "TEST") & (R["method"] == "raw")].groupby("seed")["rmsle"].mean().round(4).to_dict())

summ = (R.groupby(["eval", "method", "seed"])
          .agg(mean_coverage=("coverage", "mean"),
               abs_cov_error=("coverage", lambda c: np.abs(c - 0.9).mean()),
               proteins_below_85=("coverage", lambda c: (c < 0.85).mean()),
               proteins_above_95=("coverage", lambda c: (c > 0.95).mean()),
               median_width=("median_width", "mean"),
               interval_score=("interval_score", "mean"))
          .groupby(["eval", "method"]).mean().round(3))
print("\nPer-protein coverage, averaged over seeds (target 0.90):"); print(summ)

gs = (G.groupby(["eval", "method", "key", "seed"])
        .agg(groups=("coverage", "size"), worst=("coverage", "min"),
             p10=("coverage", lambda c: np.quantile(c, 0.1)), frac_below_80=("coverage", lambda c: (c < 0.8).mean()))
        .groupby(["eval", "method", "key"]).mean().round(3))
print("\nCoverage across protein x cell type and protein x tissue groups (>=30 cells):"); print(gs)

print("\nAbstention on TEST (lower cell RMSLE at the same kept fraction = better score):")
print(A.groupby(["score", "keep_frac"])[["cell_rmsle", "conformal_coverage"]].mean().unstack(0).round(4))

# %% Cell 6 - figure: per-protein coverage
import matplotlib.pyplot as plt
COLORS = {"raw": "#1f77ff", "conformal_cellwise": "#ff8c00"}      # distinct bright colours, blue first
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, ev in zip(axes, ["CAL_B", "TEST"]):
    sub = R[R["eval"] == ev].groupby(["method", "protein"])["coverage"].mean().reset_index()
    for i, (meth, col) in enumerate(COLORS.items()):
        v = sub.loc[sub["method"] == meth, "coverage"].to_numpy()
        ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=10, color=col, alpha=0.8)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color="black")
    ax.axhline(0.9, ls="--", color="grey")
    ax.set_xticks([0, 1], ["raw totalVI", "cell-wise conformal"])
    ax.set_title({"CAL_B": "Held-out D1 cells (same mouse)", "TEST": "D2 cells (new mouse)"}[ev])
axes[0].set_ylabel("90% interval coverage per protein")
fig.suptitle("Per-protein coverage of 90% intervals, SLN111"); fig.tight_layout()
fig.savefig(WORK / "gate0_coverage.png", dpi=150); plt.show()
