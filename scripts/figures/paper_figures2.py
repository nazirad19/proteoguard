# %% Cell FIG2 - new figures: calibration rules x corrections (both cohorts) + COMBAT mechanism
# Reads baselines_calibration.csv from the Stephenson and COMBAT folders and pred_fit2026.npz from COMBAT.
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; import matplotlib.pyplot as plt
import anndata as ad
from scipy.stats import spearmanr

S = Path(os.environ.get("STEPH_DIR_MAIN", Path.home() / "projects" / "stephenson"))
C = Path(os.environ.get("COMBAT_DIR", Path.home() / "projects" / "combat"))
OUT = S / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
RULECOL = {"cell-wise conformal": "#ff8c00", "HCP (Lee et al.)": "#7b2cff", "participant-block": "#1f5fff"}
RULENAME = {"cell-wise conformal": "Cell-wise conformal", "HCP (Lee et al.)": "HCP", "participant-block": "Participant-block"}

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.show(); plt.close(fig)
    print("saved", OUT / name)

# ---------- Figure 8: three calibration rules on identical predictions, both cohorts ----------
# main pipeline per predictor: shared map for totalVI and ridge; no post-hoc correction for ridge + sentinels
ARMS = [("totalVI", "ProteoGuard shared map"), ("ridge", "ProteoGuard shared map"), ("ridge + sentinels", "none (design mean)")]
def fig8():
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for a_i, (coh, path) in enumerate([("Stephenson", S), ("COMBAT", C)]):
        B = pd.read_csv(path / "baselines_calibration.csv")
        g = B.groupby(["predictor", "correction", "calibration"])[["frac_participants_ge90", "interval_score", "width"]].mean()
        x = np.arange(len(ARMS))
        for r_i, rule in enumerate(RULECOL):
            v = [g.loc[(p, c, rule), "frac_participants_ge90"] for p, c in ARMS]
            ax[a_i].bar(x + (r_i - 1) * 0.26, v, 0.26, color=RULECOL[rule], label=RULENAME[rule] if a_i == 0 else None)
        ax[a_i].axhline(13 / 16, color="black", ls="--", lw=1)
        ax[a_i].text(0.02, 13 / 16 + 0.015, "guaranteed level 13/16", fontsize=7, color="#333")
        ax[a_i].set_xticks(x, [p for p, _ in ARMS], fontsize=8); ax[a_i].set_ylim(0, 1.05)
        ax[a_i].set_ylabel("Test participants with ≥ 90% coverage" if a_i == 0 else "")
        ax[a_i].set_title(f"Participants reaching target coverage: {coh}")
    # width vs rule, both cohorts, main arms
    for coh, path, mk in [("Stephenson", S, "o"), ("COMBAT", C, "s")]:
        B = pd.read_csv(path / "baselines_calibration.csv")
        g = B.groupby(["predictor", "correction", "calibration"])[["width", "interval_score"]].mean()
        ax[2].scatter([], [], color="black", marker=mk, s=38, label=coh)
        for (p, c) in ARMS:
            for rule in RULECOL:
                ax[2].scatter(g.loc[(p, c, rule), "width"], g.loc[(p, c, rule), "interval_score"],
                              color=RULECOL[rule], marker=mk, s=38)
    ax[2].set_xlabel("Median interval width (log1p counts)"); ax[2].set_ylabel("Interval score")
    ax[2].set_title("Width and interval score by calibration rule")
    ax[2].legend(fontsize=7, frameon=False, title=None)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.38, -0.06))
    plt.tight_layout(); save(fig, "fig8_calibration_rules")

# ---------- Figure 9: COMBAT mechanism (participant offsets are one pool-linked factor) ----------
def fig9():
    D = np.load(C / "pred_fit2026.npz", allow_pickle=True)
    prot = D["protein"].astype(str); pat = D["patient"].astype(str); ct = D["celltype"].astype(str)
    obs = ad.read_h5ad(C / "stephenson_sub1000.h5ad", backed="r").obs
    gb = obs.groupby("patient_id", observed=True)
    pool = gb["Pool"].agg(lambda s: s.value_counts().index[0]) if "Pool" in obs else None
    df = pd.DataFrame(D["y"] - D["mean"], columns=prot); df["p"] = pat; df["ct"] = ct
    size = df.groupby(["p", "ct"]).size(); med = df.groupby(["p", "ct"]).median()[size >= 20]
    common = (med.reset_index().groupby("ct")["p"].nunique() / len(np.unique(pat)))
    common = common[common >= 0.8].index
    R = med[med.index.get_level_values("ct").isin(common)].groupby(level="p").mean()
    R = R.loc[:, R.std(axis=0) > 1e-6]; P = R.index.to_numpy()
    X = R.values - R.values.mean(0)
    def share(M): s = np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2; return s / s.sum()
    rng = np.random.default_rng(0); sh = share(X)
    null = np.stack([share(np.column_stack([rng.permutation(c) for c in X.T]))[:8] for _ in range(300)])
    V = np.linalg.svd(X, full_matrices=False)[2][0]
    if V.mean() < 0: V = -V                                                  # orient PC1 with the majority sign
    pc1 = X @ V
    depth = pd.Series(np.expm1(D["y"]).sum(1)).groupby(pat).median().reindex(P).to_numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    k = np.arange(1, 9)
    ax[0].bar(k, sh[:8], color="#1f5fff", label="observed")
    ax[0].plot(k, np.quantile(null, 0.95, 0), "k--", lw=1.2, label="permutation 95th pct")
    ax[0].set_xlabel("Component"); ax[0].set_ylabel("Fraction of variance of participant offsets")
    ax[0].set_title("Variance explained by participant-offset components"); ax[0].legend(fontsize=7, frameon=False)

    pools = sorted(pool.reindex(P).astype(str).unique()) if pool is not None else ["all"]
    PCOL = ["#1f5fff", "#ff8c00", "#00b050", "#7b2cff", "#e60000", "#00b0f0", "#996633", "#666666", "#cccc00", "#00c0a0"]
    pcol = {g: PCOL[i % len(PCOL)] for i, g in enumerate(pools)}
    grp = pool.reindex(P).astype(str).to_numpy() if pool is not None else np.array(["all"] * len(P))
    for g in pools:
        m = grp == g
        ax[1].scatter(np.log10(depth[m]), pc1[m], s=16, color=pcol[g], label=g)
    rho = spearmanr(pc1, np.log10(depth)).statistic
    ax[1].set_xlabel("log10 median ADT depth per participant"); ax[1].set_ylabel("PC1 score")
    ax[1].set_title(f"PC1 score against ADT depth (Spearman ρ = {rho:.2f})")
    ax[1].legend(fontsize=6, frameon=False, ncol=2, title="pool", title_fontsize=6)

    order = sorted(pools, key=lambda g: np.median(pc1[grp == g]))
    for i, g in enumerate(order):
        v = pc1[grp == g]
        ax[2].scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color=pcol[g])
        ax[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="black", lw=1.6)
    ax[2].set_xticks(range(len(order)), order, fontsize=7, rotation=45)
    ax[2].set_xlabel("Processing pool"); ax[2].set_ylabel("PC1 score")
    ax[2].set_title("PC1 score by processing pool")
    plt.tight_layout(); save(fig, "fig9_combat_mechanism")

for fn in [fig8, fig9]:
    try: fn()
    except (FileNotFoundError, KeyError) as e: print("skipped", fn.__name__, "-", repr(e))
