# %% Cell FIG - all paper figures from the saved result files (Stephenson + Hao)
import os
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; import matplotlib.pyplot as plt
from scipy.stats import spearmanr

S = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
H = Path(os.environ.get("HAO_DIR", Path.home() / "projects" / "hao"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
CALIB = os.environ.get("STEPH_CALIB", "calib_fit{seed}.csv")
PRED = os.environ.get("STEPH_PRED", "pred_fit{seed}.npz")
OUT = S / "figures"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300})
FITNAME = {f: f"fit {i + 1}" for i, f in enumerate(FITS)}; MK = dict(zip(FITS, ["o", "s"]))
COL = {"cell-wise conformal": "#ff8c00", "patient-block, no anchors": "#00b050", "ProteoGuard": "#1f5fff",
       "raw": "#7b2cff", "cellwise": "#ff8c00", "donor_crc": "#00b0f0", "donor_quantile": "#00b050"}
SITE_COL = {"Cambridge": "#1f5fff", "Ncl": "#00b050", "Sanger": "#ff8c00"}

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.show(); plt.close(fig)
    print("saved", OUT / name)

def panel(fn):                                           # run each figure independently; report what is missing
    try: fn()
    except FileNotFoundError as e: print("skipped", fn.__name__, "- missing", e.filename)

# ---------- Figure 2: standard calibration fails for new patients ----------
def fig2():
    C = pd.concat([pd.read_csv(S / CALIB.format(seed=s)).assign(fit=s) for s in FITS])
    C.loc[C["infinite_frac"] > 0, "below_85"] = np.nan
    names = {"raw": "totalVI intervals (uncalibrated)", "cellwise": "Cell-wise conformal", "donor_crc": "Patient-block (mean risk)",
             "donor_quantile": "Patient-block (quantile)"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.2]})
    for m, n in names.items():
        g = C[C.method == m].groupby(["K", "fit"])["below_85"].mean().reset_index()
        for f in FITS:
            gf = g[g.fit == f]; ax[0].plot(gf.K, gf.below_85, marker=MK[f], color=COL[m], lw=1.5, label=n if f == FITS[0] else None)
    ax[0].set_xlabel("Calibration patients"); ax[0].set_ylabel("Patient × protein pairs below 85% coverage")
    ax[0].set_title("Undercoverage vs calibration patients")
    ax[0].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax[0].text(0.99, 0.02, "mean-risk method: no finite interval with 5 patients", transform=ax[0].transAxes, ha="right", fontsize=6.5, color="#555")
    sr = pd.read_csv(S / "site_results.csv")
    labs = ["raw | pooled", "cellwise | pooled", "donor_quantile | pooled", "donor_quantile | same_site"]
    lname = ["Uncalibrated", "Cell-wise", "Patient-block, pooled", "Patient-block, same site"]
    lcol = ["#7b2cff", "#ff8c00", "#00b0f0", "#00b050"]
    for s_i, site in enumerate(["Cambridge", "Ncl"]):
        for l_i, lab in enumerate(labs):
            v = sr[(sr.site == site) & (sr.label == lab)]; x = s_i * 5 + l_i
            ax[1].bar(x, v.below_85.mean(), color=lcol[l_i], label=lname[l_i] if s_i == 0 else None)
            for _, r in v.iterrows(): ax[1].scatter(x, r.below_85, color="black", marker=MK.get(r.fit, "o"), s=14, zorder=3)
    ax[1].set_xticks([1.5, 6.5], ["Cambridge", "Ncl"]); ax[1].set_ylabel("Patient × protein pairs below 85% coverage")
    ax[1].set_title("Undercoverage by site (10 calibration pts)")
    ax[1].legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "fig2_calibration_failure")

# ---------- Figure 3: one patient-level factor ----------
def fig3():
    seed = FITS[0]
    D = np.load(S / PRED.format(seed=seed), allow_pickle=True)
    design = pd.read_csv(S / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    df = pd.DataFrame(D["y"] - D["mean"], columns=D["protein"].astype(str)); df["p"] = D["patient"].astype(str); df["ct"] = D["celltype"].astype(str)
    size = df.groupby(["p", "ct"]).size(); med = df.groupby(["p", "ct"]).median()[size >= 20]
    pres = med.reset_index().groupby("ct")["p"].nunique() / df["p"].nunique()
    R = med[med.index.get_level_values("ct").isin(pres[pres >= 0.8].index)].groupby(level="p").mean()
    R = R.loc[:, R.std(axis=0) > 1e-6]; pats = R.index.to_numpy(); isd = np.isin(pats, design)
    Xd = R.values[isd]; mu = Xd.mean(0)
    sh = lambda M: (lambda s: s / s.sum())(np.linalg.svd(M - M.mean(0), compute_uv=False) ** 2)
    rng = np.random.default_rng(0); obs = sh(Xd)
    null = np.quantile([sh(np.column_stack([rng.permutation(c) for c in Xd.T]))[:6] for _ in range(300)], 0.95, axis=0)
    V = np.linalg.svd(Xd - mu, full_matrices=False)[2]; pc1 = (R.values - mu) @ V[0]
    site = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(pats).to_numpy()
    adt = None
    if (S / "stephenson_sub1000.h5ad").exists():
        import anndata as ad
        o = ad.read_h5ad(S / "stephenson_sub1000.h5ad", backed="r").obs
        adt = np.log10(o.groupby("patient_id", observed=True)["adt_total"].median().reindex(pats).astype(float).to_numpy())
    L4 = pd.read_csv(S / f"lowrank_reconstruction_fit{seed}.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    k = np.arange(1, 7); ax[0].bar(k - 0.18, obs[:6], 0.36, color="#1f5fff", label="Observed"); ax[0].bar(k + 0.18, null, 0.36, color="#9aa3ad", label="Permutation 95th pct")
    ax[0].set_xticks(k); ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Variance explained")
    ax[0].set_title("Patient-level error, design patients"); ax[0].legend(fontsize=7, frameon=False)
    if adt is not None:
        for st, c in SITE_COL.items():
            m = site == st; ax[1].scatter(adt[m], pc1[m], color=c, s=22, label=st, edgecolor="white", lw=0.4)
        ax[1].set_xlabel("log10 median ADT counts per cell"); ax[1].set_ylabel("PC1 score")
        ax[1].set_title(f"PC1 vs ADT depth (Spearman ρ = {spearmanr(adt, pc1).statistic:.2f})"); ax[1].legend(fontsize=7, frameon=False)
    lc = {"shared_offset": "#1f5fff", "rank1": "#00b050", "rank2": "#e6b800", "ridge_map": "#ff8c00"}
    ln = {"shared_offset": "Shared offset", "rank1": "Rank-1", "rank2": "Rank-2", "ridge_map": "Ridge map"}
    for m, g in L4.groupby("method"):
        q = g.groupby("n_anchor")["heldout_R2"].agg(["median", lambda x: np.quantile(x, 0.1), lambda x: np.quantile(x, 0.9)])
        ax[2].plot(q.index, q["median"], "o-", color=lc[m], label=ln[m]); ax[2].fill_between(q.index, q.iloc[:, 1], q.iloc[:, 2], color=lc[m], alpha=0.12)
    ax[2].set_xticks(sorted(L4.n_anchor.unique())); ax[2].set_ylim(-0.2, 0.9); ax[2].set_xlabel("Anchor proteins"); ax[2].set_ylabel("Held-out R²")
    ax[2].set_title("Non-anchor error predicted from anchors"); ax[2].legend(fontsize=7, frameon=False, loc="lower right")
    save(fig, "fig3_patient_factor")

# ---------- Figure 4: ProteoGuard main result (per-patient) ----------
def fig4():
    R = pd.read_csv(S / "proteoguard_patient_metrics.csv")
    pm = R.groupby(["fit", "predictor", "method", "patient"])[["below_85", "interval_score", "median_width"]].mean().reset_index()
    meths = ["cell-wise conformal", "patient-block, no anchors", "ProteoGuard"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, (metric, ylab) in zip(ax, [("below_85", "Proteins below 85% coverage (per patient)"), ("interval_score", "Interval score (per patient)"),
                                      ("median_width", "Median interval width (log1p)")]):
        x0 = 0; ticks, tl = [], []
        for pred in ["totalVI", "ridge"]:
            for f in FITS:
                if not ((pm.predictor == pred) & (pm.fit == f)).any(): continue
                for m_i, m in enumerate(meths):
                    v = pm[(pm.predictor == pred) & (pm.fit == f) & (pm.method == m)][metric].to_numpy()
                    if len(v) == 0: continue
                    x = x0 + m_i
                    a.boxplot(v, positions=[x], widths=0.6, showfliers=False, patch_artist=True,
                              boxprops=dict(facecolor=COL[m], alpha=0.35, edgecolor=COL[m]), medianprops=dict(color=COL[m]),
                              whiskerprops=dict(color=COL[m]), capprops=dict(color=COL[m]))
                    a.scatter(x + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), v, s=6, color=COL[m], alpha=0.8)
                ticks.append(x0 + 1); tl.append(f"{pred}\n{FITNAME[f]}"); x0 += 4
        a.set_xticks(ticks, tl); a.set_ylabel(ylab)
    ax[0].set_title("Undercoverage, test patients"); ax[1].set_title("Interval score, test patients"); ax[2].set_title("Interval width, test patients")
    ax[2].text(0.99, 0.02, "y-axis does not start at 0", transform=ax[2].transAxes, ha="right", fontsize=6.5, color="#555")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m], alpha=0.6) for m in meths]
    fig.legend(handles, ["Cell-wise conformal (standard)", "Patient-block, no anchors", "ProteoGuard (5 anchors)"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig4_main_per_patient")

# ---------- Figure 5: anchor panels, test vs external site ----------
def fig5():
    P = pd.concat([pd.read_csv(S / f"proteoguard_fit{s}.csv").assign(fit=s) for s in FITS])
    pan = ["ProteoGuard_optimised", "ProteoGuard_random", "ProteoGuard_abundant", "ProteoGuard_variable", "ProteoGuard_top_PC1_loading", "ProteoGuard_canonical"]
    pname = ["Optimised", "Random", "Most abundant", "Most variable", "Top PC1 loading", "Canonical"]
    pcol = ["#1f5fff", "#00b0f0", "#00b050", "#e6b800", "#7b2cff", "#ff8c00"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    for a, ev in zip(ax, sorted(P["eval"].unique(), reverse=True)):
        for p_i, pred in enumerate(["totalVI", "ridge"]):
            for k, m in enumerate(pan):
                v = P[(P["eval"] == ev) & (P.predictor == pred) & (P.method == m)].groupby("fit")["below_85"].mean()
                if v.empty: continue
                x = p_i * 7.5 + k; a.bar(x, v.mean(), color=pcol[k], label=pname[k] if p_i == 0 else None)
                for f, val in v.items(): a.scatter(x, val, color="black", marker=MK[f], s=12, zorder=3)
            ref = P[(P["eval"] == ev) & (P.predictor == pred) & (P.method == "cellwise")]["below_85"].mean()
            a.hlines(ref, p_i * 7.5 - 0.5, p_i * 7.5 + 5.5, color="#ff8c00", ls="--", lw=1.2)
        a.set_xticks([2.5, 10], ["totalVI", "ridge"]); a.set_title(ev)
    ax[0].set_ylabel("Patient × protein pairs below 85% coverage"); ax[0].legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig5_anchor_panels")

# ---------- Figure 6: biomarker decisions ----------
def fig6():
    RC = pd.read_csv(S / "risk_coverage.csv")
    order = ["ProteoGuard", "anchor-corrected point (no interval)", "patient-block, no anchors", "naive (distance to threshold)", "random abstention"]
    rcol = ["#1f5fff", "#00b0f0", "#00b050", "#ff8c00", "#9aa3ad"]
    preds = [p for p in ["totalVI", "ridge"] if p in RC.predictor.unique()]
    fig, ax = plt.subplots(1, len(preds) + 1, figsize=(5 * (len(preds) + 1), 3.5))
    ax = np.atleast_1d(ax)
    for a, pred in zip(ax, preds):
        for m, c in zip(order, rcol):
            g = RC[(RC.predictor == pred) & (RC.method == m)].sort_values("call_rate")
            a.plot(g.call_rate, g.selective_error, "o-", color=c, ms=4, label=m)
        own = RC[(RC.predictor == pred) & (RC.method == "ProteoGuard (own calls)")]
        if len(own): a.axvline(own.call_rate.iloc[0], color="#555", ls=":", lw=1)
        a.set_xlabel("Call rate"); a.set_ylabel("Error among calls made"); a.set_title(f"Risk–coverage, biomarker calls ({pred})")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08))
    M = pd.read_csv(S / "marker_calls.csv"); M = M[M.predictor == "totalVI"]
    g = M.groupby(["marker", "method"])[["error_among_confident", "ambiguous_rate"]].mean().reset_index()
    mk = [m for m in ["HLA-DR", "CD38", "CD69", "CD25", "TIGIT"] if m in g.marker.unique()]; xx = np.arange(len(mk))
    nv = [g[(g.marker == m) & (g.method == "naive point prediction")].error_among_confident.iloc[0] for m in mk]
    pg = [g[(g.marker == m) & (g.method == "ProteoGuard")].error_among_confident.iloc[0] for m in mk]
    cr = [1 - g[(g.marker == m) & (g.method == "ProteoGuard")].ambiguous_rate.iloc[0] for m in mk]
    ax[-1].bar(xx - 0.2, nv, 0.4, color="#ff8c00", label="Naive point prediction"); ax[-1].bar(xx + 0.2, pg, 0.4, color="#1f5fff", label="ProteoGuard")
    for i, c in enumerate(cr): ax[-1].text(xx[i] + 0.2, pg[i] + 0.01, f"{c:.0%}\ncalled", ha="center", fontsize=7, color="#1f5fff")
    ax[-1].set_xticks(xx, mk); ax[-1].set_ylabel("Error among calls made"); ax[-1].set_title("Named markers (totalVI)"); ax[-1].legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig6_biomarker_calls")

# ---------- Supplementary: Hao sample-level factors ----------
def figS1():
    D = np.load(H / "pred_fit2026.npz", allow_pickle=True)
    df = pd.DataFrame(D["y"] - D["mean"], columns=D["protein"].astype(str)); df["s"] = D["patient"].astype(str); df["ct"] = D["celltype"].astype(str)
    size = df.groupby(["s", "ct"]).size(); med = df.groupby(["s", "ct"]).median()[size >= 20]
    med = med[med.index.get_level_values("ct").isin(["Mono", "CD4 T", "CD8 T", "NK", "B"])]
    R = med.groupby(level="s").mean(); R = R.loc[:, R.std(axis=0) > 1e-6]; X = R.values
    V = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]; pcs = (X - X.mean(0)) @ V[:2].T
    s = R.index.to_numpy(); donor = np.array([x.split("_d")[0] for x in s]); day = np.array([x.split("_d")[1] for x in s])
    batch = pd.Series(D["site"].astype(str)).groupby(D["patient"].astype(str)).first().reindex(s).to_numpy()
    dcol = dict(zip(sorted(set(donor)), ["#1f5fff", "#00b050", "#ff8c00", "#7b2cff"])); dmk = {"0": "o", "2": "s", "7": "^"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i in range(len(s)):
        ax[0].scatter(pcs[i, 0], pcs[i, 1], color=dcol[donor[i]], marker=dmk[day[i]], s=50, edgecolor="white", lw=0.5)
    for d, c in dcol.items(): ax[0].scatter([], [], color=c, label=f"{d} ({batch[donor == d][0]})")
    for d, mk in dmk.items(): ax[0].scatter([], [], color="#555", marker=mk, label=f"day {d}")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_title("Hao: sample-level error, PC1 vs PC2"); ax[0].legend(fontsize=7, frameon=False, ncol=2)
    ax[1].scatter(pcs[:, 0], X.mean(1), color="#1f5fff", s=30); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("Mean error across proteins")
    ax[1].set_title(f"Hao: PC1 vs global shift (ρ = {spearmanr(pcs[:, 0], X.mean(1)).statistic:.2f})")
    save(fig, "figS1_hao_factors")

for f in [fig2, fig3, fig4, fig5, fig6, figS1]:
    panel(f)
