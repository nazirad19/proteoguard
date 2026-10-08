# %% Cell PA - provenance audit: export the exact participant IDs and protein lists behind every analysis
# Writes provenance_participants.csv and provenance_proteins.csv per cohort folder, and prints the counts
# the manuscript states (Section 3.5 splits, Section 4.2 mechanism sets, Table 3/4 evaluation-protein counts).
import os, re, json
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad

COHORTS = {"stephenson": Path(os.environ.get("STEPH_DIR_MAIN", Path.home() / "projects" / "stephenson")),
           "combat": Path(os.environ.get("COMBAT_DIR", Path.home() / "projects" / "combat"))}
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
MIN_CELLS = 20
ISO = re.compile(r"(?i)(isotyp|igg1|igg2|igg_|iggfc|mouse_?ig|rat_?ig|hamster)")

for coh, WORK in COHORTS.items():
    if not (WORK / f"pred_fit{FITS[0]}.npz").exists(): print(f"== {coh}: no predictions, skipped =="); continue
    print(f"\n================ {coh} ================")
    obs = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad", backed="r").obs
    obs = obs[obs["adt_total"].to_numpy() > 0]
    gb = obs.groupby("patient_id", observed=True); meta = gb.agg(site=("Site", "first"))
    for c, k in [("Status", "status"), ("Pool", "pool")]:
        if c in obs: meta[k] = gb[c].agg(lambda x: x.value_counts().index[0])
    prows, prot_rows = [], []
    for seed in FITS:
        D = np.load(WORK / f"pred_fit{seed}.npz", allow_pickle=True)
        prot = D["protein"].astype(str); pat = D["patient"].astype(str); ct = D["celltype"].astype(str)
        held = pd.Index(np.unique(pat))
        train = meta.index.astype(str).difference(held)
        design = set(pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str))
        site = meta["site"].astype(str)
        # roles exactly as the analysis cells assign them
        def role_of(p):
            if p in train: return "train"
            if p in design: return "design"
            if coh == "stephenson" and site.get(p) == "Sanger": return "external_sanger"
            return "calibration_test_pool"
        roles = {p: role_of(p) for p in meta.index.astype(str)}
        # Section 4.2 mechanism sets: balanced offsets need >= MIN_CELLS in each common cell type
        df = pd.DataFrame({"p": pat, "ct": ct}); size = df.groupby(["p", "ct"]).size()
        okpair = size[size >= MIN_CELLS].reset_index()
        common = okpair.groupby("ct")["p"].nunique() / len(held)
        common = set(common[common >= 0.8].index)
        in_R = sorted(okpair[okpair["ct"].isin(common)]["p"].unique())
        pca_design = [p for p in in_R if p in design]            # Stephenson L2 PCA set; COMBAT K3 uses all of in_R
        for p in meta.index.astype(str):
            prows.append(dict(fit=seed, participant=p, role=roles[p], site=site.get(p),
                              pool=str(meta["pool"].get(p)) if "pool" in meta else "",
                              status=str(meta["status"].get(p)) if "status" in meta else "",
                              in_mechanism_offsets=p in set(in_R), in_design_pca=p in set(pca_design)))
        rc = pd.Series({r: sum(1 for p in meta.index.astype(str) if roles[p] == r) for r in
                        ["train", "design", "calibration_test_pool", "external_sanger"]})
        print(f"[fit {seed}] roles:", rc.to_dict())
        print(f"  mechanism (balanced-offset) participants: {len(in_R)} of {len(held)} held-out "
              f"(design subset used for Stephenson PCA: {len(pca_design)})")
        # protein lists: anchors per predictor, isotypes, decision-marker exclusions, evaluation set E
        sj = WORK / f"sentinels_fit{seed}.json"
        anch = json.load(open(sj)) if sj.exists() else {}
        iso = [p for p in prot if ISO.search(p)]
        union = sorted(set(sum(anch.values(), [])))
        E = [p for p in prot if p not in set(union) | set(iso)]
        print(f"  anchors: {dict((k, len(v)) for k, v in anch.items())} | union {len(union)} | isotypes {len(iso)}"
              f" | evaluation proteins |E| = {len(E)} (of {len(prot)})")
        for p in prot:
            prot_rows.append(dict(fit=seed, protein=p, isotype=p in set(iso),
                                  **{f"anchor_{k}": p in set(v) for k, v in anch.items()},
                                  in_evaluation_set=p in set(E)))
    pd.DataFrame(prows).to_csv(WORK / "provenance_participants.csv", index=False)
    pd.DataFrame(prot_rows).to_csv(WORK / "provenance_proteins.csv", index=False)
    print("wrote", WORK / "provenance_participants.csv", "and provenance_proteins.csv")
