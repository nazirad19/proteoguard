# %% Cell AU - cohort flow, split counts by site, and anchor detection rates
import os
import numpy as np, pandas as pd, anndata as ad
from pathlib import Path
from sklearn.model_selection import train_test_split
WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))

full = ad.read_h5ad(WORK / "haniffa21.processed.h5ad", backed="r").obs
s = full.groupby("sample_id", observed=True).agg(patient=("patient_id", "first"), site=("Site", "first"), status=("Status", "first"))
print("ORIGINAL: participants", full["patient_id"].nunique(), "| samples", len(s))
print("samples by status:", s["status"].value_counts().to_dict())
lps_only = set(s[s.status == "LPS"].patient) - set(s[s.status != "LPS"].patient)
print("LPS samples removed:", (s.status == "LPS").sum(), "| participants with only LPS samples:", len(lps_only))

sub = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad", backed="r").obs
sub = sub[sub["adt_total"].to_numpy() > 0]
p = sub.groupby("patient_id", observed=True).agg(site=("Site", "first"), status=("Status", "first"), n_samples=("sample_id", "nunique"), cells=("Site", "size"))
print("\nANALYSED: participants", len(p), "| cells", len(sub))
print("participants by site:", p.site.value_counts().to_dict())
print("participants by status:", p.status.value_counts().to_dict())
print("participants with >1 sample (pooled as one unit):", int((p.n_samples > 1).sum()))

ANCHORS = {(2026, "totalVI"): ["AB_CD96", "AB_FAS", "AB_CD158b", "AB_SLC3A2", "AB_CD161"],
           (2027, "totalVI"): ["AB_NCR2", "AB_IL7R", "AB_SLC3A2", "AB_TNFSF9", "AB_KLRK1"],
           (2026, "ridge"): ["AB_IL21R", "AB_CD141", "AB_KLRK1", "AB_CXCR5", "AB_CD226"],
           (2027, "ridge"): ["AB_KLRK1", "AB_CXCR5", "AB_CD123", "AB_SLC3A2", "AB_CD96"]}
for seed in [2026, 2027]:
    train_p, held_p = train_test_split(p.index.to_numpy(), test_size=0.5, random_state=seed, stratify=p.site.to_numpy())
    design = pd.read_csv(WORK / f"design_split_fit{seed}.csv")["patient"].astype(str).to_numpy()
    held = p.loc[held_p]; des = held.index.isin(design)
    rest = held[~des & (held.site != "Sanger")]; ext = held[~des & (held.site == "Sanger")]
    print(f"\nFIT {seed}: train {len(train_p)} {p.loc[train_p].site.value_counts().to_dict()} | held {len(held_p)}")
    print("  design:", held[des].site.value_counts().to_dict(), "| calibration+test pool:", rest.site.value_counts().to_dict(),
          f"(15 calibration, {len(rest) - 15} test per resample) | external Sanger:", len(ext))
    for pred, f in [("totalVI", f"pred_fit{seed}.npz"), ("ridge", f"pred_ridge_fit{seed}.npz")]:
        D = np.load(WORK / f, allow_pickle=True); prot = list(D["protein"].astype(str))
        dmask = np.isin(D["patient"].astype(str), design)
        det = {a: round(float((D["y"][dmask, prot.index(a)] > 0).mean()), 3) for a in ANCHORS[(seed, pred)] if a in prot}
        print(f"  {pred} anchors and detection rate in design cells:", det)
