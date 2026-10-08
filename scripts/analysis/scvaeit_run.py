# scVAEIT given exactly the same observed sentinels: train on mosaic data, predict held-out ADTs.
# Standalone bash version of notebook cell SV1. Run from ~/projects/stephenson (or set STEPH_DIR):
#   SV_EPOCHS=200 nohup python -u scvaeit_run.py > scvaeit.log 2>&1 &
# Query condition matches ProteoGuard's: held-out participants contribute RNA + the 5 ridge sentinel proteins only.
# RNA block: log1p of library-normalised counts. ADT block: log1p raw counts (the evaluation scale), so
# get_denoised_data returns predictions directly comparable to y. Intervals: train-cell residual 5-95% band
# per protein (same convention as the ridge baseline); calibration happens downstream in B1/M3b.
import os, json, time, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(g, True)
from scVAEIT import VAEIT
from sklearn.model_selection import train_test_split

WORK = Path(os.environ.get("STEPH_DIR", Path.home() / "projects" / "stephenson"))
FITS = [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]
EPOCHS = int(os.environ.get("SV_EPOCHS", "200")); BATCH = int(os.environ.get("SV_BATCH", "512"))
print(f"scVAEIT run | data: {WORK} | fits: {FITS} | epochs: {EPOCHS} | batch: {BATCH}", flush=True)
print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)

a = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad")
a = a[a.obs["adt_total"].to_numpy() > 0].copy()
pat = a.obs["patient_id"].astype(str).to_numpy()
p_site = a.obs.groupby("patient_id", observed=True)["Site"].first().astype(str)
site_code = a.obs["Site"].astype("category").cat.codes.to_numpy().astype(np.float32)

Xc = a.layers["counts"] if "counts" in a.layers else a.X
Xc = np.asarray(Xc.todense() if hasattr(Xc, "todense") else Xc, dtype=np.float32)
X = np.log1p(Xc / np.maximum(Xc.sum(1, keepdims=True), 1) * 1e4)                 # RNA: log1p lib-normalised
Yprot = a.obsm["protein_counts"]; prot = Yprot.columns.astype(str).to_numpy()
Y = np.log1p(Yprot.to_numpy(np.float32))                                         # ADT: log1p raw counts
nG, nP = X.shape[1], Y.shape[1]

for SEED in FITS:
    out_file = WORK / f"pred_scvaeit_fit{SEED}.npz"
    if out_file.exists(): print(out_file.name, "exists, skipping", flush=True); continue
    A = [list(prot).index(p) for p in json.load(open(WORK / f"sentinels_fit{SEED}.json"))["ridge"]]
    train_p, held_p = train_test_split(p_site.index.to_numpy(), test_size=0.5, random_state=SEED, stratify=p_site.to_numpy())
    train = np.isin(pat, train_p); held = np.where(~train)[0]; held = held[np.argsort(pat[held], kind="stable")]
    print(f"fit {SEED}: train cells {train.sum()} | held {len(held)} | sentinels {[prot[j] for j in A]}", flush=True)

    masks = np.zeros((len(X), nG + nP), dtype=np.float32)                          # 0 = observed, -1 = missing
    hidden = np.setdiff1d(np.arange(nP), A)
    masks[np.ix_(~train, nG + hidden)] = -1.                                       # held cells: ADT hidden except sentinels
    data = np.c_[X, Y].astype(np.float32); data[masks == -1] = 0.

    config = {"dim_input_arr": np.array([nG, nP]), "dim_block": np.array([nG, nP]),
              "dist_block": ["NB", "NB"], "dim_block_enc": np.array([256, 128]),
              "dim_block_dec": np.array([256, 128]), "dim_block_embed": np.array([32, 16]),
              "dimensions": [256], "dim_latent": 32, "beta_unobs": 2. / 3.,
              "beta_modal": np.array([0.15, 0.85]), "p_feat": 0.2}
    tf.keras.utils.set_random_seed(SEED)
    model = VAEIT(config, data, masks, None, site_code[:, None], None)
    t0 = time.time()
    model.train(valid=False, num_epoch=EPOCHS, batch_size=BATCH, verbose=False,
                save_every_epoch=max(EPOCHS, 1), checkpoint_dir=str(WORK / f"scvaeit_ckpt_fit{SEED}"))
    print(f"  trained {EPOCHS} epochs in {(time.time() - t0) / 60:.1f} min", flush=True)

    den = model.get_denoised_data(zero_out=False, return_mean=True, num_repeat=1, L=25, batch_size_inference=512)
    pred = np.asarray(den)[:, nG:]                                                 # ADT block, log1p scale
    q = np.quantile(Y[train] - pred[train], [0.05, 0.95], axis=0)                  # train-cell residual band
    np.savez_compressed(out_file, y=Y[held], mean=pred[held].astype(np.float32),
                        lo=(pred[held] + q[0]).astype(np.float32), hi=(pred[held] + q[1]).astype(np.float32),
                        sd=np.zeros_like(pred[held], dtype=np.float32), patient=pat[held],
                        site=a.obs["Site"].astype(str).to_numpy()[held],
                        celltype=a.obs["initial_clustering"].astype(str).to_numpy()[held], protein=prot)
    O = hidden
    rmse = float(np.sqrt(((pred[held][:, O] - Y[held][:, O]) ** 2).mean(0)).mean())
    print(f"  held-out RMSE on non-sentinel proteins: {rmse:.4f} | saved {out_file.name}", flush=True)
    del model
print("ALL DONE", flush=True)
