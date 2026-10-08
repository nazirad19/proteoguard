# %% Cell K1 - COMBAT: train totalVI and save held-out predictive intervals, for both data splits (~same time as Stephenson)
import os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, anndata as ad
import scvi
from scvi.model import TOTALVI
from scvi.distributions import NegativeBinomialMixture
from sklearn.model_selection import train_test_split

WORK = Path(os.environ.get("STEPH_DIR", os.environ.get("COMBAT_DIR", Path.home() / "projects" / "combat")))
os.environ["STEPH_DIR"] = str(WORK); os.environ["STEPH_CTS"] = "cMono,CD4,CD8,NK,B"   # later cells read these
TRAIN = dict(max_epochs=200, lr=4e-3, batch_size=512, early_stopping_patience=20)
N_DRAWS, ALPHA = 100, 0.10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

for FIT_SEED in [int(s) for s in os.environ.get("STEPH_FITS", "2026,2027").split(",")]:
    pred_file = WORK / f"pred_fit{FIT_SEED}.npz"
    if pred_file.exists(): print(pred_file.name, "exists, skipping"); continue
    a = ad.read_h5ad(WORK / "stephenson_sub1000.h5ad")
    a = a[a.obs["adt_total"].to_numpy() > 0].copy()
    truth = a.obsm["protein_counts"].copy(); pat = a.obs["patient_id"].astype(str).to_numpy()
    p_site = a.obs.groupby("patient_id", observed=True)["Site"].first().astype(str)
    train_p, held_p = train_test_split(p_site.index.to_numpy(), test_size=0.5, random_state=FIT_SEED, stratify=p_site.to_numpy())
    role = np.where(np.isin(pat, train_p), "TRAIN", "HELD")
    a.obs["role"] = pd.Categorical(role, categories=["TRAIN", "HELD"])
    counts = truth.copy(); counts.loc[role == "HELD", :] = 0.0; a.obsm["protein_counts"] = counts
    print(f"fit {FIT_SEED}: cells {a.n_obs} | proteins {truth.shape[1]} | train {len(train_p)} | held {len(held_p)}")
    print(pd.crosstab(a.obs["Site"], a.obs["role"]))

    TOTALVI.setup_anndata(a, batch_key="role", categorical_covariate_keys=["Site"], layer="counts",
                          protein_expression_obsm_key="protein_counts")
    model_dir = WORK / f"model_fit{FIT_SEED}"
    if (model_dir / "model.pt").exists():
        m = TOTALVI.load(str(model_dir), adata=a)
    else:
        scvi.settings.seed = FIT_SEED; t0 = time.time()
        m = TOTALVI(a, n_latent=20, n_hidden=256, n_layers_encoder=2, n_layers_decoder=1, gene_likelihood="nb",
                    latent_distribution="normal", empirical_protein_background_prior=False)
        m.train(max_epochs=TRAIN["max_epochs"], lr=TRAIN["lr"], batch_size=TRAIN["batch_size"],
                accelerator="gpu" if DEVICE == "cuda" else "cpu", devices=1, train_size=0.9, validation_size=0.1,
                early_stopping=True, early_stopping_patience=TRAIN["early_stopping_patience"],
                check_val_every_n_epoch=1, reduce_lr_on_plateau=True, adversarial_classifier=True, enable_progress_bar=False)
        m.save(str(model_dir), overwrite=True)
        print(f"  trained {m.history['elbo_train'].shape[0]} epochs in {(time.time() - t0) / 60:.1f} min")

    held = np.where(role == "HELD")[0]; held = held[np.argsort(pat[held], kind="stable")]
    t0 = time.time(); P = predictive(m, a, held, N_DRAWS)
    y = np.log1p(truth.to_numpy(np.float32)[held])
    np.savez_compressed(pred_file, y=y, patient=pat[held], site=a.obs["Site"].astype(str).to_numpy()[held],
                        celltype=a.obs["initial_clustering"].astype(str).to_numpy()[held],
                        protein=truth.columns.to_numpy(), **P)
    S = np.maximum(P["lo"] - y, y - P["hi"])
    print(f"  predictions for {len(held)} cells in {(time.time() - t0) / 60:.1f} min | raw cell coverage {(S <= 0).mean():.3f}")
    del m, a; torch.cuda.empty_cache()
