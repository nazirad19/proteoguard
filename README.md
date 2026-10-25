# ProteoGuard

Participant-level conformal calibration of RNA-to-protein imputation with a minimal measured panel.

Code for the RECOMB 2027 submission. Cell-wise conformal calibration of imputed surface proteins
fails for new participants; one shared technical factor per participant explains most of the error;
calibrating with participants as the exchangeable unit restores a per-participant coverage guarantee,
verified in two multi-site COVID-19 CITE-seq cohorts (Stephenson et al. 2021; COMBAT 2022). The same
order-statistic argument extends to a panel-tolerance guarantee, a site-conditional variant, and
conformal-risk-certified selective biomarker calls.

## Quickstart (CPU, ~2 minutes)

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11 or 3.12
pip install -r requirements-demo.txt
python examples/run_toy.py                          # synthetic data through the real evaluator
PYTHONPATH=scripts/analysis python -m unittest discover -s tests -v   # 23 tests
```

The toy run exercises the actual anchor/correction/calibration code on bundled synthetic counts and
prints `PASS`; it does not reproduce the cohort tables. The tests assert the statistical guarantees
themselves (rank rule, k > n behaviour, the n_g ≥ 4 Mondrian minimum, the 1/(n+1) certification
floor, loss monotonicity), so a change that breaks validity fails CI.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | full pipeline: preparation, training, calibration, decisions, figures |
| `scripts/analysis/proteoguard_upgrades.py` | panel tolerance, Mondrian, conformal risk control |
| `tests/` | CPU tests, including the guarantee suite |
| `examples/` | synthetic toy demo with expected outputs |
| `results/audit/` | regenerated values behind the paper's verification round |
| `provenance/` | participant/protein manifests (from `analysis/provenance_audit.py`) |
| `docs/` | data sources and schemas, reproduction record, stratum transfer, novelty check |

## Data

All data are public; nothing participant-level is committed (objects referenced by SHA-256 in
`docs/REPRODUCTION_2026-10-24.md`).

| Cohort | Source | File |
| --- | --- | --- |
| Stephenson et al. 2021 | COVID-19 Cell Atlas / E-MTAB-10026 | `haniffa21.processed.h5ad` |
| COMBAT 2022 | [Zenodo 6120249](https://zenodo.org/records/6120249) | `COMBAT-CITESeq-DATA.h5ad` |
| Hao et al. 2021 (single-centre) | GEO GSE164378 | prepared CITE-seq object |
| totalVI benchmark | scvi-tools datasets | loaded in script |

Prepare cohorts with `scripts/stephenson/stephenson_prep.py` / `scripts/combat/combat_prep.py`
(schemas in `docs/DATA_PREPARATION.md`).

## Full pipeline

Full training needs the `requirements.txt` environment (Python 3.11, `scvi-tools==1.4.2`, GPU;
scVAEIT optionally adds TensorFlow 2.15 in a separate environment). The runner prints the plan and
only trains with `--execute`:

```bash
python scripts/run_pipeline.py --cohort stephenson --work /path/to/stephenson --seeds 2026 2027
python scripts/run_pipeline.py --cohort combat     --work /path/to/combat     --seeds 2026 2027
```

Order: base predictors → low-rank mechanism + frozen design split → anchor selection
(`baselines_calib.py`, first pass) → measured-input ridge (`ridge_sentinels.py`) → calibration grid
(`baselines_calib.py`, second pass) → metrics, budgets, biomarker calls → figures. `STEPH_DIR`
selects the cohort directory; `STEPH_FITS` the splits.

## Figure / table map (paper numbering)

| Paper item | Script |
| --- | --- |
| Fig 1 (simulation) | `simulation/sim_donor_blocks.py` |
| Figs 2–6, S1 | `figures/paper_figures.py` |
| Fig 7 (sentinel budget) | `analysis/anchor_budget.py` → `figures/fig_budget.py` |
| Figs 8–9 (calibration rules, COMBAT mechanism) | `figures/paper_figures2.py` |
| Fig 10 (panel tolerance), Table 5 (certified calls) | `analysis/upgrades_real.py` + `proteoguard_upgrades.py` |
| Fig 11 (stratum transfer) | `analysis/pool_heldout.py`; results in `results/audit/` |
| Tables 1–2 | `analysis/proteoguard_metrics.py` |
| Tables 3–4 | `analysis/baselines_calib.py` (second pass) |
| Table S1 | `analysis/sensitivity_balanced.py` |

## Verification status

The ridge and measured-input-ridge arms were independently re-executed end-to-end from the raw
public deposits on a separate machine (24 Oct 2026; COMBAT exact, Stephenson 115,683 vs 115,708
cells, RMSEs within 0.01) — see `docs/REPRODUCTION_2026-10-24.md`. The GPU arms (totalVI, scVAEIT)
have not been independently rerun; seeds, frozen split manifests and provenance tables are provided
so each stage can be checked against the reported numbers. All evaluations are transductive
(held-out participants' RNA is present during fitting with target proteins masked); an inductive
ridge rerun left the results unchanged (paper, Results).

## Authors and license

Anel Umirbaeva and Nazira Dunbayeva (equal contribution). MIT License.
