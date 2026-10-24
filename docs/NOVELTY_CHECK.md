# Novelty hand-check — Generalized HCP and UNIPRO
Checked 24 October 2026 against ProteoGuard's contribution list. Verdict: the central claim survives.

## 1. Generalized Hierarchical Conformal Prediction (GHCP)
arXiv, 16 Aug 2026. Closest methodological neighbour. **Does not pre-empt contribution (1), but is the
most consequential citation of the two.**

What it does: HCP (Lee et al.) gives prediction sets for a new observation from a previously unseen
group under hierarchical exchangeability. GHCP handles the case where a few observations from the
test group have ALREADY been collected — which breaks HCP's required symmetry — by assigning the test
group a randomly "donated" reference group size, and uses those initial observations to sharpen the
nonconformity scores. A restricted-donor variant improves efficiency.

Why it does not pre-empt us: GHCP's target is still marginal coverage of a NEW OBSERVATION from a
group. Ours is the probability that a new participant's OWN within-participant coverage fraction
reaches gamma. Different functional, same as the distinction we already draw against HCP.

Why it matters anyway, and what to change: GHCP addresses precisely the symmetry problem our
transductive Remark wrestles with. In our setting the test participant's RNA (and, for scVAEIT, their
sentinels) is present at fitting time, so we ARE in the regime GHCP was written for. A reviewer who
knows this paper will ask why we did not use it. Actions:
  - Cite GHCP in Related Work beside HCP, with the target-functional distinction stated explicitly.
  - Add one sentence to the transductive Remark acknowledging that GHCP offers a principled route to
    restoring symmetry when test-group observations are already in hand, and that our own route is
    the permutation-invariance argument plus the inductive re-run.
  - Optional, only if time permits: GHCP as a fourth calibration comparator. NOT required - it
    answers a different question, and we already have HCP as the marginal-coverage reference point.

## 2. UNIPRO (protein inference and panel expansion)
Research Square preprint, 7 Aug 2026, DOI 10.21203/rs.3.rs-10287417/v1.
**Does not pre-empt any contribution. Must be cited as a 2026 baseline.**

What it does: a graph-based framework learning protein-aware representations from transcriptomic
similarity, tissue architecture and local microenvironment, applied to single-cell protein inference,
spatial protein reconstruction, single-cell-to-spatial transfer, and protein-panel expansion.
Evaluated across 16 spatial and single-cell CITE-seq datasets, reported to outperform prior methods
on inference and imputation.

Why it does not pre-empt us: it is a point-prediction method. No uncertainty quantification, no
coverage statement, no participant-level calibration, no selective decision layer. "Panel expansion"
there means transferring a panel across datasets, not using a small measured panel to calibrate
participant-level reliability. Our layer wraps any base predictor, UNIPRO included.

Action: cite in Related Work with cTP-net / sciPENN / scVAEIT / CAPTAIN, and note in one clause that
stronger point predictors do not by themselves produce participant-level reliability - which is the
same argument we already make with scVAEIT, where the better imputer did not win.

## 3. MoTRUST (adjacent, found incidentally)
Preprint, 24 Sep 2026. Mosaic single-cell integration and molecular recovery via neural
representations plus conditional residual diffusion; reports first place on 30 integration tasks.
Again point prediction and integration, no coverage guarantee. Cite alongside scVAEIT as recent
mosaic-integration work; no claim of ours is affected.

## 4. Also surfaced, not relevant
"Hierarchical Conformal Classification" (Aug 2025) - classification sets, not per-group coverage.
"Selection from Hierarchical Data with Conformal e-values" (Jan 2025) - selective inference on
hierarchical data, a different problem.

## Net effect on the contribution list
  (1) participant-level tolerance calibration of RNA-to-protein imputation - INTACT
  (2) shared technical factor mechanism                                    - INTACT
  (3) controlled evaluation of where a small measured panel helps          - INTACT
  (4) selective biomarker-call layer with the certifiable-risk floor       - INTACT
  (5) panel-tolerance rule                                                 - INTACT

Three citations to add and one sentence to amend in the transductive Remark. No experiment is
invalidated and no claim needs narrowing.

## Search record
arXiv: hierarchical-conformal title/abstract 2025-01-01 onward (3 hits); UNIPRO title/abstract
2024-01-01 onward (3 hits, none the right paper - it is not on arXiv).
OpenAlex 2024 onward: "conformal prediction surface protein imputation CITE-seq uncertainty" (14);
"participant-level donor-level calibration single-cell protein imputation coverage" (25);
"UNIPRO protein imputation uncertainty" (4, first hit is the paper).
No result in any query bounds a per-participant coverage fraction for imputed proteins.
