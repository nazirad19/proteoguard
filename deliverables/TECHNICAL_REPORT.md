# ProteoGuard: reliable immune-marker imputation for new participants

Course final project · revised 8 October 2026

## Summary and biological question

**Can RNA plus a small measured protein panel support reliable participant-level calls of immune activation markers in new participants?** ProteoGuard addresses this question with protein imputation, participant-block conformal calibration, and selective biomarker calls. Its therapeutic-discovery connection is the reliability of immune-phenotyping and candidate biomarker measurements. The present experiments do not validate a treatment-selection rule or a clinical diagnostic threshold.

ProteoGuard extends the **analysis pipeline around totalVI**, rather than changing its architecture or training objective. The same calibration and correction procedures also apply to ridge regression. Across the two COVID-era PBMC cohorts studied, participant-block calibration improved the frequency with which held-out participants reached the specified within-participant coverage target. Giving measured proteins directly to ridge improved interval efficiency in both cohorts. Post-hoc correction helped in Stephenson but added little, or worsened some outcomes, in COMBAT. Biomarker decisions remained selective: COMBAT supported considerably fewer calls than Stephenson.

This report summarizes the supplied analyses and includes a static inspection of the subsequently supplied code archive. The subsequent repository revision fixed execution/configuration issues and added CPU tests, but did not rerun training or independently verify numerical results against prediction files. Preliminary scVAEIT results are excluded from the main conclusions.

## Background and contribution

CITE-seq measures RNA and antibody-derived tags (ADTs) in the same cells. totalVI jointly models RNA and protein counts and can generate posterior-predictive protein distributions. For a cell whose protein panel is hidden, the analysis predicts proteins using the available RNA and model context. The response variable here is **log(1 + observed ADT count)**, which includes technical variation as well as biological signal.

Prior work already examined uncertainty under RNA-to-protein dataset transfer, including sciPENN. Hierarchical conformal methods already address grouped observations, and dsb studies shared technical variation in measured CITE-seq proteins. ProteoGuard therefore does not claim to invent uncertainty evaluation, grouped conformal prediction, or technical-factor correction. Its contribution is an evaluated framework connecting participant-level coverage, small measured panels, correction controls, and immune-marker decisions.

Cells from one participant can share errors. Pooled cell-wise calibration may attain roughly 90% coverage overall while particular participant–protein pairs remain undercovered. This is a statement about **participant–protein pairs**, not the fraction of participants who fail on their whole panel. The experiments test whether calibrating whole participant blocks improves the chosen reliability objective.

## Study design

| Cohort | Analysed data reported by the source | Model training | Held-out participant allocation |
|---|---|---|---|
| Stephenson et al. | 118 participants; 115,708 cells; 192 ADTs; 3 sites | 59 participants with RNA and protein | 20 design; 35 in the calibration/test pool, repeatedly split 15/20; 4 Sanger evaluation participants without local calibration participants |
| COMBAT | 122 participants; 118,649 cells; 192 ADTs; 2 institutes; 10 processing pools | 61 participants with RNA and protein | 20 design; 41 in the calibration/test pool, repeatedly split 15/26 |

The supplied analysis excludes LPS-stimulated Stephenson samples and low-density-neutrophil preparations in COMBAT, and caps each participant at 1,000 cells. Two separately drawn train/held-out splits were analysed per cohort. Because these splits reuse cohort participants, they are repeated evaluations within a cohort, not independent cohort replications.

The roles are distinct:

1. **Training participants:** fit the base predictor using RNA and measured proteins.
2. **Design participants:** select anchors, fit any correction, and define biomarker thresholds using their measured proteins. These participants are excluded from base-model training.
3. **Calibration participants:** use measured target proteins to compute conformal scores, after freezing the design choices.
4. **Test participants:** expose RNA and only the permitted anchor channels to the prediction pipeline. Hidden target proteins are reserved for evaluation.

**Transductive input use:** the supplied totalVI trainers include held-out participants' RNA during model fitting while masking their proteins. The ridge-with-sentinels script also fits its RNA SVD on all available RNA before fitting ridge on training participants. Thus the experiments hold out target protein labels; they are not a strictly inductive test of a frozen model on participants whose RNA was never available during fitting. This input-use distinction also matters when applying the exchangeability argument to deployment.

This deployment requires a fully measured reference/calibration cohort, even though a test participant supplies only a small panel. Sanger was represented in the broader training/design scheme: the four-person subgroup tests the absence of local calibration, not a completely unseen site. Random participant splits in COMBAT evaluate represented processing environments; new-pool transfer is a separate question.

## Methods

### Predictors and measured panels

The base predictors are totalVI and ridge regression from RNA components to log1p ADT. totalVI intervals use the efficiency-corrected posterior-predictive protein likelihood (`py_norm_` in the reported scvi-tools 1.4.2 implementation), decoded in the training batch. This readout distinction is essential for count-scale evaluation. The manuscript describes 100 predictive draws and the 5th and 95th percentiles. It also describes in-sample residual quantiles for the initial ridge intervals; this limitation should be retained until an out-of-fold implementation is verified.

For participant i and protein j, the pipeline estimates the median cell residual relative to the predictor's centre. Five anchor channels supply observed residuals in a new participant. Design participants determine the selected panel and any learned map. Named decision markers are excluded from anchor selection. The correction comparisons are:

- No participant-specific correction; the budget analysis uses a design-mean reference, which should not be confused with raw uncalibrated totalVI.
- A common offset: add the mean anchor residual to every target protein.
- A shared map: fit a protein-specific intercept and slope from the mean anchor residual.
- A multi-anchor ridge map: retain separate weights for each anchor.
- An isotype-driven map, where appropriate measured isotype controls are available.
- **Ridge with measured inputs:** append per-cell anchor values and participant anchor summaries to RNA features, without post-hoc correction.

The PCA mechanism analysis uses cell-type-balanced participant residuals. The supplied slides identify design participants as the PCA set and all 59 held-out Stephenson participants as the association-analysis set. These sets must remain separately labelled in the provenance tables.

### Calibration and its exact scope

For an initial interval [l, u], correction s, and observed log1p count y, define the cell score as max(l + s − y, y − u − s). For each calibration participant, take the ceil(0.90 × number of cells)-th smallest score. Across n calibration participants, take the k-th score, where k = ceil((n + 1)(1 − δ)) and δ = 0.20. Apply that adjustment to the new participant's intervals.

With n = 15, k = 13. Under exchangeability of complete participant blocks, with the predictor and design choices fixed appropriately, the standard rank argument gives, **for each protein separately**:

> P(empirical coverage across the new participant's evaluated cells ≥ 90%) ≥ 13/16 = 81.25%.

This is marginal over calibration and future-participant sampling. It is not a promise for every participant, simultaneous coverage across all proteins, or coverage of an unlimited population of future cells from that participant. If k > n, the mathematical definition uses an infinite quantile. Scores can be negative, allowing interval contraction. Dependence among cells is allowed within a participant block, while exchangeability of complete scores remains an assumption. Because the supplied predictors use held-out RNA transductively, their fitting procedure must also respect the relevant calibration/test symmetry; the standard independent, fixed-predictor argument is not by itself a deployment proof for an additional participant arriving later.

The archive implements the inner quantile with `np.quantile(..., method="higher")`. This can be one order statistic above the stated minimum ceil(0.90 × m), making the inner rule conservative. Its outer participant-block implementation does not guard k > n, although k = 13 is valid in the reported n = 15 setting. These implementation details should be reconciled before generalizing to other calibration sizes.

Comparators are pooled cell-wise conformal and hierarchical conformal prediction (HCP). HCP targets marginal coverage for an observation from a new group, a different objective. Higher coverage and wider intervals are a trade-off, not failure of HCP to meet ProteoGuard's lower bound.

### Biomarker decisions

For each eligible protein and cell type, a two-component mixture fitted on design measurements defines cell positivity. A participant's readout is the fraction of positive cells. The design-cohort median defines the participant high/low threshold.

Calibrate an interval around the predicted positive fraction using calibration participants. Call **high** if the entire interval lies above the threshold; call **low** if it lies below; otherwise abstain. Marginal interval coverage does not guarantee the error rate among the selected calls. Report selective error empirically, at matched call rates, with participant-level uncertainty and denominators. These thresholds are cohort-relative, not clinical.

### Evaluation

Primary summaries are the fraction of participant–protein pairs below 85% empirical coverage, the share of participants reaching at least 90% coverage per protein, interval score, and median interval width. The 85% threshold is a diagnostic; it is not the 90% target in the proposition. Common target sets contain 177–179 proteins depending on the fit after excluding measured anchors and isotypes. Cross-method comparisons must use the same target list within each evaluation.

Reported summaries average 50–200 repeated calibration/test allocations. The supplied analysis uses participant bootstrap intervals after averaging a participant's repeated test evaluations. Repeated allocations do not increase the number of independent participants. Exact target lists and participant IDs are needed to reproduce each table.

## Results

### A dominant shared residual component

In Stephenson, PC1 explained 75.6% of participant-offset variance versus a 9.8% permutation reference, with 98.4% of loadings sharing a sign. Its association with ADT depth was ρ = 0.83. No adjusted severity association was detected. These findings are consistent with a shared technical contribution, but do not establish a purely technical cause: depth and residual summaries derive from the same measurements, and absence of a detected association is not evidence of no biological effect.

One selected anchor captured much of the participant-offset signal in the budget experiment (reported R² 0.85–0.86). The shared map plateaued between approximately one and ten anchors. Deterioration at twenty is consistent with dilution by less informative anchors, but that mechanism was not isolated. High offset R² measures recovery of shared participant errors, not recovery of all protein-specific biology absent from RNA.

### Calibration and correction controls in Stephenson

| Predictor/correction | Calibration | Pairs below 85% | Participants reaching 90%, averaged across proteins | Interval score | Median width |
|---|---|---:|---:|---:|---:|
| Ridge + shared map | Cell-wise | 13.5% | 0.61 | 2.71 | 1.94 |
| Ridge + shared map | HCP | 0.9% | 0.96 | 2.94 | 2.54 |
| Ridge + shared map | Participant-block | 4.2% | 0.83 | 2.77 | 2.19 |
| Ridge + measured inputs | Cell-wise | 14.3% | 0.58 | 2.28 | 1.71 |
| Ridge + measured inputs | Participant-block | 4.5% | 0.82 | 2.33 | 1.90 |
| totalVI + shared map | Participant-block | 5.2% | 0.83 | 2.78 | 2.22 |

Participant-block summaries were consistent with the specified participant-level target. HCP attained higher empirical coverage and wider intervals. These methods have different guarantees, so interval width alone does not establish superiority.

With ridge and participant-block calibration, a common offset, isotype panel, and learned shared map gave interval scores of 2.79, 2.81, and 2.77, respectively. Their undercoverage rates were 4.4%, 4.5%, and 4.2%. Thus the learned map offered little additional benefit in these comparisons. Isotypes may provide usable measurements without adding new biological markers when suitable controls already exist in the assay; this is not a universal zero-cost deployment claim.

Giving measured proteins directly to ridge produced the lowest interval score among these completed comparisons: 2.33 versus 2.77 for the post-hoc shared map, with similar coverage. Reported RMSE decreased from 0.72 for the RNA-only predictor to 0.54 with measured inputs. The interval-score and RMSE contrasts have different reference pipelines and should be labelled separately.

### Biomarker decisions

In Stephenson, the supplied course report records 1.6–2.3% observed selective error at 27–33% call rates across the reported calibrated pipelines. Naive point thresholding made errors on roughly 31–33% of all decisions. That is not a matched-call-rate comparison.

The supplied full draft also reports the following matched comparison at a **27% call rate**, with ranges across the two base predictors:

| Selection/correction procedure | Observed error among calls |
|---|---:|
| Random abstention | 31.7–32.9% |
| Distance from the threshold | 20.1–20.6% |
| Anchor correction without calibration | 4.7–6.6% |
| ProteoGuard | 2.1–2.2% |

These descriptive ranges are not confidence intervals. For named markers, per-marker call counts and participant-level confidence intervals remain necessary before emphasizing very low or zero observed errors.

### Replication and limits in COMBAT

PC1 explained 38–40% of participant-offset variance; 99% of loadings shared a sign and the ADT-depth association was ρ = 0.63–0.69. Association with processing pool supports a technical contribution. The nonsignificant clinical-group association does not rule out biological contributions.

With measured inputs, ridge RMSE decreased from 0.64 to 0.55, interval score from 2.81 to 2.41, and median width from 2.15 to 1.83, approximately 15% narrower. The reported participant success fraction was 0.82 for this pipeline. Across calibration arms, cell-wise success was approximately 0.55 for corrected or measured-input pipelines, while HCP was approximately 0.96–0.99. Uncorrected totalVI was an exception to a blanket 0.82 summary: participant-block success was 0.90 in the supplied full table.

Post-hoc correction did not add a consistent benefit. For totalVI, undercoverage increased from 2.1% without correction to approximately 4.0–5.4% with the reported corrections. This is consistent with correction being less useful when shared shifts are small; the experiment does not by itself identify a causal bias–variance decomposition.

Biomarker calls were considerably harder: calibrated methods made only 5–9% of decisions, with 5–9% observed selective error. The framework abstained frequently; broad recovery of participant biomarker calls therefore did not replicate.

## Limitations, unfinished work, and reproducibility

The completed evidence is from two COVID-era PBMC cohorts. Benefits of measured inputs and participant-block calibration replicated in these cohorts; their universality is not established. Post-hoc correction was cohort-dependent. Four Sanger participants are insufficient to establish a general requirement for local calibration at every site. New-pool COMBAT evaluation remains a separate sensitivity analysis.

The source report describes scVAEIT at 100 epochs as preliminary and a longer run as ongoing. The supplied `scvaeit_run.py` instead defaults to 200 epochs through `SV_EPOCHS`, and skips a fit if its prediction file already exists. Changing the epoch variable does not establish that an existing result was retrained or resumed. No live server log or completed comparison was supplied. scVAEIT must be evaluated on the same observable inputs and target scale, with training and convergence documented, before making a model-superiority claim.

**Code provenance:** the supplied ZIPs contained identical non-Git files: 33 Python scripts, README, requirements, an MIT license template and gitignore. Archived source commit: `4cd21a01fed12e6176061fdc78ebc2f49f6892d9`. The revised repository is [nazirad19/proteoguard](https://github.com/nazirad19/proteoguard), kept private at this stage.

**Repository corrections (8 October 2026):** Stephenson training now accepts `STEPH_FIT`; COMBAT fit lists and previously fixed paths accept environment overrides. An ordered runner performs baseline calibration before measured-input ridge, generating the required sentinel JSON, and repeats calibration afterwards. Figure 7 now loads `anchor_budget.csv`, checks its schema/completeness, aggregates each fit separately and writes a summary CSV. The former hard-coded illustration is retained as `fig_budget_historical.py`, labelled historical. A raw-count validator and explicit preparation adapter have been added; these are not claimed to be the missing original preparation scripts.

Eight CPU unit tests passed locally, covering budget aggregation, missing/duplicate results, undefined no-call error, pipeline dependencies and count validation. All scripts pass syntax parsing. A GitHub Actions workflow repeats lightweight checks; these do not reproduce the cohort analyses. The repository targets Python 3.11 and scvi-tools 1.4.2; the historical environment lock is still needed.

| Component | Script under `scripts/` | Current status |
| --- | --- | --- |
| Preparation | `combat/combat_prep.py`, `prepare_data.py` | Original COMBAT script; reconstructed adapter for other layouts |
| totalVI predictions | `stephenson/stephenson_donor_calibration.py`, `combat/combat_totalvi.py` | Configurable fit seeds; efficiency-corrected predictive sampling |
| Controls and intervals | `analysis/baselines_calib.py`, `analysis/proteoguard_metrics.py` | Present; not rerun on real predictions locally |
| Biomarker decisions | `analysis/marker_calls*.py`, `analysis/risk_coverage*.py` | Present; numerical outputs not supplied |
| Provenance exports | `analysis/provenance_audit.py` | Script present; server-generated manifests still needed |
| Anchor-budget figure | `figures/fig_budget.py` | CSV-driven replacement; historical illustration retained separately |

Remaining work is explicit: recover or validate exact Stephenson/Hao preparation; export source participant/target manifests, environment lock, hardware metadata and result CSVs; rerun the ordered pipeline and figures on the original cohort files; and confirm the longer scVAEIT run. The full AnnData/HVG preparation path has not been tested on the cohort sources locally. Numerical research results in this report have not changed as a consequence of these code fixes. Fresh-clone end-to-end reproduction remains unverified.

## Conclusion

Participant-block calibration improved coverage reliability in both cohorts studied, and a small measured protein panel improved interval efficiency when used as predictor inputs. Post-hoc correction helped in Stephenson but added little in COMBAT. Biomarker calls remained selective, with substantially fewer usable calls in COMBAT. These findings support evaluating protein imputation at the participant level and measuring a small panel where feasible, while reporting abstention and the limits of transfer.

## References

- [Gayoso et al. 2021: totalVI](https://www.nature.com/articles/s41592-020-01050-x)
- [Stephenson et al. 2021: single-cell multi-omics in COVID-19](https://www.nature.com/articles/s41591-021-01329-2)
- [COMBAT Consortium 2022: a blood atlas of COVID-19](https://doi.org/10.1016/j.cell.2022.01.012)
- [Lee, Barber and Willett: distribution-free inference with hierarchical data](https://arxiv.org/abs/2306.06342)
- [Dunn, Wasserman and Ramdas: two-layer hierarchical prediction sets](https://arxiv.org/abs/1809.07441)
- [Lakkis et al. 2022: sciPENN](https://doi.org/10.1038/s42256-022-00545-w)
- [Du et al. 2022: scVAEIT](https://doi.org/10.1073/pnas.2214414119)
- [Mulè et al. 2022: dsb](https://www.nature.com/articles/s41467-022-29356-8)

Numerical source documents: *ProteoGuard final project report.md* (8 October 2026), *ProteoGuard — Final Project.pptx*, and the previously supplied *ProteoGuard Paper Draft (2).md* for explicitly identified full-table and matched-call-rate results. Code-inspection source: the two subsequently supplied ProteoGuard ZIP archives, commit `4cd21a01fed12e6176061fdc78ebc2f49f6892d9`.
