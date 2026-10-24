# COMBAT pool-held-out calibration — completed 24 October 2026

Converts the paper's "a pool-held-out sensitivity analysis is future work" into a finished result.
Predictor: measured-input ridge (pred_ridgeanchor), both fits, 182 evaluation proteins, participant-block
calibration at gamma=0.90, delta=0.20, bound 13/16 = 0.8125. No retraining: only the calibration/test
allocation changed. 30 calibration draws per pool.

## Design
For each processing pool g, calibrate on 15 participants drawn ONLY from the other pools, then evaluate
on the held-out participants belonging to g. Comparison arm swaps one calibration participant for a
same-pool one, holding calibration size at 15. 13 of 122 participants span more than one pool and were
assigned their modal pool.

## Result
Pooled over 10 pools and both fits:
  pool entirely absent from calibration      attainment 0.810
  one same-pool calibration participant      attainment 0.810
  guaranteed level                                      0.8125

The guarantee holds when a whole processing pool is unrepresented in calibration, and having a
same-pool participant present makes no measurable difference. 10 of 20 pool-by-fit cells sit below
0.8125, which is what a tight marginal bound looks like from one realisation; per-pool n_test is only
2-7, so individual pools are noise and only the pooled figure should be quoted.

## Why this is not in tension with the Sanger finding
The pools genuinely differ on the shared factor. Reproduced here from the ridge residuals:
  PC1 share of participant-offset variance   0.421 (fit 2026), 0.435 (fit 2027)
  loadings sharing a sign                    1.00, 0.99
  Kruskal-Wallis of PC1 across pools         p = 1.0e-06, 1.0e-05
(The paper reports 0.384/0.399 and p = 9e-7/9e-6 from the totalVI residuals; the variance share differs
because the predictor differs, the pool separation is the same.)

So pool identity shifts the factor strongly yet does not break exchangeability of the participant
SCORES. The reconciliation is COMBAT's own small between-participant shifts relative to cell noise --
the same property that made post-hoc correction unnecessary there. Stephenson's Sanger site, by
contrast, carries roughly twice the ADT depth of the other sites, and pooled calibration left it at
0.75 against 0.817 elsewhere.

## The statement this supports
Local calibration participants are needed when a stratum's technical shift is large relative to
cell-level noise (Stephenson sites), and not when it is small (COMBAT pools), even where the stratum
is a highly significant source of variation in the shared factor. That is a sharper claim than either
cohort gives alone, and it replaces an n=4 descriptive caveat with a 10-stratum test.

Design rule still applies for the group-CONDITIONAL variant: a finite Mondrian lambda needs n_g >= 4
calibration participants per group at delta = 0.2.
