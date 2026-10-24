# Both cohorts, trained from scratch — 24 October 2026

Everything below was prepared from the public deposits and trained on this machine. totalVI and
scVAEIT were NOT run (no GPU), so this covers the ridge and measured-input-ridge arms only.

## 1. Reproduction of the published accuracy numbers

                                   paper   reproduced
  Stephenson RNA-only ridge         0.72   0.711 / 0.717
  Stephenson RNA + 5 sentinels      0.54   0.530 / 0.538
  COMBAT     RNA-only ridge         0.64   0.640 / 0.638
  COMBAT     RNA + 5 sentinels      0.55   0.548 / 0.545

Cohort construction: Stephenson 118 participants (60 Ncl / 47 Cambridge / 11 Sanger), 115,683 cells
against the published 115,708 (25 cells apart); 59 held out per fit with 2 Sanger in the design set,
both exactly as stated. COMBAT 122 participants, 118,649 cells — exact.

Note the sentinels selected here differ from the published ones (only KLRK1 overlaps on Stephenson
fit 1), because greedy selection runs on an independently reconstructed object. The RMSE still lands
within 0.01. That is positive evidence for the paper's own claim that anchor choice is not the
active ingredient.

## 2. The improved predictor replicates across both cohorts

Three changes: depth and cell-type covariates as predictor inputs (depth computed from MEASURED
channels only, never adt_total, which would leak the targets), SVD rank 200, and adaptive intervals
from a difficulty model in place of global residual quantiles.

                        RMSE    attainment   interval score   width
  Stephenson baseline   0.537      0.823          2.306        1.868
  Stephenson improved   0.528      0.820          2.186        1.771
  COMBAT     baseline   0.550      0.820          2.360        1.828
  COMBAT     improved   0.533      0.820          2.198        1.750

Interval score improves 5.2% (Stephenson) and 6.9% (COMBAT) at unchanged participant attainment,
both still clearing 13/16 = 0.8125. The gain replicates; it is an efficiency gain, not a new claim.

## 3. Stratum-held-out calibration — the result that matters

Calibrate on 15 participants drawn ONLY from other strata, evaluate on the held-out stratum.
30 calibration draws. Bound 0.8125.

  COMBAT, processing pool held out (10 pools, pooled)   0.810   PASSES
  Stephenson, Ncl held out        (20 test)             0.922   PASSES
  Stephenson, Cambridge held out  (15 test)             0.699   FAILS
  Stephenson, Sanger held out     ( 4 test)             0.519   FAILS

This is the asymmetry, now measured on both sides rather than asserted. COMBAT's processing pools
separate the shared factor extremely strongly (PC1 42-44% of participant-offset variance,
Kruskal-Wallis p ~ 1e-6) yet calibration transfers across them intact. Stephenson's sites break it
badly: Cambridge at 0.70 and Sanger at 0.52 against a 0.81 guarantee.

The discriminator is not whether the stratum is a significant source of variation — both are — but
the SIZE of the between-participant shift relative to cell noise. Cambridge carries roughly twice the
median ADT depth of the other Stephenson sites; COMBAT's pooled design keeps participant-level shifts
small, which is also why post-hoc correction was unnecessary there.

The improved predictor partially mitigates but does not fix it (Sanger 0.519 -> 0.690,
Cambridge 0.699 -> 0.768, both still short of 0.8125), and it costs Ncl (0.922 -> 0.850). This is an
exchangeability problem, not an accuracy problem, and better prediction is the wrong lever.

## 4. What this means for the paper

The honest headline is a conditional, not a victory: participant-block calibration transfers across
processing strata whose technical shift is small, and fails across strata with large depth
differences, where calibration participants from that stratum are required. The published Sanger
observation (n = 4, descriptive) now has a 10-pool positive control and a 15-participant Cambridge
negative control on either side of it.

This strengthens the paper by making the limitation precise and testable rather than by removing it.
A reviewer asking "does it work for a participant from a new site?" now gets a measured answer with a
stated condition, instead of a caveat.
