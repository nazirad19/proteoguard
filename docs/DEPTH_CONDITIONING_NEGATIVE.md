# Can depth-conditional calibration fix cross-site transfer? No.
Stephenson, measured-input ridge, site withheld from calibration, 30 draws. Bound 0.8125.

                              Cambridge (n=15)   Sanger (n=4)   Newcastle (n=20)
  plain participant-block           0.699            0.519            0.922
  multiplicative depth scale        0.672            0.546            0.934
  per-protein depth offset          0.552            0.662            0.950
  shared-slope depth offset         0.631            0.576            0.946

None restores transfer. Two of three make Cambridge worse.

## Why — the informative part
Participant score tracks antibody depth almost deterministically WITHIN a site
(Spearman -0.963 in Newcastle, -0.779 Cambridge, -0.577 Sanger) and the slopes agree across sites
(-0.137, -0.100, -0.104 within; -0.089 between), so this is not a Simpson's-paradox artefact and the
covariate is genuinely informative about the score's LOCATION.

But calibration does not depend on the location of the score; it depends on its upper quantile, which
is what lambda is. Those are not the same functional, and depth does not predict the second.
Cambridge has the highest depth (4.67 vs 3.84 Newcastle, 3.66 Sanger), so every negative-slope model
concludes "high depth, therefore easy, therefore narrower interval" and halves its widening
(median lambda 0.084 -> 0.041) at exactly the site that needed widening. The adjustment is
confidently wrong in the direction that hurts.

Scores also cross zero (site median participant score -0.024 to +0.100), which makes any
multiplicative construction ill-posed on this quantity regardless of power.

## What this is worth
A paper-ready explanation for why the transfer failure is not fixable by conditioning on the obvious
covariate: the shared technical factor shifts the score distribution's centre, not its tail. It
belongs in the Discussion next to the transfer result. It is a negative result, not a new method.

## What would actually be needed
A construction that targets the upper quantile of the participant score as a function of covariates
(quantile regression on participant scores, or weighted/localized conformal with weights estimated in
score-tail space rather than score-location space), with enough design participants to fit it -
n_design = 20 is thin for anything per-protein. That is a research project, not a ten-day change.
