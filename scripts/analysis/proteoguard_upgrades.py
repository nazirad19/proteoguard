"""
proteoguard_upgrades.py — calibration upgrades for ProteoGuard.

Implements, on top of the existing participant-block rule:

  1. NORMALIZED (multiplicative) nonconformity scores      -> narrower intervals, same validity
  2. PANEL-LEVEL tolerance (simultaneous / fraction-tau)   -> a guarantee about the panel, not one protein
  3. MONDRIAN group-conditional calibration                -> site / processing-pool conditional
  4. CONFORMAL RISK CONTROL for the selective call layer   -> controls the decision risk

Score and interval conventions follow the existing pipeline:
    a_icj = max(l_icj + s_ij - y_icj,  y_icj - u_icj - s_ij)
    interval = [l + s - lam, u + s + lam]
so coverage of cell c  <=>  a_icj <= lam.  Scores may be negative (intervals shrink).

Requires numpy >= 1.22 (uses np.quantile(method=...)).
"""

import numpy as np

EPS = 1e-8
PRED_KEYS = ("y", "mean", "lo", "hi", "patient", "site", "celltype", "protein")


# --------------------------------------------------------------------------
# I/O against the documented NPZ schema
# --------------------------------------------------------------------------

def load_predictions(path):
    """Load pred_*.npz. Labels are object arrays, hence allow_pickle."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in PRED_KEYS if k in z}
    if "sd" in z:
        out["sd"] = z["sd"]
    missing = [k for k in ("y", "mean", "lo", "hi", "patient", "protein") if k not in out]
    if missing:
        raise KeyError(f"{path}: missing required keys {missing}")
    for k in ("patient", "site", "celltype", "protein"):
        if k in out:
            out[k] = np.asarray(out[k]).astype(str)
    return out


def check_contiguous(patient):
    """Analyses build participant slices, so cells must be contiguous by participant.

    Counts runs, not the distinct labels that start a run: a participant appearing in two
    separate blocks (A A B B A A) yields more runs than labels and must fail. Comparing
    unique(p[starts]) with unique(p) does NOT detect that case -- it passes everything.
    """
    p = np.asarray(patient)
    starts = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    return len(starts) == len(np.unique(p))


def first_block_value(values, patient, pat):
    """Per-participant label (site / pool) read from that participant's cells."""
    v = np.asarray(values)[np.asarray(patient) == pat]
    u = np.unique(v)
    if len(u) != 1:
        raise ValueError(f"participant {pat} has {len(u)} distinct values: {u[:5]}")
    return u[0]


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def half_width(lo, hi):
    """Base interval half-width, the difficulty proxy used for normalization."""
    return np.maximum((hi - lo) / 2.0, EPS)


def cell_scores(y, lo, hi, shift=None, sigma=None):
    """(n_cells, J) nonconformity scores; divide by sigma for the normalized variant."""
    s = 0.0 if shift is None else shift
    a = np.maximum(lo + s - y, y - hi - s)
    if sigma is not None:
        a = a / np.maximum(sigma, EPS)
    return a


def inner_quantile(a_cells, gamma=0.90, adjust=True):
    """Per-participant summary: the 'higher' empirical gamma-quantile of its cell scores.

    adjust=True uses the conformal-adjusted level min(ceil((m+1)gamma)/m, 1), matching the
    main calibration script; adjust=False uses the plain level used by the ablation scripts.
    The two differ by at most one order statistic.
    """
    m = a_cells.shape[0]
    level = min(np.ceil((m + 1) * gamma) / m, 1.0) if adjust else gamma
    return np.quantile(a_cells, level, axis=0, method="higher")


def participant_score_matrix(y, lo, hi, patient, gamma=0.90, shift_by_pat=None,
                             normalize=False, sd=None, adjust=True, pats=None):
    """-> (pats, R) with R of shape (n_participants, J); R[i, j] is r_ij."""
    pats = np.unique(patient) if pats is None else np.asarray(pats)
    sigma = None
    if normalize:
        sigma = half_width(lo, hi) if sd is None else np.maximum(np.asarray(sd), EPS)
    R = np.empty((len(pats), y.shape[1]), dtype=float)
    for i, p in enumerate(pats):
        m = patient == p
        s = None if shift_by_pat is None else shift_by_pat[p][None, :]
        sg = None if sigma is None else sigma[m]
        R[i] = inner_quantile(cell_scores(y[m], lo[m], hi[m], shift=s, sigma=sg), gamma, adjust)
    return pats, R


# --------------------------------------------------------------------------
# 1. Participant-block calibration (baseline and normalized share this rule)
# --------------------------------------------------------------------------

def block_lambda(R_cal, delta=0.20):
    """lam_j = k-th smallest of the n calibration participant scores, k = ceil((n+1)(1-delta)).

    Guarantee, per protein j, for exchangeable participants:
        P(new participant's within-participant coverage >= gamma) >= k/(n+1) >= 1-delta
    k > n returns +inf (uninformative interval) rather than silently capping the rank.
    """
    R_cal = np.atleast_2d(R_cal)
    n = R_cal.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    if k > n:
        return np.full(R_cal.shape[1], np.inf), k, 1.0
    return np.sort(R_cal, axis=0)[k - 1], k, k / (n + 1)


def cellwise_lambda(y, lo, hi, patient, pats_cal, shift_by_pat=None,
                    normalize=False, sd=None, gamma=0.90):
    """Pooled cell-wise split conformal: the comparator that ignores participant structure."""
    m = np.isin(patient, pats_cal)
    sigma = None
    if normalize:
        sigma = half_width(lo[m], hi[m]) if sd is None else np.maximum(sd[m], EPS)
    s = None
    if shift_by_pat is not None:
        s = np.stack([shift_by_pat[p] for p in patient[m]])
    a = cell_scores(y[m], lo[m], hi[m], shift=s, sigma=sigma)
    N = a.shape[0]
    level = min(np.ceil((N + 1) * gamma) / N, 1.0)
    return np.quantile(a, level, axis=0, method="higher")


# --------------------------------------------------------------------------
# 2. Panel-level tolerance: one statement about many proteins
# --------------------------------------------------------------------------

def panel_lambda(R_cal, scale, delta=0.20, tau=0.0):
    """Interpolates between per-protein and simultaneous coverage over the panel.

    With per-protein scales c_j fixed on DESIGN participants, let ratio_ij = r_ij / c_j and

        tau = 0 : R_i = max_j ratio_ij
        tau > 0 : R_i = the (1-tau) 'higher' quantile of {ratio_ij}_j

    Lam = k-th smallest of {R_i}, lam_j = Lam * c_j. Then with probability >= k/(n+1):
        tau = 0 : EVERY protein in the panel attains within-participant coverage >= gamma
        tau > 0 : at least a fraction (1-tau) of the panel does

    tau=1/J recovers a single-protein-may-fail statement; the per-protein rule is the
    J independent statements you get from block_lambda.
    """
    R_cal = np.atleast_2d(R_cal)
    c = np.maximum(np.asarray(scale, dtype=float), EPS)
    ratio = R_cal / c
    Rp = ratio.max(axis=1) if tau <= 0 else np.quantile(ratio, 1.0 - tau, axis=1, method="higher")
    n = len(Rp)
    k = int(np.ceil((n + 1) * (1.0 - delta)))
    if k > n:
        return np.full(len(c), np.inf), np.inf, k, 1.0
    Lam = np.sort(Rp)[k - 1]
    return Lam * c, Lam, k, k / (n + 1)


def panel_attainment(R_test, lam, tau=0.0):
    """Did each test participant satisfy the panel-level statement?"""
    ok = R_test <= lam[None, :]
    frac_ok = ok.mean(axis=1)
    return (frac_ok >= 1.0 - 1e-12) if tau <= 0 else (frac_ok >= 1.0 - tau - 1e-12)


# --------------------------------------------------------------------------
# 3. Mondrian group-conditional calibration
# --------------------------------------------------------------------------

def mondrian_lambda(R_cal, groups_cal, delta=0.20, shrink_n0=None):
    """Per-group lambda, optionally shrunk toward the pooled value.

    strict  : lam_j^g from that group's participants only. Valid conditionally on the group,
              but +inf whenever ceil((n_g+1)(1-delta)) > n_g (e.g. n_g=4 at delta=0.2).
    shrunk  : w_g*lam^g + (1-w_g)*lam_pooled with w_g = n_g/(n_g+shrink_n0).
              HEURISTIC -- this is a data-dependent blend and carries NO finite-sample
              guarantee. Report it as such; it buys width, not validity.
    """
    R_cal = np.atleast_2d(R_cal)
    g = np.asarray(groups_cal).astype(str)
    pooled, _, _ = block_lambda(R_cal, delta)
    out = {}
    for gg in np.unique(g):
        m = g == gg
        lam_g, k_g, cov_g = block_lambda(R_cal[m], delta)
        rec = {"n": int(m.sum()), "k": k_g, "bound": cov_g,
               "lambda_strict": lam_g, "valid_strict": bool(np.isfinite(lam_g).all())}
        if shrink_n0 is not None:
            w = rec["n"] / (rec["n"] + float(shrink_n0))
            base = np.where(np.isfinite(lam_g), lam_g, pooled)
            rec["lambda_shrunk"] = w * base + (1.0 - w) * pooled
            rec["shrink_weight"] = w
        out[gg] = rec
    out["_pooled"] = {"lambda": pooled, "n": int(R_cal.shape[0])}
    return out


# --------------------------------------------------------------------------
# 4. Conformal risk control for the selective call layer
# --------------------------------------------------------------------------

def crc_threshold(margin_grid, per_participant_loss, alpha=0.05, loss_bound=1.0):
    """Smallest margin whose CRC upper bound on a new participant's risk is <= alpha.

        h_hat = inf { h : (n * mean_i L_i(h) + B) / (n + 1) <= alpha }

    per_participant_loss: (n_cal, n_grid), L_i(h) must be NON-INCREASING in h.
    Use a FIXED-denominator loss -- wrong calls divided by all decisions, not by calls made.
    Error-among-calls is a ratio whose denominator also shrinks with h, is not monotone,
    and is therefore not a valid CRC loss; report it as an observed diagnostic only.

    Guarantee: E[L_{n+1}(h_hat)] <= alpha, in expectation over a new exchangeable participant.
    """
    L = np.asarray(per_participant_loss, dtype=float)
    n = L.shape[0]
    bound = (n * L.mean(axis=0) + loss_bound) / (n + 1)
    ok = np.flatnonzero(bound <= alpha)
    if len(ok) == 0:
        return None, bound, None
    idx = int(ok[0])
    return float(np.asarray(margin_grid)[idx]), bound, idx


def decision_losses(pred_frac, true_frac, threshold, participant, margin_grid):
    """Per-participant wrong-call rate over a margin grid, plus call-rate diagnostics.

    A decision is CALLED when |pred_frac - threshold| > h, and is WRONG when the predicted
    and the measured fraction fall on opposite sides of the threshold.
    """
    pred_frac = np.asarray(pred_frac, float)
    true_frac = np.asarray(true_frac, float)
    thr = np.asarray(threshold, float)
    part = np.asarray(participant).astype(str)
    grid = np.asarray(margin_grid, float)

    margin = np.abs(pred_frac - thr)
    wrong = (pred_frac > thr) != (true_frac > thr)
    pats = np.unique(part)

    loss = np.empty((len(pats), len(grid)))
    rate = np.empty_like(loss)
    err = np.full_like(loss, np.nan)
    for i, p in enumerate(pats):
        m = part == p
        mg, wr = margin[m], wrong[m]
        tot = max(m.sum(), 1)
        for t, h in enumerate(grid):
            called = mg > h
            nc = called.sum()
            loss[i, t] = wr[called].sum() / tot          # monotone, CRC-valid
            rate[i, t] = nc / tot
            if nc:
                err[i, t] = wr[called].mean()            # diagnostic only
    return pats, loss, rate, err


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def attainment(R_test, lam):
    """Fraction of test participants attaining within-participant coverage >= gamma, per protein."""
    ok = R_test <= lam[None, :]
    return ok.mean(axis=0), ok


def interval_report(y, lo, hi, patient, pats_test, lam, shift_by_pat=None,
                    normalize=False, sd=None, alpha=0.10):
    """Empirical coverage, median width and interval score on the widened intervals."""
    m = np.isin(patient, pats_test)
    yy, ll, hh = y[m], lo[m], hi[m]
    s = 0.0 if shift_by_pat is None else np.stack([shift_by_pat[p] for p in patient[m]])
    if normalize:
        sg = half_width(ll, hh) if sd is None else np.maximum(sd[m], EPS)
        wid = lam[None, :] * sg
    else:
        wid = np.broadcast_to(lam[None, :], yy.shape)
    lo_f, hi_f = ll + s - wid, hh + s + wid
    cov = (yy >= lo_f) & (yy <= hi_f)
    width = hi_f - lo_f
    isc = width + (2.0 / alpha) * (np.clip(lo_f - yy, 0, None) + np.clip(yy - hi_f, 0, None))
    return {"coverage": float(cov.mean()),
            "median_width": float(np.median(width)),
            "interval_score": float(isc.mean())}


# --------------------------------------------------------------------------
# Diagnostic: will normalization pay off on THIS data?
# --------------------------------------------------------------------------

def normalization_diagnostic(y, mean_, lo, hi, patient, pats, sd=None):
    """Rank correlation between the base interval half-width and |residual|, per protein.

    Normalized scores reclaim width only where the base interval already tracks error
    magnitude. Run this before adopting them: a median rho near 0 means the base widths
    carry no difficulty signal and multiplicative widening will not help.
    """
    from scipy.stats import spearmanr
    m = np.isin(patient, pats)
    sig = half_width(lo[m], hi[m]) if sd is None else np.maximum(sd[m], EPS)
    resid = np.abs(y[m] - mean_[m])
    rho = np.array([spearmanr(sig[:, j], resid[:, j]).statistic for j in range(y.shape[1])])
    return {"rho_per_protein": rho, "median_rho": float(np.median(rho)),
            "frac_proteins_rho_gt_0.2": float((rho > 0.2).mean())}
