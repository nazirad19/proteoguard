"""
run_upgrades.py -- apply the ProteoGuard calibration upgrades to real cohort predictions.

Usage:
    python run_upgrades.py --pred /path/pred_ridgeanchor_fit2026.npz \
                           --design design_ids.txt --cal cal_ids.txt --test test_ids.txt \
                           --sentinels sentinels_fit2026.json --out results/

Participant role files are one ID per line. Sentinel/isotype proteins are excluded from the
evaluation set, matching the 177-179 matched target list.
"""
import argparse, json, os
import numpy as np, pandas as pd
import proteoguard_upgrades as pg

GAMMA, DELTA = 0.90, 0.20


def read_ids(p):
    return np.array([l.strip() for l in open(p) if l.strip()], dtype=str)


def fit_shared_map(y, mean_, patient, design, sent_idx, pats_all):
    """Design-fit shared-offset map; also returns the common-offset control."""
    B = {p: np.median((y - mean_)[patient == p], axis=0) for p in pats_all}
    x = {p: B[p][sent_idx].mean() for p in pats_all}
    Xd = np.array([x[p] for p in design]); Bd = np.array([B[p] for p in design])
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(Xd)), Xd], Bd, rcond=None)
    shared = {p: coef[0] + coef[1] * x[p] for p in pats_all}
    common = {p: np.full(y.shape[1], x[p]) for p in pats_all}
    return shared, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--design", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--sentinels", required=True, nargs="+",
                    help="one JSON per predictor; ALL are excluded, matching the paper's "
                         "'all proteins minus anchors of either predictor' target set")
    ap.add_argument("--isotypes", default=None,
                    help="file listing isotype-control channel names, one per line. "
                         "Required to reproduce the 177-179 matched set; name matching is "
                         "NOT used because real markers (IgD, IgM) would be caught by it.")
    ap.add_argument("--expect-targets", type=int, default=None,
                    help="fail unless the evaluation set has exactly this many proteins")
    ap.add_argument("--group-key", default="site", choices=["site", "celltype"])
    ap.add_argument("--out", default="upgrade_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = pg.load_predictions(a.pred)
    if not pg.check_contiguous(d["patient"]):
        raise SystemExit("cells are not contiguous by participant -- participant slicing is unsafe")

    prot = d["protein"]

    # Evaluation set = all proteins minus the anchors of EVERY predictor and the isotypes,
    # so the new numbers are comparable with the paper's 177-179 matched target list.
    sent_names = set()
    for f in a.sentinels:
        obj = json.load(open(f))
        if isinstance(obj, dict):
            vals = [v for v in obj.values() if isinstance(v, list)]
            if len(vals) != 1:
                raise SystemExit(f"{f}: ambiguous sentinel JSON with keys {list(obj)}; "
                                 "pass a flat list or split the file per predictor")
            obj = vals[0]
        sent_names |= {str(x) for x in obj}
    sent_idx = np.array([i for i, p in enumerate(prot) if p in sent_names], dtype=int)
    if len(sent_idx) != len(sent_names):
        missing = sent_names - set(prot[sent_idx])
        raise SystemExit(f"sentinels absent from this prediction file: {sorted(missing)}")

    if a.isotypes:
        iso_names = {l.strip() for l in open(a.isotypes) if l.strip()}
        iso = np.array([i for i, p in enumerate(prot) if p in iso_names], dtype=int)
        if len(iso) != len(iso_names):
            raise SystemExit(f"isotype names absent from this file: "
                             f"{sorted(iso_names - set(prot[iso]))}")
    else:
        iso = np.array([], dtype=int)
        print("WARNING: no --isotypes file; isotype channels remain in the evaluation set, "
              "so the target count will not match the paper's 177-179.")
    keep = np.setdiff1d(np.arange(len(prot)), np.union1d(sent_idx, iso))
    print(f"{len(prot)} proteins -> {len(keep)} evaluation targets "
          f"({len(sent_idx)} sentinels, {len(iso)} isotypes excluded)")
    if a.expect_targets is not None and len(keep) != a.expect_targets:
        raise SystemExit(f"evaluation set is {len(keep)}, expected {a.expect_targets} -- "
                         "the target list does not match the paper's matched set")

    design, cal, test = read_ids(a.design), read_ids(a.cal), read_ids(a.test)
    pats_all = np.unique(d["patient"])
    shared_f, common_f = fit_shared_map(d["y"], d["mean"], d["patient"], design, sent_idx, pats_all)
    shared = {p: v[keep] for p, v in shared_f.items()}
    common = {p: v[keep] for p, v in common_f.items()}
    y, lo, hi = d["y"][:, keep], d["lo"][:, keep], d["hi"][:, keep]
    mean_ = d["mean"][:, keep]
    patient = d["patient"]

    print("\n[diagnostic] does the base width track error?")
    dg = pg.normalization_diagnostic(y, mean_, lo, hi, patient, test)
    print(f"  median rho = {dg['median_rho']:.3f}; "
          f"{dg['frac_proteins_rho_gt_0.2']:.0%} of proteins above 0.2 "
          f"-> normalization {'should' if dg['median_rho'] > 0.15 else 'is unlikely to'} help")

    rows = []
    for cname, shift in (("shared map", shared), ("common offset", common), ("none", None)):
        for norm in (False, True):
            sc = lambda P: pg.participant_score_matrix(y, lo, hi, patient, GAMMA, shift, norm, pats=P)[1]
            Rc, Rt = sc(cal), sc(test)
            lam, k, bnd = pg.block_lambda(Rc, DELTA)
            rep = pg.interval_report(y, lo, hi, patient, test, lam, shift, norm)
            rows.append(dict(correction=cname, score="normalized" if norm else "absolute",
                             calibration="participant-block", k=k, bound=bnd,
                             attain=pg.attainment(Rt, lam)[0].mean(), **rep))
            lcw = pg.cellwise_lambda(y, lo, hi, patient, cal, shift, norm, gamma=GAMMA)
            rows.append(dict(correction=cname, score="normalized" if norm else "absolute",
                             calibration="cell-wise", k=np.nan, bound=np.nan,
                             attain=pg.attainment(Rt, lcw)[0].mean(),
                             **pg.interval_report(y, lo, hi, patient, test, lcw, shift, norm)))
    pd.DataFrame(rows).to_csv(f"{a.out}/calibration_rules.csv", index=False)

    sc = lambda P: pg.participant_score_matrix(y, lo, hi, patient, GAMMA, shared, False, pats=P)[1]
    Rd, Rc, Rt = sc(design), sc(cal), sc(test)
    c_j = pg.block_lambda(Rd, DELTA)[0]
    prows = []
    lam_pp = pg.block_lambda(Rc, DELTA)[0]
    prows.append(dict(rule="per-protein", tau=np.nan,
                      whole_panel_ok=float((Rt <= lam_pp[None, :]).all(axis=1).mean()),
                      **pg.interval_report(y, lo, hi, patient, test, lam_pp, shared)))
    for tau in (0.0, 1.0 / len(keep), 0.05, 0.10):
        lam_p = pg.panel_lambda(Rc, c_j, DELTA, tau=tau)[0]
        prows.append(dict(rule="panel tolerance", tau=tau,
                          whole_panel_ok=float(pg.panel_attainment(Rt, lam_p, tau).mean()),
                          **pg.interval_report(y, lo, hi, patient, test, lam_p, shared)))
    pd.DataFrame(prows).to_csv(f"{a.out}/panel_tolerance.csv", index=False)

    gkey = d.get(a.group_key)
    if gkey is not None:
        gof = {p: pg.first_block_value(gkey, patient, p) for p in pats_all}
        mon = pg.mondrian_lambda(Rc, [gof[p] for p in cal], DELTA, shrink_n0=5.0)
        mrows = []
        for g, r in mon.items():
            if g == "_pooled":
                continue
            tm = np.array([gof[p] == g for p in test])
            rec = dict(group=g, n_cal=r["n"], k=r["k"], strict_finite=r["valid_strict"],
                       n_test=int(tm.sum()),
                       attain_pooled=float(pg.attainment(Rt[tm], lam_pp)[0].mean()) if tm.any() else np.nan)
            if r["valid_strict"] and tm.any():
                rec["attain_strict"] = float(pg.attainment(Rt[tm], r["lambda_strict"])[0].mean())
            if "lambda_shrunk" in r and tm.any():
                rec["attain_shrunk_HEURISTIC"] = float(pg.attainment(Rt[tm], r["lambda_shrunk"])[0].mean())
            mrows.append(rec)
        pd.DataFrame(mrows).to_csv(f"{a.out}/mondrian_{a.group_key}.csv", index=False)

    n = len(cal)
    pd.DataFrame([dict(n_calibration=n, min_certifiable_selective_risk=1.0 / (n + 1),
                       n_needed_for_5pct=int(np.ceil(1 / 0.05 - 1)),
                       n_needed_for_2pct=int(np.ceil(1 / 0.02 - 1)))]).to_csv(
        f"{a.out}/crc_risk_floor.csv", index=False)
    print(f"\nwrote {a.out}/  -- calibration_rules, panel_tolerance, mondrian, crc_risk_floor")
    print(f"CRC floor at n={n}: no selective risk below {1/(n+1):.4f} can be certified.")


if __name__ == "__main__":
    main()
