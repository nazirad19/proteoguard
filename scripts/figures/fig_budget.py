"""Render Figure 7 from anchor_budget.csv, never from embedded result values.

Aggregate each fit separately, then average fits, matching anchor_budget.py.
The multi map equals the shared map at K=0 and K=1; those rows are copied
only where the analysis intentionally omitted them. Missing larger budgets
are errors, not interpolated results.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"fit", "predictor", "map", "K", "rep", "protein", "family",
            "offset_r2", "below_85", "n_calls", "n_wrong", "n_decisions"}


def summarize(frame):
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"anchor_budget.csv missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("anchor_budget.csv has no rows")
    keys = ["fit", "predictor", "map", "K", "rep", "protein"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate fit/predictor/map/budget/rep/protein rows")
    if ((frame.n_wrong > frame.n_calls) | (frame.n_calls > frame.n_decisions) |
            (frame[["n_wrong", "n_calls", "n_decisions"]] < 0).any(axis=1)).any():
        raise ValueError("Invalid biomarker call counts")
    # R2 is repeated across calibration resamples; count each protein once.
    rows = []
    for key, group in frame.groupby(["fit", "predictor", "map", "K"], sort=True):
        if (group.groupby("protein").offset_r2.nunique() > 1).any():
            raise ValueError("Offset R2 changes across resamples for the same protein")
        rows.append(dict(zip(["fit", "predictor", "map", "K"], key),
            offset_r2=group.groupby("protein").offset_r2.first().median(),
            below_85=group.below_85.mean(),
            error_among_calls=group.n_wrong.sum() / group.n_calls.sum() if group.n_calls.sum() else np.nan,
            call_rate=group.n_calls.sum() / group.n_decisions.sum() if group.n_decisions.sum() else np.nan))
    per_fit = pd.DataFrame(rows)
    copies = []
    for _, row in per_fit[(per_fit["map"] == "shared") & per_fit.K.isin([0, 1])].iterrows():
        exists = ((per_fit.fit == row.fit) & (per_fit.predictor == row.predictor) &
                  (per_fit["map"] == "multi") & (per_fit.K == row.K)).any()
        if not exists:
            row = row.copy(); row["map"] = "multi"; copies.append(row)
    if copies:
        per_fit = pd.concat([per_fit, pd.DataFrame(copies)], ignore_index=True)
    # Require comparable fit sets at each budget rather than silently dropping failed runs.
    expected = set(per_fit.fit)
    budgets = sorted(per_fit.K.unique())
    for (_, _), group in per_fit.groupby(["predictor", "map"]):
        if set(group.K) != set(budgets):
            raise ValueError("Incomplete budget curve; a predictor/map is missing a budget")
        if any(set(g.fit) != expected for _, g in group.groupby("K")):
            raise ValueError("Incomplete fit set at a budget")
    return per_fit.groupby(["predictor", "map", "K"], as_index=False)[
        ["offset_r2", "below_85", "error_among_calls", "call_rate"]].mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    summary = summarize(frame)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    colors = ["#245EFF", "#D64B00", "#117D54", "#7937FF"]
    budgets = sorted(summary.K.unique()); x = np.arange(len(budgets))
    for i, ((pred, kind), group) in enumerate(summary.groupby(["predictor", "map"])):
        group = group.set_index("K").loc[budgets]
        for ax, metric, scale in zip(axes[:3], ["offset_r2", "below_85", "error_among_calls"], [1, 100, 100]):
            ax.plot(x, group[metric] * scale, "o-", color=colors[i % len(colors)], label=f"{pred}, {kind}")
    family = frame[(frame.predictor == "totalVI") & (frame.family != "other")]
    for i, (name, group) in enumerate(family.groupby("family")):
        g = summarize(group); g = g[g["map"] == "multi"].sort_values("K")
        axes[3].plot([budgets.index(k) for k in g.K], g.offset_r2, "o-", label=name, color=colors[i % len(colors)])
    titles = ["Offset R² (median over proteins)", "Pairs below 85% coverage (%)",
              "Error among biomarker calls (%)", "Offset R² by family (totalVI, multi)"]
    for ax, title in zip(axes, titles):
        ax.set_xticks(x, budgets); ax.set_xlabel("Measured sentinel proteins"); ax.set_title(title, fontsize=10); ax.grid(alpha=.25)
    axes[0].set_ylim(-.05, 1); axes[3].set_ylim(-.05, 1)
    axes[3].legend(fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .12, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight"); plt.close(fig)
    summary.to_csv(args.output.with_suffix(".summary.csv"), index=False)
    print(f"Saved {args.output}; aggregated from {args.input}")


if __name__ == "__main__":
    main()
