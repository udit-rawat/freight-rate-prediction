"""Charts. Everything here reads from results/ and data/, nothing recomputes.

Rendered at 1600x900 so they stay legible when a feed scales them down.
"""

from __future__ import annotations

import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C, data as D, split as S

FIG = C.RESULTS_DIR / "figures"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.figsize": (8, 4.5), "figure.dpi": 200,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": "#b8b7b2", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    # Dollar signs are data here, not maths. Without this, a string containing
    # two of them gets parsed as LaTeX and rendered as italic nonsense.
    "text.parse_math": False,
})


def frame(ax, title, subtitle=None, wrap=88):
    """Title above subtitle, both left aligned, neither overlapping the other.

    matplotlib puts the title just above the axes, so a subtitle has to be given
    room explicitly rather than dropped at a hopeful y offset.
    """
    lines = textwrap.wrap(subtitle, wrap) if subtitle else []
    pad = 20 + 13 * len(lines)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=INK, pad=pad)
    # offset points from the top-left of the axes: exact, unlike an axes-fraction
    # guess, which drifts with the subplot geometry
    for i, line in enumerate(lines):
        ax.annotate(line, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 5 + 13 * (len(lines) - 1 - i)),
                    textcoords="offset points", fontsize=9, color=MUTED, va="bottom")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"  results/figures/{name}")


# --- 1. target formulation -------------------------------------------------

def fig_target(train):
    rate = train[C.RAW_TARGET]
    rpm = rate / train["distance"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    for ax, series, colour, label, cv in (
        (axes[0], rate, ORANGE, "Raw posted rate ($)", rate.std() / rate.mean()),
        (axes[1], rpm, BLUE, "Rate per mile ($/mi)", rpm.std() / rpm.mean()),
    ):
        clipped = series[series < series.quantile(0.995)]
        ax.hist(clipped, bins=70, color=colour, edgecolor=SURFACE, linewidth=0.3)
        ax.set_xlabel(label, fontsize=9)
        ax.set_yticks([])
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(0.96, 0.93, f"CV {cv:.3f}", transform=ax.transAxes, ha="right",
                fontsize=15, fontweight="bold", color=colour)
    fig.suptitle("Same loads, two targets", x=0.02, y=0.99, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.925, "Dividing by distance removes 56% of the variance "
             "before a model sees anything", fontsize=9, color=MUTED)
    fig.subplots_adjust(top=0.84)
    save(fig, "01_target_formulation.png", tight=False)


# --- 2. validation design --------------------------------------------------

def fig_split(train):
    folds = S.forward_chain_folds(train, 3, 61)
    fig, ax = plt.subplots()
    rows = [(f"Fold {f.index}", f.train_start, f.train_end, f.test_start, f.test_end)
            for f in folds]
    rows.append(("Real task", train[C.DATE_COL].min(), train[C.DATE_COL].max(),
                 pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31")))
    for i, (label, ts, te, vs, ve) in enumerate(rows):
        y = len(rows) - 1 - i
        final = label == "Real task"
        ax.barh(y, (te - ts).days, left=ts, height=0.44, color=BLUE,
                alpha=0.30 if final else 0.85)
        ax.barh(y, (ve - vs).days, left=vs, height=0.44,
                color=ORANGE, alpha=0.45 if final else 1.0,
                hatch="///" if final else None, edgecolor=SURFACE)
        ax.text(ts + pd.Timedelta(days=4), y, label, va="center",
                fontsize=9, color="white" if not final else MUTED, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp("2024-12-20"), pd.Timestamp("2026-01-12"))
    ax.axvline(pd.Timestamp("2025-11-01"), color=MUTED, linestyle=":", linewidth=1.1)
    ax.text(pd.Timestamp("2025-11-05"), 3.34, "labels stop", fontsize=8.5, color=MUTED)
    frame(ax, "Forward chaining, not a random split",
          "Each fold trains on history and is tested on the 61 days straight after it, matching the gap in the real task")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.set_ylim(-0.75, 3.62)
    ax.legend(handles, ["train", "test"], frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=9)
    save(fig, "02_validation_split.png")


# --- 3. ablation -----------------------------------------------------------

def fig_ablation(abl):
    """The ladder, taken from model.ABLATION so it cannot drift out of step."""
    from src import model as M

    order = list(M.ABLATION)
    labels = ["null baseline"] + order + ["lane target encoding"]
    means = abl.groupby("model", sort=False).mae.mean()
    values = ([means["null (median rpm)"]] + [means[o] for o in order]
              + [means["rejected: lane target encoding"]])

    colours = [MUTED] + [BLUE] * len(order) + [ORANGE]
    colours[1 + order.index(M.SELECTED_LABEL)] = AQUA

    fig, ax = plt.subplots()
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.55, color=colours)
    for yi, v, lab in zip(y, values, labels):
        note = ""
        if lab == M.SELECTED_LABEL:
            note = "   shipped"
        elif lab == "lane target encoding":
            note = "   rejected, costs $24"
        ax.text(v + 3, yi, f"${v:,.0f}{note}", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("Mean absolute error, dollars, across three folds")
    ax.set_xlim(0, max(values) * 1.30)
    frame(ax, "What each feature group was actually worth",
          "Added one at a time. The best score is lane native at $219; the calendar groups cost "
          "$9.70 against a $30 fold spread, and I shipped them anyway so the model can express season")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    save(fig, "03_ablation.png")


# --- 4. the error floor ----------------------------------------------------

def fig_floor(abl, inflated_pct):
    g = abl.groupby("model")[["mae", "mae_clean"]].mean()
    null, best = g.loc["null (median rpm)"], g.loc["+ fourier season"]
    fig, ax = plt.subplots()
    labels = ["Null baseline", "Final model"]
    reducible = [null.mae_clean, best.mae_clean]
    floor = [null.mae - null.mae_clean, best.mae - best.mae_clean]
    y = [1, 0]
    ax.barh(y, reducible, height=0.34, color=BLUE, label="error the model can attack")
    ax.barh(y, floor, height=0.34, left=reducible, color=ORANGE,
            label="floor from randomly inflated loads",
            edgecolor=SURFACE, linewidth=1.6)
    for yi, r, f in zip(y, reducible, floor):
        ax.text(r / 2, yi, f"${r:,.0f}", va="center", ha="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(r + f / 2, yi, f"${f:,.0f}", va="center", ha="center",
                color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlabel("Mean absolute error, dollars")
    pct = floor[1] / (reducible[1] + floor[1]) * 100
    frame(ax, "Part of the error was never reachable",
          f"{inflated_pct:.2f}% of loads are inflated at random. They contribute about the same dollars "
          f"to the baseline and to the tuned model, and are {pct:.0f}% of what is left")
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_ylim(-0.55, 1.55)
    ax.legend(frameon=False, fontsize=9, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, -0.34))
    save(fig, "04_error_floor.png")


# --- 5. the December probe -------------------------------------------------

def fig_december():
    d = pd.read_csv(C.DECEMBER_CSV, parse_dates=["date"])
    r = d.predicted_rate.astype(float)
    fig, ax = plt.subplots()
    ax.plot(d.date, r, color=BLUE, linewidth=2.0, marker="o", markersize=3.6)
    ax.fill_between(d.date, r, r.min() - 3, color=BLUE, alpha=0.07)
    peak = d.date[r.idxmax()]
    ax.annotate("peak season tightens\ninto Christmas",
                xy=(peak, r.max()), xytext=(-120, -30), textcoords="offset points",
                fontsize=9, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    # the cliff itself, not the lowest point: the biggest single day fall
    trough = int(np.argmin(np.diff(r.to_numpy()))) + 1
    ax.annotate("freight stops moving,\nmarket falls away",
                xy=(d.date[trough], r[trough]), xytext=(-96, 34), textcoords="offset points",
                fontsize=9, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1))
    ax.set_ylabel("Predicted rate ($)")
    ax.set_ylim(r.min() - 8, r.max() + 12)
    ax.set_xlim(d.date.min() - pd.Timedelta(days=1), d.date.max() + pd.Timedelta(days=1))
    frame(ax, "One lane, every day of December",
          "Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Only the date changes, so the shape is purely what the model learned about time")
    fig.autofmt_xdate(rotation=35)
    save(fig, "05_december_probe.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train = D.repair_weight(D.load_train())
    abl = pd.read_csv(C.RESULTS_DIR / "ablation_scores.csv")
    fig_target(train); fig_split(train); fig_ablation(abl)
    fig_floor(abl, D.flag_inflated(train).mean() * 100)
    fig_december()


if __name__ == "__main__":
    main()
