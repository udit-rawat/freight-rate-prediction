"""My validation harness.

This is where I spent most of my time, and the report is mostly about it.

Why this is its own module rather than part of model.py
-------------------------------------------------------
The task looks like a regression but it is a forecast. Training labels stop on
31 October and my predictions run from 1 November to 31 December, so I never see
the answer for a single date I am scored on.

The obvious thing is a random 80/20 split. I did not use one, because it puts
rows from every date in both halves, so the model trains on 15 October and gets
tested on 14 October. It has already seen the market on the days it is being
marked on. The score comes out lovely and means nothing. What makes it dangerous
is that nothing warns you. There is no error, just a good number.

What I did instead, and why
---------------------------
I train on a stretch of history and test on the 61 days straight after it, three
times, stepping forward. The training window grows each time instead of sliding,
because the model I actually ship is fitted on everything I have, so the test
should work the same way.

61 days because that is the real gap. My predictions run 1 to 61 days past the
last label, and one 61 day window covers that whole range. A shorter window would
flatter me by testing an easier problem.

No gap between training and testing. I considered adding one. The real task has a
one day gap, so a two week buffer would have made my test harder than the thing I
am marked on and pushed me toward a needlessly timid model.

The leak worth worrying about is not the gap anyway. It is anything built from the
answer column, like an average rate per lane. Those have to come only from rows
inside the fold's own training window. `evaluate` handles that for me by passing
a training frame and a test frame with the answer stripped out, so a fold
physically cannot see past its own window.

Three folds because that is what the data allows. 304 training days and a 61 day
horizon means a fourth fold drops the smallest training window under four months,
and then I am measuring "not enough data" rather than "how hard is forecasting".

One caveat for anyone reading the numbers. The folds are not equally fair. Fold
one tests a rising market at its peak and fold three tests a softening one. I am
predicting a soft November and December, so fold three is the closest match and I
leaned on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# --- Fold construction -----------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One train and test boundary, held as dates rather than row numbers."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def train_mask(self, frame: pd.DataFrame) -> pd.Series:
        dates = frame[C.DATE_COL]
        return (dates >= self.train_start) & (dates <= self.train_end)

    def test_mask(self, frame: pd.DataFrame) -> pd.Series:
        dates = frame[C.DATE_COL]
        return (dates >= self.test_start) & (dates <= self.test_end)

    def describe(self) -> str:
        return (
            f"fold {self.index}  train {self.train_start.date()} to {self.train_end.date()}"
            f"  test {self.test_start.date()} to {self.test_end.date()}"
        )


def forward_chain_folds(
    frame: pd.DataFrame,
    n_folds: int = 3,
    horizon_days: int = 61,
) -> list[Fold]:
    """Build the folds, ending at the last day in the frame.

    The test windows are the final `n_folds` blocks of `horizon_days` with no
    overlap, and each fold trains on everything before its own block.
    """
    start = frame[C.DATE_COL].min()
    end = frame[C.DATE_COL].max()
    total_days = (end - start).days + 1

    required = n_folds * horizon_days
    if required >= total_days:
        raise ValueError(
            f"{n_folds} folds of {horizon_days} days needs more than {total_days} available days"
        )

    folds: list[Fold] = []
    for position in range(n_folds):
        test_end = end - pd.Timedelta(days=horizon_days * (n_folds - 1 - position))
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        folds.append(
            Fold(
                index=position + 1,
                train_start=start,
                train_end=test_start - pd.Timedelta(days=1),
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def matched_random_folds(
    frame: pd.DataFrame,
    folds: list[Fold],
    seed: int = C.RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random splits with the same row counts, so I can price the leak.

    Each random fold draws from exactly the rows its dated counterpart used and
    keeps the same train and test sizes. So the only thing that differs between
    the two harnesses is how I drew the boundary, not how much data was seen.
    """
    rng = np.random.default_rng(seed)
    matched = []
    for fold in folds:
        train_rows = frame.index[fold.train_mask(frame)].to_numpy()
        test_rows = frame.index[fold.test_mask(frame)].to_numpy()
        pool = np.concatenate([train_rows, test_rows])
        shuffled = rng.permutation(pool)
        matched.append((shuffled[: len(train_rows)], shuffled[len(train_rows):]))
    return matched


# --- Metrics ---------------------------------------------------------------


def score(actual: pd.Series, predicted: np.ndarray, clean: pd.Series | None = None) -> dict:
    """How wrong I am, measured on the posted rate since that is what I deliver.

    `mae` is the average number of dollars I am off by and `mape` is the same as a
    percentage. `medae` is the median error, which steps over the inflated loads.
    `mae_clean` is the average with those same rows taken out.

    I report both because roughly 0.69% of loads have been inflated at random.
    Nobody can predict them, so they put a floor under the average that has
    nothing to do with how good my model is.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = np.abs(actual - predicted)

    result = {
        "mae": float(error.mean()),
        "mape": float((error / actual).mean() * 100),
        "medae": float(np.median(error)),
    }
    if clean is not None:
        keep = np.asarray(clean, dtype=bool)
        result["mae_clean"] = float(error[keep].mean())
        result["outlier_share"] = float((~keep).mean() * 100)
    return result


# --- Evaluation ------------------------------------------------------------


def evaluate(fit_predict, frame: pd.DataFrame, folds: list[Fold], label: str) -> pd.DataFrame:
    """Run a model across every fold and return one row of scores per fold.

    `fit_predict(train, test_features)` has to return a predicted rate for each
    row of `test_features`. I strip the answer column out of the test frame before
    the model sees it, so it cannot read the answer even by accident.
    """
    inflated = frame["_inflated"] if "_inflated" in frame else None
    rows = []

    for fold in folds:
        train = frame[fold.train_mask(frame)]
        test = frame[fold.test_mask(frame)]
        test_features = test.drop(columns=[C.RAW_TARGET])

        predicted = fit_predict(train, test_features)
        clean = ~inflated.loc[test.index] if inflated is not None else None

        rows.append({
            "model": label,
            "harness": "forward_chain",
            "fold": fold.index,
            "train_end": fold.train_end.date(),
            "test_start": fold.test_start.date(),
            "test_end": fold.test_end.date(),
            "n_train": len(train),
            "n_test": len(test),
            **score(test[C.RAW_TARGET], predicted, clean),
        })
    return pd.DataFrame(rows)


def evaluate_random(fit_predict, frame: pd.DataFrame, folds: list[Fold], label: str) -> pd.DataFrame:
    """The same scoring over random splits, only so I can price the leak.

    This deliberately makes the mistake the rest of this module exists to avoid,
    so I can show what a random split costs in dollars instead of just claiming it
    is a problem.
    """
    inflated = frame["_inflated"] if "_inflated" in frame else None
    rows = []

    for fold, (train_rows, test_rows) in zip(folds, matched_random_folds(frame, folds)):
        train = frame.loc[train_rows]
        test = frame.loc[test_rows]
        test_features = test.drop(columns=[C.RAW_TARGET])

        predicted = fit_predict(train, test_features)
        clean = ~inflated.loc[test.index] if inflated is not None else None

        rows.append({
            "model": label,
            "harness": "random",
            "fold": fold.index,
            "train_end": "shuffled",
            "test_start": "shuffled",
            "test_end": "shuffled",
            "n_train": len(train),
            "n_test": len(test),
            **score(test[C.RAW_TARGET], predicted, clean),
        })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Squash the per fold rows into one line each, keeping the spread."""
    return (
        results.groupby(["model", "harness"], sort=False)
        .agg(
            folds=("fold", "count"),
            mae=("mae", "mean"),
            mae_sd=("mae", "std"),
            mape=("mape", "mean"),
            medae=("medae", "mean"),
            mae_clean=("mae_clean", "mean"),
        )
        .round(3)
        .reset_index()
    )
