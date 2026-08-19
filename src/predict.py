"""Writing the two files I have to hand in.

What comes out
--------------
1. `validation_predictions.csv` at the repository root. Two columns exactly,
   12,000 rows, ids TE-000001 to TE-012000, every rate above zero. score.py
   rejects anything else.

2. `data/december_chart_inputs.csv` filled in place. score.py reads it at that
   path, and filling it in place means the git diff on that file is itself proof
   of what I predicted. I rewrite it from the original strings so only the last
   column changes.

Both are generated, and both are committed anyway. They are the output the
project exists to produce, so a reader should not have to run an eight minute
pipeline before the repo says anything.

The final model is fitted on all 48,000 training rows. The harness picked the
model, and once it has I want the shipped version using everything I have, since
the most recent stretch is also the most relevant to November and December.

One thing to remember: the model predicts rate per mile, so I multiply back by
distance before writing anything out.

Checking the output
-------------------
score.py checks the format and nothing else. It would happily accept a file that
prices a 2,400 mile load at 50 dollars. There is no answer key for November and
December, so nothing will ever tell me these numbers are wrong. `sanity_report`
is my only defence, and it checks that the patterns I learned from training are
still there in the output.

It is a correctness check, not something to tune against. Changing the model after
looking at predictions I cannot score would just be fitting to my own
expectations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import features as F
from . import model as M


# --- Prediction ------------------------------------------------------------


def predict_validation(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    fit_predict = M.make_gbdt(M.SELECTED)
    return fit_predict(train, validation)


def predict_december(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Fill in the chart's missing columns, then predict its 31 rows."""
    december = F.complete_december(D.load_december(), validation, train)
    fit_predict = M.make_gbdt(M.SELECTED, market_frames=[validation])
    december["prediction"] = fit_predict(train, december.drop(columns=["predicted_rate"]))
    return december


# --- Writing ---------------------------------------------------------------


def write_validation_predictions(validation: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Write the two column file, checking every rule score.py enforces first."""
    out = pd.DataFrame({
        C.ID_COL: validation[C.ID_COL].to_numpy(),
        "predicted_rate": np.round(predicted.astype(float), 2),
    })

    expected = {f"TE-{index:06d}" for index in range(1, 12_001)}
    assert list(out.columns) == ["load_id", "predicted_rate"], "column names or order wrong"
    assert len(out) == 12_000, f"expected 12,000 rows, got {len(out):,}"
    assert not out[C.ID_COL].duplicated().any(), "duplicate load_id"
    assert set(out[C.ID_COL]) == expected, "load_id set does not match the template"
    assert np.isfinite(out.predicted_rate).all(), "non-finite prediction"
    assert (out.predicted_rate > 0).all(), "non-positive prediction"

    out.to_csv(C.PREDICTIONS_CSV, index=False)
    return out


def write_december(december: pd.DataFrame) -> pd.DataFrame:
    """Fill in the rate column, leaving every other value exactly as it was.

    I re-read the original file as plain strings so that 360 does not turn into
    360.0 and the dates keep their format. Only the last column changes, which
    keeps the git diff readable as evidence of what I predicted.
    """
    raw = pd.read_csv(C.DECEMBER_CSV, dtype=str)
    lookup = dict(zip(december[C.DATE_COL].dt.strftime("%Y-%m-%d"),
                      december["prediction"].round(2)))

    raw["predicted_rate"] = raw["date"].map(lookup)
    assert raw["predicted_rate"].notna().all(), "a December date failed to match"
    assert (raw["predicted_rate"].astype(float) > 0).all(), "non-positive December prediction"
    assert list(raw.columns) == ["pickup", "delivery", "distance", "equipment",
                                 "weight", "date", "predicted_rate"], "column order changed"
    assert len(raw) == 31, f"expected 31 rows, got {len(raw)}"

    raw.to_csv(C.DECEMBER_CSV, index=False)
    return raw


# --- Sanity ----------------------------------------------------------------


def sanity_report(train: pd.DataFrame, validation: pd.DataFrame,
                  predicted: np.ndarray) -> None:
    """Check the patterns I learned from training are still in the output."""
    train = train.copy()
    train["rpm"] = train[C.RAW_TARGET] / train["distance"]
    check = validation.copy()
    check["predicted_rate"] = predicted
    check["rpm"] = check.predicted_rate / check.distance

    print("  rate distribution, training actual against predicted")
    stats = pd.DataFrame({
        "train_actual": train[C.RAW_TARGET].describe(percentiles=[.25, .5, .75]),
        "predicted": check.predicted_rate.describe(percentiles=[.25, .5, .75]),
    })
    print(stats.round(2).to_string())

    print("\n  implied rate per mile")
    print(f"    training  mean {train.rpm.mean():.4f}  sd {train.rpm.std():.4f}  "
          f"1% {train.rpm.quantile(.01):.3f}  99% {train.rpm.quantile(.99):.3f}")
    print(f"    predicted mean {check.rpm.mean():.4f}  sd {check.rpm.std():.4f}  "
          f"1% {check.rpm.quantile(.01):.3f}  99% {check.rpm.quantile(.99):.3f}")
    print(f"    ratio of predicted spread to actual: {check.rpm.std() / train.rpm.std():.3f}")

    print("\n  equipment premium over Dry Van, percent")
    def premium(frame, column):
        med = frame.groupby("equipment")[column].median()
        return ((med / med["Dry Van"] - 1) * 100).round(2)
    print(pd.DataFrame({
        "train_actual": premium(train, "rpm"),
        "predicted": premium(check, "rpm"),
    }).to_string())

    print("\n  rate per mile by haul band, median")
    bands = [0, 400, 800, 1500, 99999]
    labels = ["short <400", "mid 400-800", "long 800-1500", "xlong 1500+"]
    print(pd.DataFrame({
        "train_actual": train.groupby(pd.cut(train.distance, bands, labels=labels),
                                      observed=True).rpm.median(),
        "predicted": check.groupby(pd.cut(check.distance, bands, labels=labels),
                                   observed=True).rpm.median(),
    }).round(4).to_string())

    print("\n  by month")
    print(check.groupby(check[C.DATE_COL].dt.month).agg(
        loads=("predicted_rate", "size"),
        mean_rate=("predicted_rate", "mean"),
        mean_rpm=("rpm", "mean"),
    ).round(3).to_string())

    print(f"\n  minimum predicted rate {check.predicted_rate.min():,.2f}  "
          f"(training minimum was {train[C.RAW_TARGET].min():,.2f})")
    print(f"  maximum predicted rate {check.predicted_rate.max():,.2f}  "
          f"(training maximum was {train[C.RAW_TARGET].max():,.2f})")
    below = (check.predicted_rate < train[C.RAW_TARGET].min()).sum()
    print(f"  predictions below the training minimum: {below}")
