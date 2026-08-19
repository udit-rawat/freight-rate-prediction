"""Loading the data, checking it, and repairing what is broken in it.

This module only gets trustworthy frames into memory and fixes damage in the
source files. Anything I derive, so indicators, encodings and time terms, lives
in features.py instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --- Loading ---------------------------------------------------------------


def load_train() -> pd.DataFrame:
    """48,000 loads with answers, 1 January to 31 October 2025."""
    return pd.read_csv(C.TRAIN_CSV, parse_dates=[C.DATE_COL])


def load_validation() -> pd.DataFrame:
    """12,000 loads without answers, 1 November to 31 December 2025."""
    return pd.read_csv(C.VALIDATION_CSV, parse_dates=[C.DATE_COL])


def load_december() -> pd.DataFrame:
    """The 31 chart rows, with everything frozen except the date."""
    return pd.read_csv(C.DECEMBER_CSV, parse_dates=[C.DATE_COL])


def load_template() -> pd.DataFrame:
    """The submission template, load_id plus an empty rate column."""
    return pd.read_csv(C.TEMPLATE_CSV)


# --- Repair ----------------------------------------------------------------


def repair_weight(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix the weights that came through with a negative sign.

    292 training rows and 145 validation rows have a negative weight. My reflex
    was to drop them, and I would have if the damage reached the target. It does
    not. The absolute values cover 5,000 to 47,500, exactly the range of the clean
    rows, the medians match at 31,822 against 31,494, and rate per mile is the
    same in both groups at 2.157 against 2.145. So the problem is confined to this
    one column, and taking the absolute value gives me back the real number rather
    than inventing one.

    Returns a copy, so the notebook can still show the before and after.
    """
    repaired = frame.copy()
    repaired["weight"] = repaired["weight"].abs()
    return repaired


def flag_inflated(frame: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
    """Mark the loads that look like they have been inflated on purpose.

    I compare each load against the median rate for its own distance band, which
    strips out the short haul effect and leaves only real excess. The ratio jumps
    from 1.40 at the 99th percentile to 2.86 at the 99.5th, and real distributions
    do not jump between neighbouring percentiles like that, so 2.0 sits safely in
    the gap.

    These 0.69% of loads are spread evenly across dates, lanes and equipment
    types. Expedited freight would cluster on short runs and a capacity crunch
    would cluster on dates, so this looks like rows picked at random and
    multiplied.

    I only use this for reporting. It reads the answer column, so it must never
    become a feature, and I never drop these rows from a test window either. Doing
    that would flatter my score by deleting the loads nobody can predict.
    """
    bands = pd.cut(frame["distance"], [0, 200, 400, 700, 1000, 1500, 2500, np.inf])
    band_median = frame.groupby(bands, observed=True)[C.RAW_TARGET].transform("median")
    return frame[C.RAW_TARGET] / band_median > threshold


# --- Integrity -------------------------------------------------------------


def verify_integrity(raise_on_failure: bool = True) -> list[tuple[str, bool, str]]:
    """Check the generated files have the shape the rest of the code assumes.

    Everything in data/ comes out of make_synthetic_data.py, so these assertions
    are really a contract between the generator and the modelling code. If I
    change a distribution in one and forget the other, this fails loudly instead
    of silently producing a worse model. Returns label, passed, detail.
    """
    train = load_train()
    validation = load_validation()
    december = load_december()
    template = load_template()

    expected_ids = {f"TE-{index:06d}" for index in range(1, 12_001)}
    results: list[tuple[str, bool, str]] = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        results.append((label, bool(passed), detail))

    # Shapes
    check("train shape", train.shape == C.EXPECTED_TRAIN_SHAPE, str(train.shape))
    check("validation shape", validation.shape == C.EXPECTED_VALIDATION_SHAPE, str(validation.shape))
    check("december rows", len(december) == C.EXPECTED_DECEMBER_ROWS, str(len(december)))
    check("template rows", len(template) == 12_000, str(len(template)))

    # Columns
    check(
        "validation == train minus target",
        list(validation.columns) == [c for c in train.columns if c != C.RAW_TARGET],
    )
    check(
        "december column order",
        list(december.columns)
        == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"],
    )

    # The dates, which is the fact my whole validation design rests on
    check("no train/validation date overlap", train[C.DATE_COL].max() < validation[C.DATE_COL].min())
    check(
        "train date range",
        (str(train[C.DATE_COL].min().date()), str(train[C.DATE_COL].max().date()))
        == C.TRAIN_DATE_RANGE,
    )
    check(
        "validation date range",
        (str(validation[C.DATE_COL].min().date()), str(validation[C.DATE_COL].max().date()))
        == C.VALIDATION_DATE_RANGE,
    )
    check("train has 304 distinct days", train[C.DATE_COL].nunique() == 304)
    check("validation has 61 distinct days", validation[C.DATE_COL].nunique() == 61)

    # Identifiers
    check("train load_id unique", not train[C.ID_COL].duplicated().any())
    check("validation load_id unique", not validation[C.ID_COL].duplicated().any())
    check("validation ids are TE-000001..TE-012000", set(validation[C.ID_COL]) == expected_ids)
    check("template ids match validation", set(template[C.ID_COL]) == expected_ids)

    # The December chart rows hold every value score.py checks
    check("december pickup", december["pickup"].eq(C.DECEMBER_PICKUP).all())
    check("december delivery", december["delivery"].eq(C.DECEMBER_DELIVERY).all())
    check("december distance", december["distance"].eq(C.DECEMBER_DISTANCE).all())
    check("december equipment", december["equipment"].eq(C.DECEMBER_EQUIPMENT).all())
    check("december weight", december["weight"].eq(C.DECEMBER_WEIGHT).all())
    check(
        "december covers 2025-12-01..2025-12-31",
        december[C.DATE_COL].nunique() == 31
        and str(december[C.DATE_COL].min().date()) == C.DECEMBER_START
        and str(december[C.DATE_COL].max().date()) == C.DECEMBER_END,
    )

    failures = [label for label, passed, _ in results if not passed]
    if failures and raise_on_failure:
        raise AssertionError("Data integrity check failed: " + "; ".join(failures))
    return results


if __name__ == "__main__":
    for label, passed, detail in verify_integrity(raise_on_failure=False):
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {label}" + (f"  {detail}" if detail else ""))
