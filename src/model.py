"""The models I tried.

All four scored on the same harness in split.py:

    median rate per mile times distance   the baseline, the number to beat
    ridge regression                      is this problem even non linear?
    histogram gradient boosting           what I shipped
    LightGBM                              the challenger, finished 33 cents ahead

I need the last two to handle missing values natively rather than as a nicety.
There is about 2.5 times more missing data at prediction time than in training, so
I cannot drop rows, and which values are missing turns out to be worth knowing.

Every model predicts rate per mile and multiplies back by distance, because
distance alone correlates 0.909 with the rate and predicting it directly just
spends the model's effort relearning multiplication.

On the loss. I use absolute error rather than squared error. About 0.69% of loads
have been inflated at random, nobody can predict them, and squaring would let
those few rows drag the whole fit toward them.

How a model plugs in
--------------------
Every model is a function `fit_predict(train, test_features)` that returns a
predicted rate. `test_features` never contains the answer column, so a model
cannot read it even by accident. Any encoder gets fitted inside that function,
which is what keeps each fold sealed off from the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import features as F


def _rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return frame[C.RAW_TARGET] / frame["distance"]


# --- Rung 0 ----------------------------------------------------------------


def null_model(train: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
    """One number for the whole market, median rate per mile times distance.

    This exists to be beaten. I also used it to check the harness itself, because
    I know in advance roughly how it should behave, so a strange score here would
    mean my harness was broken rather than that I had found something.
    """
    median_rpm = _rate_per_mile(train).median()
    return median_rpm * test_features["distance"].to_numpy()


# --- Leakage demonstration -------------------------------------------------


def date_lookup_model(train: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
    """Look up the median rate per mile on the load's own date.

    Not a real candidate. I wrote it to price what a random split costs. With
    dated folds every test date sits beyond the training window, so the lookup
    always misses and falls back to the overall median. With a random split the
    same dates sit in both halves and it scores better for no real reason.

    The gap between those two numbers is what the leak is worth, and it shows the
    actual danger. A random split cannot tell a model that generalises apart from
    one that has simply memorised dates, and it will happily recommend the latter.
    """
    by_date = _rate_per_mile(train).groupby(train[C.DATE_COL]).median()
    fallback = _rate_per_mile(train).median()
    looked_up = test_features[C.DATE_COL].map(by_date).fillna(fallback)
    return looked_up.to_numpy() * test_features["distance"].to_numpy()


# --- Feature backed rungs --------------------------------------------------


def _prepare(train, test_features, groups, add_log_distance=False, market_frames=None):
    """Build the train and test tables, fitting every encoder inside the fold.

    `market_frames` adds extra frames to the daily market series only. I need it
    for the December chart, whose 31 rows would otherwise leave a November sized
    hole between the end of training and the chart dates, which would wreck the
    rolling averages. Passing validation closes the gap. I only read the market
    index out of these frames, never an answer column.
    """
    market = F.market_levels(train, test_features, *(market_frames or []))

    lane_train = lane_test = None
    if "lane_encoded" in groups:
        encoder = F.LaneEncoder().fit(train)
        lane_train = encoder.transform_train()
        lane_test = encoder.transform(test_features)

    x_train = F.build_matrix(train, groups, market, lane_train, add_log_distance)
    x_test = F.build_matrix(test_features, groups, market, lane_test, add_log_distance)
    return x_train, F.align_categories(x_train, x_test)


def recency_weights(train: pd.DataFrame, halflife_days: float | None) -> np.ndarray | None:
    """Weight recent loads more heavily, halving every `halflife_days`.

    I tried this because of a drift I had already measured, not as a generic
    trick. At the same market index, median rate per mile is 2.015 in the first
    half of the year and 2.140 in the second. Training treats January the same as
    October, and yet November and December sit right next to October.

    It made no difference at 60, 120 or 240 days, so I did not ship it. The risk
    was always that it throws away real signal, since the early year is half my
    data.
    """
    if halflife_days is None:
        return None
    age = (train[C.DATE_COL].max() - train[C.DATE_COL]).dt.days.to_numpy()
    return 0.5 ** (age / halflife_days)


def _hist_estimator(loss: str, kwargs: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        max_iter=kwargs.get("max_iter", 300),
        learning_rate=kwargs.get("learning_rate", 0.06),
        max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
        min_samples_leaf=kwargs.get("min_samples_leaf", 40),
        l2_regularization=kwargs.get("l2_regularization", 1.0),
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=C.RANDOM_SEED,
    )


def make_gbdt(groups: tuple[str, ...], loss: str = "absolute_error",
              halflife_days: float | None = None, market_frames=None, **kwargs):
    """Histogram gradient boosting, which is what I ended up shipping."""

    def fit_predict(train: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
        x_train, x_test = _prepare(train, test_features, groups, market_frames=market_frames)
        estimator = _hist_estimator(loss, kwargs)
        estimator.fit(x_train, _rate_per_mile(train),
                      sample_weight=recency_weights(train, halflife_days))
        return estimator.predict(x_test) * test_features["distance"].to_numpy()

    return fit_predict


def make_lgbm(groups: tuple[str, ...], objective: str = "l1",
              halflife_days: float | None = None, **kwargs):
    """LightGBM, the challenger.

    I ran this as a real alternative rather than a box to tick. It finished 33
    cents ahead, and my scores move by about 11 dollars from fold to fold, so that
    is a tie. I shipped the simpler option instead of taking on a dependency for
    noise.
    """
    import lightgbm as lgb

    def fit_predict(train: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
        x_train, x_test = _prepare(train, test_features, groups)
        estimator = lgb.LGBMRegressor(
            objective=objective,
            n_estimators=kwargs.get("n_estimators", 400),
            learning_rate=kwargs.get("learning_rate", 0.06),
            num_leaves=kwargs.get("num_leaves", 31),
            min_child_samples=kwargs.get("min_child_samples", 40),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            random_state=C.RANDOM_SEED,
            verbose=-1,
        )
        estimator.fit(x_train, _rate_per_mile(train),
                      sample_weight=recency_weights(train, halflife_days))
        return estimator.predict(x_test) * test_features["distance"].to_numpy()

    return fit_predict


def fit_for_inspection(train: pd.DataFrame, test_features: pd.DataFrame,
                       groups: tuple[str, ...] = None, **kwargs):
    """Fit once and hand back the pieces, so I can shuffle features afterwards.

    The normal `fit_predict` function keeps its tables to itself, which is right
    for the harness and no use when I want to look inside the model.
    """
    groups = groups or SELECTED
    x_train, x_test = _prepare(train, test_features, groups)
    estimator = _hist_estimator(kwargs.pop("loss", "absolute_error"), kwargs)
    estimator.fit(x_train, _rate_per_mile(train),
                  sample_weight=recency_weights(train, kwargs.get("halflife_days")))
    return estimator, x_train, x_test


def make_ridge(groups: tuple[str, ...]):
    """Ridge regression, to find out whether this problem is even non linear.

    I include log distance here and leave it out of the tree models. A tree splits
    on thresholds, so asking whether log distance beats log 800 is the same
    question as asking whether distance beats 800, and the transform is invisible
    to it. A linear model genuinely sees a different shape.
    """

    def fit_predict(train: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
        x_train, x_test = _prepare(train, test_features, groups, add_log_distance=True)
        categorical = [c for c in x_train.columns
                       if isinstance(x_train[c].dtype, pd.CategoricalDtype)]
        numeric = [c for c in x_train.columns if c not in categorical]

        pipeline = Pipeline([
            ("prep", ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ])),
            ("model", Ridge(alpha=1.0, random_state=C.RANDOM_SEED)),
        ])
        pipeline.fit(x_train, _rate_per_mile(train))
        return pipeline.predict(x_test) * test_features["distance"].to_numpy()

    return fit_predict


# --- Registries ------------------------------------------------------------

BASELINES = {
    "null (median rpm)": null_model,
    "date lookup (leak probe)": date_lookup_model,
}

# I added one group at a time, so the gap between each line and the one above it
# is what that group was actually worth.
ABLATION: dict[str, tuple[str, ...]] = {
    "base": ("base",),
    "+ equipment": ("base", "equipment"),
    "+ market": ("base", "equipment", "market"),
    "+ day of week": ("base", "equipment", "market", "time"),
    "+ lane native": ("base", "equipment", "market", "time", "lane_native"),
    "+ month and trend": ("base", "equipment", "market", "time", "lane_native", "time_position"),
    "+ fourier season": ("base", "equipment", "market", "time", "lane_native", "time_position",
                         "seasonal"),
}

# What I ended up shipping.
#
# I expected `time_position` to fail. November and December never appear in
# training, so month and days elapsed sit outside anything the model has seen, and
# a tree can only clamp them to the last value it knows. Fold three tests exactly
# that, training through August and scoring the next 61 days, the same distance
# ahead as the real task. It improved every fold, so clamping to late October
# turns out to be better than having no correction at all, and it is what makes up
# for the market drift. At the same market index, median rate per mile is 2.015 in
# the first half of the year and 2.140 in the second.
SELECTED_LABEL = "+ fourier season"
SELECTED = ABLATION[SELECTED_LABEL]

# Things I tried and threw out. Each is what I shipped plus one extra group, so
# the gap against the shipped row is what that group cost me. I left them in the
# code because the report should show what I tried, not only what survived.
REJECTED: dict[str, tuple[str, ...]] = {
    # The expensive one. I built LaneEncoder as a smoothed hierarchy, lane falling
    # back to its two cities and then to the global median, and it costs $34. The
    # median lane carries three loads, so a lane average is mostly noise, and the
    # shrinkage that should rescue it also strips out most of what made it worth
    # having. There is a second problem I only found by measuring: training rows
    # are encoded from earlier dates only, while scoring rows get the full
    # statistics, so the two see systematically different feature strengths.
    # Native categorical splits on pickup and delivery do the same job without
    # either failure.
    "rejected: lane target encoding": SELECTED + ("lane_encoded",),
    "rejected: geography": SELECTED + ("geo",),
    "rejected: quote_signal": SELECTED + ("quote",),
    "rejected: market as ratio": ("base", "equipment", "time", "lane_native",
                                  "time_position", "market_relative"),
}
