"""Feature engineering.

The rule I had to work around
-----------------------------
Every feature has to work for all three sets of rows I touch: the 48,000 training
loads, the 12,000 I am scored on, and the 31 rows behind the December chart.

That last one is the awkward one. `december_chart_inputs.csv` carries seven
columns and no coordinates, no market index and no quote signal, because the
probe is meant to vary the date and nothing else. So `complete_december` runs
first and fills those columns from data I already hold: coordinates from the city
lookup in training, and the market columns from `validation.csv`, which covers
all of December.

What is allowed to see everything, and what is not
--------------------------------------------------
The market index is an input I am handed for every row I have to score, so
averaging it across training and scoring rows together is fine. It matches how
this works in real life, where a broker knows today's market when quoting today's
load. No answer column ever gets that freedom.

Anything built from the posted rate, which here means the lane encoder, uses
earlier dates only and gets rebuilt from scratch inside every fold. This is the
easiest place in the whole project to leak the future backwards without noticing.

Things I decided against
------------------------
Holiday features, because all four holidays in training land within 0.7% of the
weeks either side of them once the season is taken out. A comparison between
straight line distance and the distance column, because the two correlate 0.9995
so there was nothing to gain. And an interaction between equipment and trip
length, because the premium turns out to be flat across every distance band.

I also left log distance out of the tree features on purpose. A tree splits on
thresholds, and asking whether log distance beats log 800 is the same question as
asking whether distance beats 800, so logging the column tells it nothing. I keep
it for the linear model, which does see a difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --- December completion ---------------------------------------------------


def city_coordinates(train: pd.DataFrame) -> pd.DataFrame:
    """One coordinate pair per city.

    I checked that every city has exactly one pair, and that a city used as an
    origin has the identical coordinate to the same city used as a destination,
    down to six decimal places. So either one works as the lookup.
    """
    pickup = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    pickup.columns = ["lat", "lon"]
    delivery = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    delivery.columns = ["lat", "lon"]
    return pd.concat([pickup, delivery]).groupby(level=0).first()


def complete_december(
    december: pd.DataFrame,
    validation: pd.DataFrame,
    train: pd.DataFrame,
) -> pd.DataFrame:
    """Fill in the columns the chart file leaves out, so it can use the same code.

    Coordinates come from the city lookup in training. The market index and quote
    signal come from the daily average across validation rows on the same date,
    which is a value I already hold rather than one I invented.
    """
    coords = city_coordinates(train)
    frame = december.copy()

    frame["pickup_lat"] = frame.pickup.map(coords.lat)
    frame["pickup_lon"] = frame.pickup.map(coords.lon)
    frame["delivery_lat"] = frame.delivery.map(coords.lat)
    frame["delivery_lon"] = frame.delivery.map(coords.lon)

    daily = validation.groupby(C.DATE_COL)[["market_index", "quote_signal"]].mean()
    frame["market_index"] = frame[C.DATE_COL].map(daily.market_index)
    frame["quote_signal"] = frame[C.DATE_COL].map(daily.quote_signal)

    missing = frame[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].isna()
    if missing.any().any():
        raise ValueError(f"December completion left gaps: {missing.sum().to_dict()}")
    return frame


# --- Market levels ---------------------------------------------------------


def market_levels(*frames: pd.DataFrame) -> pd.DataFrame:
    """The daily market level and its seven day average.

    Row by row the market index correlates 0.084 with rate per mile, weak enough
    that I nearly dropped it. Averaged per day it correlates 0.637, because day to
    day movement is 6.7 times bigger than the movement between loads on the same
    day. The column was never weak, I was just reading it at the wrong resolution.
    """
    stacked = pd.concat([f[[C.DATE_COL, "market_index"]] for f in frames], ignore_index=True)
    daily = stacked.groupby(C.DATE_COL).market_index.mean().sort_index()
    roll7 = daily.rolling(7, min_periods=1).mean()
    roll28 = daily.rolling(28, min_periods=1).mean()
    return pd.DataFrame({
        "market_daily": daily,
        "market_roll7": roll7,
        "market_roll28": roll28,
        # The raw market level drifts against what I am predicting. At the same
        # market level, median rate per mile is 2.015 in the first half of the year
        # and 2.140 in the second, so a model trained on early data misreads later
        # data. Dividing the daily level by its own trailing month was my attempt
        # to remove that drift. It scored worse, so I did not ship it, but I left
        # it here because the report explains why it failed.
        "market_ratio": daily / roll28,
        "market_ratio7": roll7 / roll28,
    })


# --- Geography -------------------------------------------------------------


def bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Which way the load is travelling, in degrees.

    The coordinates are a made up map rather than the real United States, but a
    perfectly consistent one, so direction still means something even though the
    places are in the wrong spots.
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# --- Lane encoding ---------------------------------------------------------


class LaneEncoder:
    """Average rate per lane, falling back to the cities when history is thin.

    Two things forced this design. 12.17% of the loads I am scored on run on a lane
    that never appears in training, and eight cities only turn up at prediction
    time, so I need a fallback and it fires often. And lane history is thin anyway,
    with a median of 10 loads and 537 lanes under five, so a plain lane average is
    mostly noise.

    So each level leans on the one above it, weighted by how much history actually
    exists. A lane blends toward the average of its two cities, and each city
    blends toward the overall median. With smoothing at 20, a lane with 10 loads
    ends up one third its own history and two thirds its cities.

    On leakage. Training rows are encoded using only loads dated strictly earlier,
    so a row can never feed into its own feature and no future rate flows
    backwards. Scoring rows sit entirely after the fitting window, so they use the
    full statistics.

    I clip rate per mile at 5 dollars before working out any average, because that
    is where the inflated loads start, and one of them would otherwise wreck a thin
    lane.
    """

    smoothing: float = 20.0
    clip: float = 5.0

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    @staticmethod
    def _lane(frame: pd.DataFrame) -> pd.Series:
        return frame.pickup.astype(str) + " > " + frame.delivery.astype(str)

    def _rpm(self, frame: pd.DataFrame) -> pd.Series:
        return (frame[C.RAW_TARGET] / frame["distance"]).clip(upper=self.clip)

    def fit(self, train: pd.DataFrame) -> "LaneEncoder":
        work = train[[C.DATE_COL, "pickup", "delivery", "distance", C.RAW_TARGET]].copy()
        work["lane"] = self._lane(work)
        work["rpm"] = self._rpm(work)

        self.global_ = float(work.rpm.median())
        self.stats_ = {
            key: work.groupby(key).rpm.agg(["mean", "size"])
            for key in ("pickup", "delivery", "lane")
        }
        self._train_index = train.index
        self._expanding = self._expanding_encoding(work)
        return self

    def _shrink(self, mean: pd.Series, count: pd.Series, parent) -> pd.Series:
        weight = count / (count + self.smoothing)
        return weight * mean.fillna(0) + (1 - weight) * parent

    def _expanding_encoding(self, work: pd.DataFrame) -> pd.DataFrame:
        """Encode each row using only loads dated strictly before it."""
        out = {}
        for key in ("pickup", "delivery", "lane"):
            daily = (
                work.groupby([key, C.DATE_COL], observed=True)
                .rpm.agg(total="sum", n="size")
                .reset_index()
                .sort_values([key, C.DATE_COL])
            )
            grouped = daily.groupby(key, observed=True)
            # running total up to and including this date, minus this date itself
            daily["prior_total"] = grouped.total.cumsum() - daily.total
            daily["prior_n"] = grouped.n.cumsum() - daily.n

            merged = work[[key, C.DATE_COL]].merge(
                daily[[key, C.DATE_COL, "prior_total", "prior_n"]],
                on=[key, C.DATE_COL], how="left",
            )
            count = merged.prior_n.fillna(0).to_numpy()
            total = merged.prior_total.fillna(0).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
            out[key] = (pd.Series(mean, index=work.index), pd.Series(count, index=work.index))

        pickup_enc = self._shrink(out["pickup"][0], out["pickup"][1], self.global_)
        delivery_enc = self._shrink(out["delivery"][0], out["delivery"][1], self.global_)
        parent = (pickup_enc + delivery_enc) / 2
        lane_enc = self._shrink(out["lane"][0], out["lane"][1], parent)

        return pd.DataFrame({
            "lane_rpm": lane_enc,
            "pickup_rpm": pickup_enc,
            "delivery_rpm": delivery_enc,
            "lane_history": out["lane"][1],
        }, index=work.index)

    def transform_train(self) -> pd.DataFrame:
        return self._expanding

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        lane = self._lane(frame)
        pieces = {}
        for key, values in (("pickup", frame.pickup), ("delivery", frame.delivery), ("lane", lane)):
            stats = self.stats_[key]
            pieces[key] = (
                values.map(stats["mean"]).astype(float),
                values.map(stats["size"]).fillna(0).astype(float),
            )

        pickup_enc = self._shrink(*pieces["pickup"], self.global_)
        delivery_enc = self._shrink(*pieces["delivery"], self.global_)
        parent = (pickup_enc + delivery_enc) / 2
        lane_enc = self._shrink(*pieces["lane"], parent)

        return pd.DataFrame({
            "lane_rpm": lane_enc,
            "pickup_rpm": pickup_enc,
            "delivery_rpm": delivery_enc,
            "lane_history": pieces["lane"][1],
        }, index=frame.index)


# --- Feature groups --------------------------------------------------------

def build_matrix(
    frame: pd.DataFrame,
    groups: tuple[str, ...],
    market: pd.DataFrame,
    lane: pd.DataFrame | None = None,
    add_log_distance: bool = False,
) -> pd.DataFrame:
    """Put together the feature table for whichever groups I asked for."""
    raw_weight = frame["weight"]
    out = pd.DataFrame(index=frame.index)

    if "base" in groups:
        out["distance"] = frame["distance"]
        out["weight"] = raw_weight
        out["weight_missing"] = raw_weight.isna().astype(int)
        out["at_weight_cap"] = (raw_weight == C.WEIGHT_CAP).astype(int)
        out["weight_per_mile"] = raw_weight / frame["distance"]
        if add_log_distance:
            out["log_distance"] = np.log(frame["distance"])

    if "equipment" in groups:
        out["equipment"] = frame["equipment"].astype("category")

    if "market" in groups:
        out["market_index"] = frame["market_index"]
        out["market_missing"] = frame["market_index"].isna().astype(int)
        out["market_daily"] = frame[C.DATE_COL].map(market.market_daily)
        out["market_roll7"] = frame[C.DATE_COL].map(market.market_roll7)

    if "market_relative" in groups:
        out["market_missing"] = frame["market_index"].isna().astype(int)
        out["market_ratio"] = frame[C.DATE_COL].map(market.market_ratio)
        out["market_ratio7"] = frame[C.DATE_COL].map(market.market_ratio7)

    if "time" in groups:
        out["day_of_week"] = frame[C.DATE_COL].dt.dayofweek

    if "time_position" in groups:
        # I expected these to fail. November and December never appear in training,
        # so a tree can only clamp them to the last value it saw. Fold three tests
        # exactly that, scoring two months it never trained on. They improved every
        # fold, so I was wrong and I kept them.
        out["month"] = frame[C.DATE_COL].dt.month
        out["day_of_year"] = frame[C.DATE_COL].dt.dayofyear
        out["days_elapsed"] = (frame[C.DATE_COL] - pd.Timestamp("2025-01-01")).dt.days

    if "seasonal" in groups:
        # Fourier terms on day of year.
        #
        # This group exists because of a failure mode in `time_position`. Month,
        # day of year and days elapsed all take values in November and December
        # that lie beyond anything in training, so a tree can only clamp them to
        # the last threshold it learned. They stop varying exactly when I need
        # them most, which is why the December chart came out flat.
        #
        # Sine and cosine are bounded and periodic, so December values land back
        # inside the range the model already saw in the spring. The tree can keep
        # splitting on them across the whole scored window. Three harmonics is
        # enough for an annual cycle with a summer peak and shoulder seasons.
        doy = frame[C.DATE_COL].dt.dayofyear.to_numpy()
        for k in (1, 2, 3):
            out[f"season_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            out[f"season_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    if "lane_native" in groups:
        out["pickup"] = frame["pickup"].astype("category")
        out["delivery"] = frame["delivery"].astype("category")

    if "lane_encoded" in groups and lane is not None:
        for column in lane.columns:
            out[column] = lane[column]

    if "geo" in groups:
        out["bearing"] = bearing(frame.pickup_lat, frame.pickup_lon,
                                 frame.delivery_lat, frame.delivery_lon)

    if "quote" in groups:
        out["quote_signal"] = frame["quote_signal"]

    return out


def align_categories(train_matrix: pd.DataFrame, test_matrix: pd.DataFrame) -> pd.DataFrame:
    """Give the test table the same category levels as the training table.

    Eight cities only turn up at prediction time. Handing them to the model as an
    unknown level would error, so instead they stay as an unseen category and the
    model sends them down its default branch.
    """
    aligned = test_matrix.copy()
    for column in train_matrix.columns:
        if isinstance(train_matrix[column].dtype, pd.CategoricalDtype):
            aligned[column] = pd.Categorical(
                aligned[column], categories=train_matrix[column].cat.categories
            )
    return aligned[train_matrix.columns]
