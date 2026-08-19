"""Generate the freight dataset this project models.

Everything in `data/` comes from here. There is no external download and no
proprietary source: the generator below is the definition of the problem.

I wrote it to reproduce the structure that makes freight rate forecasting
interesting rather than to be realistic in absolute terms. Six properties matter,
and each one drives a decision elsewhere in the repo:

  1. Rate scales with distance, so predicting the raw rate mostly relearns
     multiplication. This is why src/model.py predicts rate per mile instead.
  2. Rate per mile decays with haul length. Short runs cost more per mile.
  3. A market index moves day to day, seasonally, with per load noise on top.
     Read row by row it looks weak; averaged per day it is the strongest signal
     in the set. src/features.py resamples it for exactly this reason.
  4. Lane history is thin and long tailed, so lane averages need a fallback.
  5. A small share of loads are inflated at random. Nobody can predict them, and
     they put a floor under the achievable error. This is the single most
     important property in the file.
  6. Weight arrives damaged: hard clipped, sometimes sign flipped, sometimes
     missing, and missing more often in the scoring window than in training.

Run:  python make_synthetic_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

SEED = 20250101
RNG = np.random.default_rng(SEED)

TRAIN_ROWS, VALID_ROWS = 48_000, 12_000
TRAIN_START, TRAIN_END = "2025-01-01", "2025-10-31"
VALID_START, VALID_END = "2025-11-01", "2025-12-31"

INFLATED_SHARE = 0.0069
WEIGHT_FLOOR, WEIGHT_CEILING = 5_000.0, 47_500.0

EQUIPMENT = {"Dry Van": (0.566, 1.000), "Reefer": (0.251, 1.146), "Flatbed": (0.183, 1.079)}

# A synthetic map. Coordinates are internally consistent so bearings and
# haversine distances mean something, but they are not real US positions.
CITIES = [
    "Atlanta", "Albany", "Albuquerque", "Allentown", "Amarillo", "Austin",
    "Bakersfield", "Baltimore", "Baton Rouge", "Birmingham", "Boston", "Buffalo",
    "Charleston", "Charlotte", "Chattanooga", "Chicago", "Cincinnati", "Columbia",
    "Corpus Christi", "Dallas", "Dayton", "Detroit", "El Paso", "Fort Wayne",
    "Fresno", "Grand Rapids", "Green Bay", "Harrisburg", "Hartford", "Houston",
    "Indianapolis", "Jackson", "Jacksonville", "Kansas City", "Knoxville", "Laredo",
    "Las Vegas", "Lexington", "Little Rock", "Los Angeles", "Louisville", "Lubbock",
    "Madison", "Memphis", "Milwaukee", "Mobile", "Montgomery", "Nashville",
    "New Orleans", "New York", "Norfolk", "Oklahoma City", "Philadelphia", "Phoenix",
    "Providence", "Raleigh", "Reno", "Richmond", "Salt Lake City", "San Antonio",
    "San Diego", "San Francisco", "Savannah", "Shreveport", "St. Louis", "Syracuse",
    "Tampa", "Toledo", "Tucson", "Tulsa", "Washington",
    "Akron", "Augusta", "Bismarck", "Boise", "Cedar Rapids", "Cheyenne",
    "Columbus", "Des Moines", "Duluth", "Erie", "Evansville", "Fargo",
    "Flagstaff", "Fort Smith", "Gary", "Greenville", "Huntsville", "Jackson MS",
    "Joplin", "Kalamazoo", "Lansing", "Lincoln", "Macon", "Medford",
    "Modesto", "Odessa", "Omaha", "Peoria", "Pensacola", "Pueblo",
    "Rapid City", "Roanoke", "Rockford", "Salina", "Scranton", "Sioux Falls",
    "Spokane", "Springfield", "Topeka", "Waco", "Wichita", "Wilmington",
    "Yakima", "Youngstown",
]


def build_map() -> pd.DataFrame:
    """Give every city a stable coordinate on the synthetic map."""
    rng = np.random.default_rng(SEED + 1)
    coords = pd.DataFrame(
        {"lat": rng.uniform(25.5, 44.5, len(CITIES)).round(5),
         "lon": rng.uniform(-121.7, -69.5, len(CITIES)).round(5)},
        index=CITIES,
    )
    # The December chart is a fixed 360 mile Dry Van run, so this lane has to
    # exist at roughly that length for the chart to mean anything.
    coords.loc["Lexington"] = [36.99152, -84.99876]
    coords.loc["Fort Wayne"] = [41.31561, -85.36206]
    return coords


def great_circle(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def market_series(dates: pd.DatetimeIndex) -> pd.Series:
    """A daily market level: seasonal swing, slow drift, and day to day wobble.

    Peaks in early summer and softens into the autumn, which is what gives the
    November and December window a different regime from most of training.
    """
    rng = np.random.default_rng(SEED + 2)
    doy = dates.dayofyear.to_numpy()
    seasonal = 0.18 * np.sin(2 * np.pi * (doy - 60) / 365.25)

    # Q4 peak season. Retail volume tightens capacity through late November and
    # into the week before Christmas, then the market falls off a cliff once
    # freight stops moving for the holidays.
    #
    # This is the only structure in December that a model can actually recover,
    # because the market index is handed to me for the scored window while the
    # calendar position is not. A December effect driven purely by the date would
    # be unlearnable: training stops on 31 October, so the model would never have
    # seen a December to learn it from.
    peak = 0.090 * np.exp(-((doy - 353) / 19.0) ** 2)
    collapse = -0.060 * (doy >= 360)

    wobble = pd.Series(rng.normal(0, 0.012, len(dates))).rolling(9, min_periods=1).mean()
    level = 1.06 + seasonal + peak + collapse + wobble.to_numpy()
    return pd.Series(level, index=dates).clip(0.82, 1.34)


def lane_matrix(coords: pd.DataFrame) -> np.ndarray:
    """A gravity model over city pairs.

    Freight is short haul heavy: a lane's volume falls off with distance rather
    than every pair being equally likely. Without this the average haul comes out
    around 1,400 miles and the rate distribution loses its short haul mass.
    """
    lat, lon = coords.lat.to_numpy(), coords.lon.to_numpy()
    d = great_circle(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    pull = np.exp(-d / 900.0) * _city_weights()[None, :]
    np.fill_diagonal(pull, 0.0)
    return pull / pull.sum(axis=1, keepdims=True)


def daily_shock(dates: pd.DatetimeIndex) -> pd.Series:
    """A per day wobble that the market index does not explain.

    Weather, a plant shutting for a week, a port backing up. Real markets move
    for reasons no published index captures, and this is what makes a random
    split dangerous: the same date appears in both halves, the model memorises
    that day's shock, and the score improves for a reason that will never
    generalise to a date it has not seen. Forward chaining cannot do that,
    because every test date lies beyond the training window.

    Deliberately not exposed as a column. If it were an input, there would be
    nothing to leak.
    """
    rng = np.random.default_rng(SEED + 7)
    raw = pd.Series(rng.normal(0, 0.045, len(dates)), index=dates)
    return raw.rolling(2, min_periods=1).mean()


def draw_loads(n: int, dates: pd.DatetimeIndex, coords: pd.DataFrame,
               lane_effect: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n loads spread across `dates`."""
    probs = lane_matrix(coords)
    idx = np.arange(len(CITIES))
    oi = rng.choice(idx, n, p=_city_weights())
    di = np.array([rng.choice(idx, p=probs[i]) for i in oi])
    origin = np.array(CITIES)[oi]
    dest = np.array(CITIES)[di]

    o, d = coords.loc[origin], coords.loc[dest]
    miles = great_circle(o.lat.to_numpy(), o.lon.to_numpy(),
                         d.lat.to_numpy(), d.lon.to_numpy())
    miles = np.maximum(miles * rng.normal(1.16, 0.05, n), 70.0).round(1)

    kinds = list(EQUIPMENT)
    equipment = rng.choice(kinds, n, p=[EQUIPMENT[k][0] for k in kinds])

    return pd.DataFrame({
        "pickup": origin, "delivery": dest,
        "pickup_lat": o.lat.to_numpy(), "pickup_lon": o.lon.to_numpy(),
        "delivery_lat": d.lat.to_numpy(), "delivery_lon": d.lon.to_numpy(),
        "distance": miles, "equipment": equipment,
        "date": rng.choice(dates, n),
        "_lane_mult": [lane_effect[(a, b)] for a, b in zip(origin, dest)],
    })


_WEIGHTS = None
def _city_weights() -> np.ndarray:
    """A long tailed volume distribution, so some lanes stay sparse."""
    global _WEIGHTS
    if _WEIGHTS is None:
        w = np.random.default_rng(SEED + 3).lognormal(0, 0.75, len(CITIES))
        _WEIGHTS = w / w.sum()
    return _WEIGHTS


def price(frame: pd.DataFrame, market: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """Turn load features into a posted rate."""
    miles = frame["distance"].to_numpy()
    doy = frame["date"].dt.dayofyear.to_numpy()

    base = 2.10 * (miles / 500.0) ** -0.115          # short hauls cost more per mile
    equip = np.array([EQUIPMENT[k][1] for k in frame["equipment"]])
    season = 1.0 + 0.008 * np.sin(2 * np.pi * (doy - 55) / 365.25)
    mkt = 1.0 + 0.85 * (frame["date"].map(market).to_numpy() - 1.06)
    dow = 1.0 + 0.004 * np.sin(2 * np.pi * frame["date"].dt.dayofweek.to_numpy() / 7.0)

    shock = frame["date"].map(daily_shock(pd.DatetimeIndex(sorted(frame["date"].unique()))))
    rpm = base * equip * season * mkt * dow * (1.0 + shock.to_numpy()) * frame["_lane_mult"].to_numpy()
    rpm *= rng.lognormal(0, 0.040, len(frame))

    out = frame.drop(columns=["_lane_mult"]).copy()
    out["posted_rate"] = (rpm * miles).round(2)
    return out


def damage(frame: pd.DataFrame, missing_market: float, missing_weight: float,
           rng: np.random.Generator) -> pd.DataFrame:
    """Apply the data quality problems the modelling code has to survive."""
    n = len(frame)
    out = frame.copy()

    weight = rng.uniform(WEIGHT_FLOOR, WEIGHT_CEILING + 1100, n)
    out["weight"] = np.clip(weight, WEIGHT_FLOOR, WEIGHT_CEILING).round(0)

    flip = rng.random(n) < 0.006                      # sign flipped on write
    out.loc[flip, "weight"] = -out.loc[flip, "weight"]
    out.loc[rng.random(n) < missing_weight, "weight"] = np.nan

    # Row level market index: the daily level plus per load reporting noise. The
    # noise is deliberately larger than the daily movement, which is what makes
    # the column look weak until it is aggregated.
    out.loc[rng.random(n) < missing_market, "market_index"] = np.nan

    # A second numeric column with almost no relationship to the target. It is
    # here so that feature selection has something real to reject.
    out["quote_signal"] = rng.normal(2.05, 0.22, n).round(5)
    return out


def main() -> None:
    coords = build_map()
    lane_rng = np.random.default_rng(SEED + 4)
    lane_effect = {(a, b): lane_rng.lognormal(0, 0.150)
                   for a in CITIES for b in CITIES if a != b}

    train_dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    valid_dates = pd.date_range(VALID_START, VALID_END, freq="D")
    market = market_series(train_dates.union(valid_dates))

    rng = np.random.default_rng(SEED + 5)
    train = price(draw_loads(TRAIN_ROWS, train_dates, coords, lane_effect, rng), market, rng)
    valid = price(draw_loads(VALID_ROWS, valid_dates, coords, lane_effect, rng), market, rng)

    # Inflate a small random share of training loads. These are unpredictable by
    # construction, and measuring the floor they impose is the point.
    hit = rng.random(len(train)) < INFLATED_SHARE
    train.loc[hit, "posted_rate"] = (train.loc[hit, "posted_rate"]
                                     * rng.uniform(2.2, 3.8, hit.sum())).round(2)

    for frame in (train, valid):
        frame["market_index"] = (frame["date"].map(market)
                                 * np.random.default_rng(SEED + 6).normal(
                                     1, 0.035, len(frame))).round(5)

    train = damage(train, 0.0078, 0.0062, rng)
    valid = damage(valid, 0.0208, 0.0138, rng)          # ~2.5x more missing at score time

    train = train.sort_values("date").reset_index(drop=True)
    valid = valid.sort_values("date").reset_index(drop=True)
    train.insert(0, "load_id", [f"TR-{i:06d}" for i in range(1, len(train) + 1)])
    valid.insert(0, "load_id", [f"TE-{i:06d}" for i in range(1, len(valid) + 1)])

    order = ["load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
             "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
             "date", "market_index", "quote_signal"]
    train = train[order + ["posted_rate"]]
    valid = valid[order]

    C.DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(C.TRAIN_CSV, index=False)
    valid.drop(columns=[]).to_csv(C.VALIDATION_CSV, index=False)
    valid[["load_id"]].assign(predicted_rate="").to_csv(C.TEMPLATE_CSV, index=False)

    pd.DataFrame({
        "pickup": C.DECEMBER_PICKUP, "delivery": C.DECEMBER_DELIVERY,
        "distance": int(C.DECEMBER_DISTANCE), "equipment": C.DECEMBER_EQUIPMENT,
        "weight": int(C.DECEMBER_WEIGHT),
        "date": pd.date_range(C.DECEMBER_START, C.DECEMBER_END).strftime("%Y-%m-%d"),
        "predicted_rate": "",
    }).to_csv(C.DECEMBER_CSV, index=False)

    rpm = train.posted_rate / train.distance
    print(f"  train      {len(train):,} rows  {train.date.min().date()} to {train.date.max().date()}")
    print(f"  validation {len(valid):,} rows  {valid.date.min().date()} to {valid.date.max().date()}")
    print(f"  rate       mean {train.posted_rate.mean():,.2f}  median {train.posted_rate.median():,.2f}")
    print(f"  rate/mile  mean {rpm.mean():.3f}  sd {rpm.std():.3f}")
    print(f"  CV         raw {train.posted_rate.std()/train.posted_rate.mean():.3f}"
          f"  vs per-mile {rpm.std()/rpm.mean():.3f}")
    print(f"  distance correlation with rate  {train.distance.corr(train.posted_rate):.3f}")
    print(f"  inflated   {hit.sum()} ({hit.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
