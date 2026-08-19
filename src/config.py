"""Paths, column groups and fixed constants.

I keep every path and magic number here so there is one place to look, rather
than path strings scattered through the other modules.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "train_test.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
DECEMBER_CSV = DATA_DIR / "december_chart_inputs.csv"
TEMPLATE_CSV = DATA_DIR / "validation_predictions_template.csv"

# The 12,000 scored predictions.
PREDICTIONS_CSV = ROOT / "validation_predictions.csv"

RESULTS_DIR = ROOT / "results"
HARNESS_SCORES_CSV = RESULTS_DIR / "harness_scores.csv"

# --- Reproducibility -------------------------------------------------------

RANDOM_SEED = 42

# --- Target ----------------------------------------------------------------

# The label as given to me. I do not predict this directly. I predict rate per
# mile and multiply back by distance, because distance alone correlates 0.909
# with the rate and predicting it straight just relearns multiplication.
RAW_TARGET = "posted_rate"

# --- Columns ---------------------------------------------------------------

ID_COL = "load_id"
DATE_COL = "date"

# --- Data quality constants I measured -------------------------------------

# Weight is hard capped at 5,000 and 47,500, and a couple of percent of rows sit
# exactly on the ceiling. A value pinned to a bound is a censored measurement
# rather than a real one, so those rows get flagged instead of trusted.
WEIGHT_CAP = 47_500.0

# --- The December probe ----------------------------------------------------
#
# 31 rows with everything frozen except the date, so whatever shape comes out is
# only what the model learned about time. If a model ignores the calendar this
# comes out as a flat line, which makes it a cheap diagnostic.

DECEMBER_PICKUP = "Lexington"
DECEMBER_DELIVERY = "Fort Wayne"
DECEMBER_DISTANCE = 360.0
DECEMBER_EQUIPMENT = "Dry Van"
DECEMBER_WEIGHT = 32_000.0
DECEMBER_START = "2025-12-01"
DECEMBER_END = "2025-12-31"

# --- Expected shapes (integrity gate) --------------------------------------

EXPECTED_TRAIN_SHAPE = (48_000, 14)
EXPECTED_VALIDATION_SHAPE = (12_000, 13)
EXPECTED_DECEMBER_ROWS = 31

TRAIN_DATE_RANGE = ("2025-01-01", "2025-10-31")
VALIDATION_DATE_RANGE = ("2025-11-01", "2025-12-31")
