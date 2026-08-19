"""Entry point for everything.

    python run.py verify     # check the source data is intact
    python run.py harness    # build the folds and score the baseline
    python run.py ablation   # add one feature group at a time and measure each
    python run.py train      # compare models, losses and feature importance
    python run.py predict    # write both output files
    python run.py all        # run the lot in order

Any stage runs on its own.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config as C
from src import data, model, split


def stage_verify() -> None:
    results = data.verify_integrity(raise_on_failure=False)
    for label, passed, detail in results:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {label}" + (f"  {detail}" if detail else ""))

    failures = [label for label, passed, _ in results if not passed]
    if failures:
        raise SystemExit(f"\n{len(failures)} integrity check(s) failed.")
    print(f"\nAll {len(results)} integrity checks passed.")


def stage_harness() -> None:
    frame = data.repair_weight(data.load_train())
    frame["_inflated"] = data.flag_inflated(frame)

    folds = split.forward_chain_folds(frame, n_folds=3, horizon_days=61)

    print("Forward chaining folds")
    for fold in folds:
        train_n = int(fold.train_mask(frame).sum())
        test_n = int(fold.test_mask(frame).sum())
        print(f"  {fold.describe()}   {train_n:>6,} train   {test_n:>6,} test")

    print(f"\nInjected outliers in the data: {frame._inflated.sum()} "
          f"({frame._inflated.mean() * 100:.2f}%)")

    results = []
    for label, fit_predict in model.BASELINES.items():
        results.append(split.evaluate(fit_predict, frame, folds, label))
        results.append(split.evaluate_random(fit_predict, frame, folds, label))
    results = pd.concat(results, ignore_index=True)

    print("\nPer fold")
    columns = ["model", "harness", "fold", "test_start",
               "n_test", "mae", "mape", "medae", "mae_clean"]
    print(results[columns].round(3).to_string(index=False))

    summary = split.summarise(results)
    print("\nSummary")
    print(summary.to_string(index=False))

    # I measure the leak inside each harness rather than across the two. Random
    # folds draw from a wider stretch of dates than their dated counterparts, so
    # the raw error is not comparable between them. What is comparable is how much
    # a date memorising model gains over the baseline under each one, because
    # there both models are looking at identical data.
    print("\nWhat a random split would have told you")
    for harness in ["forward_chain", "random"]:
        rows = summary[summary.harness == harness].set_index("model")
        gain = rows.loc["null (median rpm)", "mae"] - \
            rows.loc["date lookup (leak probe)", "mae"]
        print(
            f"  {harness:<14} date lookup beats the null model by ${gain:>6.2f} of MAE")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(C.HARNESS_SCORES_CSV, index=False)
    print(f"\nWrote {C.HARNESS_SCORES_CSV.relative_to(C.ROOT)}")


def stage_ablation() -> None:
    """Add one feature group at a time and measure what each is worth."""
    frame = data.repair_weight(data.load_train())
    frame["_inflated"] = data.flag_inflated(frame)
    folds = split.forward_chain_folds(frame, n_folds=3, horizon_days=61)

    results = []

    def run(label, fit_predict):
        scored = split.evaluate(fit_predict, frame, folds, label)
        results.append(scored)
        per_fold = "  ".join(
            f"f{row.fold} {row.mae:6.1f}" for row in scored.itertuples())
        print(
            f"  {label:<26} mean {scored.mae.mean():7.2f}  sd {scored.mae.std():6.2f}   {per_fold}")
        return scored

    print("Cumulative ablation, gradient boosted trees")
    run("null (median rpm)", model.null_model)
    for label, groups in model.ABLATION.items():
        run(label, model.make_gbdt(groups))

    print("\nTested and rejected")
    for label, groups in model.REJECTED.items():
        run(label, model.make_gbdt(groups))

    print("\nLinear rung")
    run("ridge (no trend term)", model.make_ridge(
        model.ABLATION["+ lane native"]))
    # I kept this one on purpose. Ridge carries the trend straight past the end of
    # its training window, where a tree just clamps to the last value it saw. Fold
    # two trains through the summer peak then tests the decline, so the line climbs
    # while the market drops and the error triples. It is the clearest reason the
    # same feature is fine in one model and disastrous in another.
    run("ridge (with trend term)", model.make_ridge(model.SELECTED))

    results = pd.concat(results, ignore_index=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(C.RESULTS_DIR / "ablation_scores.csv", index=False)

    selected = results[results.model == model.SELECTED_LABEL]
    columns = ["fold", "test_start", "n_train",
               "n_test", "mae", "mape", "medae", "mae_clean"]
    print("\nSelected feature set, per fold")
    print(selected[columns].round(3).to_string(index=False))
    print(f"\nagainst the null model: {model.null_model.__name__} "
          f"{results[results.model.str.startswith('null')].mae.mean():.2f} "
          f"-> {selected.mae.mean():.2f}  "
          f"({(1 - selected.mae.mean() / results[results.model.str.startswith('null')].mae.mean()) * 100:.1f}% better)")
    print(f"Wrote {(C.RESULTS_DIR / 'ablation_scores.csv').relative_to(C.ROOT)}")


def stage_train() -> None:
    """Compare models, losses, recency weighting, importance and the chart."""
    import numpy as np

    from src import features as F

    frame = data.repair_weight(data.load_train())
    frame["_inflated"] = data.flag_inflated(frame)
    folds = split.forward_chain_folds(frame, n_folds=3, horizon_days=61)
    groups = model.SELECTED
    results = []

    def run(label, fit_predict):
        scored = split.evaluate(fit_predict, frame, folds, label)
        results.append(scored)
        per_fold = "  ".join(f"f{r.fold} {r.mae:6.1f}" for r in scored.itertuples())
        print(f"  {label:<30} mean {scored.mae.mean():7.2f}  sd {scored.mae.std():6.2f}   {per_fold}")
        return scored.mae.mean()

    print("1. Challenger: LightGBM against the incumbent")
    incumbent = run("HistGradientBoosting", model.make_gbdt(groups))
    challenger = run("LightGBM", model.make_lgbm(groups))

    print("\n2. Loss function, is absolute error actually better?")
    run("absolute error (current)", model.make_gbdt(groups, loss="absolute_error"))
    run("squared error", model.make_gbdt(groups, loss="squared_error"))

    print("\n3. Recency weighting, halflife in days")
    for halflife in (60, 120, 240):
        run(f"halflife {halflife}d", model.make_gbdt(groups, halflife_days=halflife))

    print("\n4. A small hyperparameter pass, kept small on purpose")
    for name, kw in {
        "deeper (63 leaves)": dict(max_leaf_nodes=63),
        "slower (lr .03, 600)": dict(learning_rate=0.03, max_iter=600),
        "more regularised": dict(min_samples_leaf=120, l2_regularization=5.0),
    }.items():
        run(name, model.make_gbdt(groups, **kw))

    print("\n5. What a random split says once a real model is in the loop")
    real_fc = split.evaluate(model.make_gbdt(groups), frame, folds, "gbdt")
    real_rand = split.evaluate_random(model.make_gbdt(groups), frame, folds, "gbdt")
    null_fc = split.evaluate(model.null_model, frame, folds, "null")
    null_rand = split.evaluate_random(model.null_model, frame, folds, "null")
    print(f"  forward chaining   gbdt beats null by ${null_fc.mae.mean() - real_fc.mae.mean():7.2f}")
    print(f"  random split       gbdt beats null by ${null_rand.mae.mean() - real_rand.mae.mean():7.2f}")
    print(f"  a random split overstates the model's value by "
          f"${(null_rand.mae.mean() - real_rand.mae.mean()) - (null_fc.mae.mean() - real_fc.mae.mean()):.2f}")

    print("\n6. Feature importance on fold 3, dollars of error added by shuffling each")
    fold = folds[-1]
    tr, te = frame[fold.train_mask(frame)], frame[fold.test_mask(frame)]
    te_features = te.drop(columns=[C.RAW_TARGET])
    estimator, _, x_test = model.fit_for_inspection(tr, te_features, groups)
    actual = te[C.RAW_TARGET].to_numpy()
    distance = te_features["distance"].to_numpy()
    base = np.abs(actual - estimator.predict(x_test) * distance).mean()

    rng = np.random.default_rng(C.RANDOM_SEED)
    importance = {}
    for column in x_test.columns:
        shuffled = x_test.copy()
        shuffled[column] = shuffled[column].to_numpy()[rng.permutation(len(shuffled))]
        importance[column] = np.abs(actual - estimator.predict(shuffled) * distance).mean() - base
    for column, delta in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"    {column:<20} {delta:+8.2f}")

    print("\n7. December probe preview")
    validation = data.repair_weight(data.load_validation())
    december = F.complete_december(data.load_december(), validation, frame)
    fit_predict = model.make_gbdt(groups)
    preview = fit_predict(frame, december.drop(columns=["predicted_rate"]))
    december["preview"] = preview
    december["weekday"] = december[C.DATE_COL].dt.day_name().str[:3]
    print(december[[C.DATE_COL, "weekday", "market_index", "preview"]].round(2).to_string(index=False))
    peak = december.loc[december.preview.idxmax()]
    trough = december.loc[december.preview.idxmin()]
    print(f"\n  range ${december.preview.min():.2f} to ${december.preview.max():.2f}  "
          f"({(december.preview.max() / december.preview.min() - 1) * 100:.1f}% swing)")
    print(f"  peak   {peak[C.DATE_COL].date()} ({peak.weekday})")
    print(f"  trough {trough[C.DATE_COL].date()} ({trough.weekday})")
    print(f"  distinct values: {december.preview.nunique()} of 31 "
          f"(a flat line would mean time was not modelled)")

    pd.concat(results, ignore_index=True).to_csv(C.RESULTS_DIR / "model_scores.csv", index=False)
    print(f"\nincumbent {incumbent:.2f}   challenger {challenger:.2f}   "
          f"difference {abs(incumbent - challenger):.2f}")
    print(f"Wrote {(C.RESULTS_DIR / 'model_scores.csv').relative_to(C.ROOT)}")


def stage_predict() -> None:
    """Fit on everything, write both output files, then check they look sane."""
    from src import predict

    train = data.repair_weight(data.load_train())
    validation = data.repair_weight(data.load_validation())

    print(f"Fitting on all {len(train):,} training rows "
          f"({train.date.min().date()} to {train.date.max().date()})")

    predicted = predict.predict_validation(train, validation)
    out = predict.write_validation_predictions(validation, predicted)
    print(f"  wrote {C.PREDICTIONS_CSV.name}  {len(out):,} rows, all checks passed")

    december = predict.predict_december(train, validation)
    predict.write_december(december)
    print(f"  wrote {C.DECEMBER_CSV.relative_to(C.ROOT)}  31 rows, all checks passed")

    print("\nDecember chart")
    view = december[[C.DATE_COL, "market_index", "prediction"]].copy()
    view["weekday"] = view[C.DATE_COL].dt.day_name().str[:3]
    print(view[[C.DATE_COL, "weekday", "market_index", "prediction"]].round(2).to_string(index=False))
    swing = (december.prediction.max() / december.prediction.min() - 1) * 100
    print(f"\n  range {december.prediction.min():,.2f} to {december.prediction.max():,.2f}"
          f"  ({swing:.2f}% swing, against a 2.09% weekly amplitude in training)")
    print(f"  peak   {december.loc[december.prediction.idxmax(), C.DATE_COL].date()}")
    print(f"  trough {december.loc[december.prediction.idxmin(), C.DATE_COL].date()}")

    print("\nSanity checks")
    predict.sanity_report(train, validation, predicted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freight rate prediction pipeline.")
    parser.add_argument(
        "stage", choices=["verify", "harness", "ablation", "train", "predict", "all"]
    )
    args = parser.parse_args()

    if args.stage in ("verify", "all"):
        stage_verify()
    if args.stage in ("harness", "all"):
        stage_harness()
    if args.stage in ("ablation", "all"):
        stage_ablation()
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage in ("predict", "all"):
        stage_predict()


if __name__ == "__main__":
    main()
