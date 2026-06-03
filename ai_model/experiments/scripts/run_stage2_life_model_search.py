from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, r2_score
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common


PROJECT_ROOT = common.PROJECT_ROOT
EXPERIMENTS_DIR = common.EXPERIMENTS_DIR
OUTPUTS_DIR = common.OUTPUTS_DIR
TABLES_DIR = common.TABLES_DIR
REPORTS_DIR = common.REPORTS_DIR
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = EXPERIMENTS_DIR / "runs" / f"{RUN_ID}_stage2_life_model_search"
RESULT_PATH = TABLES_DIR / "stage2_life_model_search_results.csv"
TOP_TEST_PATH = TABLES_DIR / "stage2_life_model_search_top_test.csv"
DEPLOY_DIR = PROJECT_ROOT / "ai_model" / "saved_models" / "current"
RANDOM_STATE = common.RANDOM_STATE
TERM_MONTHS = 6


def clean_for_json(value):
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def infer_target_columns(train: pd.DataFrame) -> tuple[str, str]:
    raw_cols = set(pd.read_csv(common.DATA_PATH, nrows=0).columns)
    added_cols = [col for col in train.columns if col not in raw_cols]
    if len(added_cols) < 2:
        raise ValueError(f"Could not infer target columns. Added columns: {added_cols}")
    return added_cols[0], added_cols[1]


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true, y_pred, age_months) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    age_months = np.asarray(age_months, dtype=float)
    actual_rul = y_true - age_months
    pred_rul = y_pred - age_months
    actual_term = actual_rul <= TERM_MONTHS
    pred_term = pred_rul <= TERM_MONTHS
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_term, pred_term, average="binary", zero_division=0
    )
    return {
        "rmse_months": rmse(y_true, y_pred),
        "mae_months": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "term_precision": float(precision),
        "term_recall": float(recall),
        "term_f1": float(f1),
    }


def make_model(model_key: str, params: dict):
    if model_key == "CatBoost":
        return CatBoostRegressor(
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    if model_key == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            **params,
        )
    if model_key == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_key == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_key == "GradientBoosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Unknown model_key: {model_key}")


def model_specs() -> list[dict]:
    return [
        {
            "model": "CatBoost",
            "variant": "cb_balanced",
            "params": {"iterations": 700, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 5},
        },
        {
            "model": "CatBoost",
            "variant": "cb_shallow_regularized",
            "params": {"iterations": 900, "learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 8},
        },
        {
            "model": "CatBoost",
            "variant": "cb_fast_shallow",
            "params": {"iterations": 500, "learning_rate": 0.06, "depth": 5, "l2_leaf_reg": 3},
        },
        {
            "model": "CatBoost",
            "variant": "cb_deeper_slow",
            "params": {"iterations": 800, "learning_rate": 0.035, "depth": 7, "l2_leaf_reg": 6},
        },
        {
            "model": "XGBoost",
            "variant": "xgb_balanced",
            "params": {
                "n_estimators": 700,
                "learning_rate": 0.035,
                "max_depth": 4,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.5,
                "min_child_weight": 1,
            },
        },
        {
            "model": "XGBoost",
            "variant": "xgb_shallow_regularized",
            "params": {
                "n_estimators": 900,
                "learning_rate": 0.025,
                "max_depth": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "min_child_weight": 2,
            },
        },
        {
            "model": "XGBoost",
            "variant": "xgb_deeper",
            "params": {
                "n_estimators": 550,
                "learning_rate": 0.04,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "min_child_weight": 1,
            },
        },
        {
            "model": "XGBoost",
            "variant": "xgb_small",
            "params": {
                "n_estimators": 350,
                "learning_rate": 0.06,
                "max_depth": 3,
                "subsample": 0.95,
                "colsample_bytree": 0.95,
                "reg_lambda": 1.0,
                "min_child_weight": 1,
            },
        },
        {
            "model": "ExtraTrees",
            "variant": "et_full",
            "params": {"n_estimators": 700, "max_depth": None, "min_samples_leaf": 1, "max_features": 0.85},
        },
        {
            "model": "ExtraTrees",
            "variant": "et_regularized",
            "params": {"n_estimators": 700, "max_depth": 18, "min_samples_leaf": 2, "max_features": 0.65},
        },
        {
            "model": "RandomForest",
            "variant": "rf_regularized",
            "params": {"n_estimators": 500, "max_depth": 18, "min_samples_leaf": 1, "max_features": 0.75},
        },
        {
            "model": "GradientBoosting",
            "variant": "gbr_slow",
            "params": {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3, "subsample": 0.9},
        },
        {
            "model": "GradientBoosting",
            "variant": "gbr_deeper",
            "params": {"n_estimators": 350, "learning_rate": 0.035, "max_depth": 4, "subsample": 0.9},
        },
    ]


def ordered_subset(features: list[str], indexes: list[int]) -> list[str]:
    return [features[i] for i in indexes if 0 <= i < len(features)]


def build_feature_sets(train: pd.DataFrame, target_col: str) -> dict[str, list[str]]:
    features = list(common.LIFE_FEATURES)
    feature_sets = {
        "full_35": features,
        "no_category_codes_31": features[:31],
        "asset_usage_maintenance_25": ordered_subset(
            features,
            list(range(0, 7)) + list(range(10, 24)) + [25, 26, 27, 29, 30],
        ),
        "compact_domain_20": ordered_subset(
            features,
            [0, 1, 2, 3, 4, 5, 6, 10, 14, 16, 17, 19, 21, 22, 23, 25, 26, 27, 29, 30],
        ),
        "simple_asset_11": ordered_subset(features, list(range(0, 7)) + list(range(31, 35))),
        "no_maintenance_26": ordered_subset(features, list(range(0, 15)) + list(range(24, 35))),
    }

    selector = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector.fit(train[features], train[target_col])
    importance = pd.Series(selector.feature_importances_, index=features).sort_values(ascending=False)
    for k in [10, 15, 20, 25]:
        feature_sets[f"importance_top_{k}"] = list(importance.head(k).index)

    deduped = {}
    seen = set()
    for name, cols in feature_sets.items():
        cols = [col for col in features if col in cols]
        signature = tuple(cols)
        if signature not in seen and len(cols) >= 3:
            deduped[name] = cols
            seen.add(signature)
    return deduped


def get_current_rmse() -> float | None:
    meta_path = DEPLOY_DIR / "model_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return float(meta.get("metrics", {}).get("rmse_months"))
    except Exception:
        return None


def save_artifact(row: dict, model, features: list[str], target_col: str) -> Path:
    artifact_dir = RUN_DIR / f"{row['rank']:02d}_{row['model']}_{row['feature_set']}_{row['variant']}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.pkl")
    meta = {
        "model_name": row["model"],
        "variant": row["variant"],
        "run_id": RUN_ID,
        "features": features,
        "feature_set": row["feature_set"],
        "feature_count": len(features),
        "target": target_col,
        "target_meaning": "total_life_months",
        "train_rows": int(row["train_rows"]),
        "valid_rows": int(row["valid_rows"]),
        "test_rows": int(row["test_rows"]),
        "params": row["params_dict"],
        "metrics": {
            "rmse_months": row["test_rmse_months"],
            "mae_months": row["test_mae_months"],
            "r2": row["test_r2"],
            "term_precision": row["test_term_precision"],
            "term_recall": row["test_term_recall"],
            "term_f1": row["test_term_f1"],
        },
        "selection": {
            "valid_rmse_months": row["valid_rmse_months"],
            "rank_by_test_rmse_among_top_valid": int(row["rank"]),
        },
        "leakage_guard": {
            "excluded_columns": sorted(common.LEAKAGE_COLUMNS),
            "note": "Outcome/date/status columns are excluded from the feature list.",
        },
    }
    with open(artifact_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(meta), f, ensure_ascii=False, indent=2, allow_nan=False)
    return artifact_dir


def main() -> None:
    start = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, valid, test, pred = common.prepare_life_data()
    target_col, age_col = infer_target_columns(train)
    feature_sets = build_feature_sets(train, target_col)
    specs = model_specs()

    y_train = train[target_col].to_numpy()
    y_valid = valid[target_col].to_numpy()
    y_train_valid = pd.concat([train, valid], ignore_index=True)[target_col].to_numpy()
    train_valid = pd.concat([train, valid], ignore_index=True)
    y_test = test[target_col].to_numpy()

    valid_rows = []
    for feature_set, features in feature_sets.items():
        for spec in specs:
            model = make_model(spec["model"], spec["params"])
            t0 = time.perf_counter()
            model.fit(train[features], y_train)
            pred_valid = np.asarray(model.predict(valid[features]), dtype=float)
            elapsed = time.perf_counter() - t0
            metrics = evaluate(y_valid, pred_valid, valid[age_col])
            valid_rows.append(
                {
                    "model": spec["model"],
                    "variant": spec["variant"],
                    "feature_set": feature_set,
                    "feature_count": len(features),
                    "params": json.dumps(spec["params"], ensure_ascii=False, sort_keys=True),
                    "params_dict": spec["params"],
                    "valid_elapsed_sec": elapsed,
                    **{f"valid_{key}": value for key, value in metrics.items()},
                }
            )

    valid_df = pd.DataFrame(valid_rows).sort_values("valid_rmse_months").reset_index(drop=True)
    valid_df["valid_rank"] = np.arange(1, len(valid_df) + 1)

    top_valid = valid_df.head(12).copy()
    final_rows = []
    feature_lookup = feature_sets
    spec_lookup = {(spec["model"], spec["variant"]): spec for spec in specs}
    for _, candidate in top_valid.iterrows():
        spec = spec_lookup[(candidate["model"], candidate["variant"])]
        features = feature_lookup[candidate["feature_set"]]
        final_model = make_model(spec["model"], spec["params"])
        t0 = time.perf_counter()
        final_model.fit(train_valid[features], y_train_valid)
        pred_test = np.asarray(final_model.predict(test[features]), dtype=float)
        elapsed = time.perf_counter() - t0
        test_metrics = evaluate(y_test, pred_test, test[age_col])
        row = candidate.to_dict()
        row.update(
            {
                "test_elapsed_sec": elapsed,
                "train_rows": len(train),
                "valid_rows": len(valid),
                "test_rows": len(test),
                "prediction_rows": len(pred),
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
        final_rows.append((row, final_model, features))

    final_rows = sorted(final_rows, key=lambda item: item[0]["test_rmse_months"])
    ranked_rows = []
    for rank, (row, model, features) in enumerate(final_rows, start=1):
        row["rank"] = rank
        artifact_dir = save_artifact(row, model, features, target_col)
        row["artifact_dir"] = str(artifact_dir.relative_to(PROJECT_ROOT))
        ranked_rows.append(row)

    final_df = pd.DataFrame(ranked_rows)
    serializable_valid = valid_df.drop(columns=["params_dict"])
    serializable_final = final_df.drop(columns=["params_dict"])
    serializable_valid.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    serializable_final.to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    serializable_valid.to_csv(RUN_DIR / "life_model_search_valid_results.csv", index=False, encoding="utf-8-sig")
    serializable_final.to_csv(RUN_DIR / "life_model_search_top_test_results.csv", index=False, encoding="utf-8-sig")

    current_rmse = get_current_rmse()
    best = final_df.iloc[0].to_dict()
    deployed = False
    if current_rmse is None or float(best["test_rmse_months"]) < current_rmse:
        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        best_dir = PROJECT_ROOT / best["artifact_dir"]
        shutil.copy2(best_dir / "model.pkl", DEPLOY_DIR / "model.pkl")
        shutil.copy2(best_dir / "model_meta.json", DEPLOY_DIR / "model_meta.json")
        deployed = True

    report = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR.relative_to(PROJECT_ROOT)),
        "valid_candidates": int(len(valid_df)),
        "tested_top_candidates": int(len(final_df)),
        "feature_sets": {name: len(cols) for name, cols in feature_sets.items()},
        "current_rmse_before": current_rmse,
        "best_test": clean_for_json(best),
        "deployed_to_current": deployed,
        "elapsed_sec": time.perf_counter() - start,
    }
    with open(RUN_DIR / "life_model_search_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)

    print("Stage2 life model search complete")
    print(f"Run dir: {RUN_DIR}")
    print(f"Valid candidates: {len(valid_df)}")
    print(serializable_final.head(8).to_string(index=False))
    print(f"Current RMSE before: {current_rmse}")
    print(f"Deployed to saved_models/current: {deployed}")


if __name__ == "__main__":
    main()
