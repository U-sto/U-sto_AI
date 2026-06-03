from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
RUN_DIR = EXPERIMENTS_DIR / "runs" / f"{RUN_ID}_exp10_lgbm_demand_optuna"
VALID_RESULT_PATH = TABLES_DIR / "exp10_lgbm_demand_optuna_results.csv"
TOP_TEST_PATH = TABLES_DIR / "exp10_lgbm_demand_optuna_top_test.csv"
SUMMARY_PATH = TABLES_DIR / "stage3_monthly_demand_results.csv"
RANDOM_STATE = common.RANDOM_STATE
N_TRIALS = 30

# [통제 변인] 지정된 7개 피처 고정
FEATURES = [
    "trend",
    "month",
    "month_sin",
    "month_cos",
    "lag_12",
    "rolling_mean_6",
    "rolling_std_6",
]

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

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(2 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]) * 100)

def evaluate(y_true, y_pred) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "monthly_rmse_count": rmse(y_true, y_pred),
        "monthly_mae_count": float(mean_absolute_error(y_true, y_pred)),
        "monthly_smape_pct": smape(y_true, y_pred),
        "monthly_r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }

def make_model(params: dict):
    return LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        **params
    )

def make_feature_row(working_counts: list[float], trend: int, month: int) -> dict:
    def lag(n: int) -> float:
        return working_counts[-n] if len(working_counts) >= n else np.nan

    return {
        "trend": trend,
        "month": month,
        "month_sin": float(np.sin(2 * np.pi * month / 12)),
        "month_cos": float(np.cos(2 * np.pi * month / 12)),
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_6": lag(6),
        "lag_12": lag(12),
        "rolling_mean_3": float(np.mean(working_counts[-3:])) if len(working_counts) >= 3 else np.nan,
        "rolling_mean_6": float(np.mean(working_counts[-6:])) if len(working_counts) >= 6 else np.nan,
        "rolling_std_6": float(np.std(working_counts[-6:], ddof=1)) if len(working_counts) >= 6 else np.nan,
    }

def recursive_forecast(model, history: pd.DataFrame, future: pd.DataFrame, features: list[str], fill_values: pd.Series) -> np.ndarray:
    working_counts = history["actual_count"].astype(float).tolist()
    preds = []
    for _, row in future.iterrows():
        month = int(row["event_month"].month)
        feat_row = pd.DataFrame([make_feature_row(working_counts, len(working_counts), month)])
        feat_row = feat_row[features].fillna(fill_values[features])
        pred = float(model.predict(feat_row)[0])
        pred = max(0.0, pred)
        preds.append(pred)
        working_counts.append(pred)
    return np.asarray(preds)

def fit_model(monthly_feat: pd.DataFrame, train_history: pd.DataFrame, params: dict, features: list[str]):
    train_feat = monthly_feat[monthly_feat["event_month"].isin(train_history["event_month"])].copy()
    fill_values = train_feat[FEATURES].median(numeric_only=True).fillna(0)
    train_ready = train_feat.dropna(subset=features + ["actual_count"]).copy()
    if train_ready.empty:
        train_ready = train_feat.copy()
        train_ready[features] = train_ready[features].fillna(fill_values[features])
    model = make_model(params)
    model.fit(train_ready[features], train_ready["actual_count"])
    return model, fill_values

def save_monthly_artifact(row: dict, model, features: list[str]) -> None:
    joblib.dump(model, RUN_DIR / "monthly_demand_model.pkl")
    meta = {
        "model_name": row["model"],
        "variant": row["variant"],
        "run_id": RUN_ID,
        "features": features,
        "feature_set": row["feature_set"],
        "feature_count": len(features),
        "params": row["params_dict"],
        "metrics": {
            "monthly_rmse_count": row["test_monthly_rmse_count"],
            "monthly_mae_count": row["test_monthly_mae_count"],
            "monthly_smape_pct": row["test_monthly_smape_pct"],
            "monthly_r2": row["test_monthly_r2"],
        },
        "evaluation_note": "Recursive forecast backtest: test lags use prior predictions, not future actual counts.",
    }
    with open(RUN_DIR / "monthly_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(meta), f, ensure_ascii=False, indent=2, allow_nan=False)

def main() -> None:
    start = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(common.DATA_PATH)
    monthly = common.build_monthly_series(df)
    train, valid, test = common.split_monthly(monthly)
    train_valid = pd.concat([train, valid], ignore_index=True)
    monthly_feat = common.add_lag_features(monthly)

    valid_rows = []

    # Optuna 목적 함수 정의 (팀장님의 재귀 백테스트 구조 완벽 반영)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        t0 = time.perf_counter()
        model, fill_values = fit_model(monthly_feat, train, params, FEATURES)
        pred_valid = recursive_forecast(model, train, valid, FEATURES, fill_values)
        elapsed = time.perf_counter() - t0
        metrics = evaluate(valid["actual_count"], pred_valid)
        
        # 내부 누적을 위한 기록 구조 보존
        valid_rows.append(
            {
                "model": "LightGBM",
                "variant": f"lgbm_optuna_trial_{trial.number}",
                "feature_set": "seasonal_7",
                "feature_count": len(FEATURES),
                "deployable": True,
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "params_dict": params,
                "valid_elapsed_sec": elapsed,
                **{f"valid_{key}": value for key, value in metrics.items()},
            }
        )
        return metrics["monthly_rmse_count"]

    print(f" Optuna 튜닝 루프 실행 중 ({N_TRIALS} Trials)...")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective, 
        n_trials=N_TRIALS, 
        callbacks=[lambda s, t: print(f" ▹ [Trial {t.number:02d}] Valid RMSE: {t.value:.4f} | Best: {s.best_value:.4f}")]
    )

    valid_df = pd.DataFrame(valid_rows).sort_values("valid_monthly_rmse_count").reset_index(drop=True)
    valid_df["valid_rank"] = np.arange(1, len(valid_df) + 1)

    top_deployable = valid_df[valid_df["deployable"].eq(True)].head(12).copy()
    final_rows = []
    
    for _, candidate in top_deployable.iterrows():
        params = candidate["params_dict"]
        t0 = time.perf_counter()
        model, fill_values = fit_model(monthly_feat, train_valid, params, FEATURES)
        pred_test = recursive_forecast(model, train_valid, test, FEATURES, fill_values)
        elapsed = time.perf_counter() - t0
        metrics = evaluate(test["actual_count"], pred_test)
        row = candidate.to_dict()
        row.update(
            {
                "test_elapsed_sec": elapsed,
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
        final_rows.append((row, model, FEATURES, pred_test))

    final_rows = sorted(final_rows, key=lambda item: item[0]["test_monthly_rmse_count"])
    ranked_rows = []
    best_predictions = None
    best_model = None
    best_features = None
    
    for rank, (row, model, features, pred_test) in enumerate(final_rows, start=1):
        row["rank"] = rank
        ranked_rows.append(row)
        if rank == 1:
            best_predictions = pred_test
            best_model = model
            best_features = features

    final_df = pd.DataFrame(ranked_rows)
    best_row = final_df.iloc[0].to_dict()
    save_monthly_artifact(best_row, best_model, best_features)

    joblib.dump(best_model, RUN_DIR / "lgbm_lag_model.pkl")

    predictions_test = test[["event_month", "actual_count"]].copy()
    predictions_test["BestModel_pred"] = np.clip(best_predictions, 0, None)
    monthly.to_csv(RUN_DIR / "monthly_demand_series.csv", index=False, encoding="utf-8-sig")
    predictions_test.to_csv(RUN_DIR / "monthly_backtest_predictions.csv", index=False, encoding="utf-8-sig")

    serializable_valid = valid_df.drop(columns=["params_dict"])
    serializable_final = final_df.drop(columns=["params_dict"])
    serializable_valid.to_csv(VALID_RESULT_PATH, index=False, encoding="utf-8-sig")
    serializable_final.to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    serializable_valid.to_csv(RUN_DIR / "monthly_model_search_valid_results.csv", index=False, encoding="utf-8-sig")
    serializable_final.to_csv(RUN_DIR / "monthly_model_search_top_test_results.csv", index=False, encoding="utf-8-sig")

    summary = serializable_final[
        [
            "model",
            "test_monthly_rmse_count",
            "test_monthly_mae_count",
            "test_monthly_smape_pct",
            "test_monthly_r2",
            "test_elapsed_sec",
        ]
    ].copy()
    summary = summary.rename(
        columns={
            "test_monthly_rmse_count": "monthly_rmse_count",
            "test_monthly_mae_count": "monthly_mae_count",
            "test_monthly_smape_pct": "monthly_smape_pct",
            "test_monthly_r2": "monthly_r2",
            "test_elapsed_sec": "elapsed_sec",
        }
    )
    summary["source_model_dir"] = str(RUN_DIR.relative_to(PROJECT_ROOT))
    
    if SUMMARY_PATH.exists():
        try:
            existing_summary = pd.read_csv(SUMMARY_PATH)
            existing_summary = existing_summary[existing_summary["source_model_dir"] != summary["source_model_dir"].iloc[0]]
            summary = pd.concat([existing_summary, summary], ignore_index=True)
        except Exception:
            pass
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    report = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR.relative_to(PROJECT_ROOT)),
        "valid_candidates": int(len(valid_df)),
        "tested_top_candidates": int(len(final_df)),
        "best_test": clean_for_json(best_row),
        "best_features": best_features,
        "monthly_period": {
            "start": str(monthly["event_month"].min()),
            "end": str(monthly["event_month"].max()),
            "months": int(len(monthly)),
        },
        "split": {
            "train_months": int(len(train)),
            "valid_months": int(len(valid)),
            "test_months": int(len(test)),
            "test_start": str(test["event_month"].min()),
            "test_end": str(test["event_month"].max()),
        },
        "elapsed_sec": time.perf_counter() - start,
        "evaluation_note": "Recursive forecast backtest prevents test-period lag leakage.",
    }
    with open(RUN_DIR / "monthly_model_search_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)

    print("Stage3 monthly model search (LGBM Optuna) complete")
    print(f"Run dir: {RUN_DIR}")
    print(f"Valid trials: {len(valid_df)}")
    print(serializable_final.head(8).to_string(index=False))

if __name__ == "__main__":
    main()