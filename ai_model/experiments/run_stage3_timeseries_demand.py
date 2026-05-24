from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    SARIMAX = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "create_data" / "data_ml" / "phase4_training_data.csv"
STAGE2_RESULTS_PATH = PROJECT_ROOT / "ai_model" / "experiments" / "stage2_valid_tuning_results.csv"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_ROOT / "ai_model" / "experiments" / "runs" / f"{RUN_ID}_stage3_timeseries_demand"
SUMMARY_PATH = PROJECT_ROOT / "ai_model" / "experiments" / "stage3_timeseries_demand_results.csv"
RANDOM_STATE = 42


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


def make_event_date(df: pd.DataFrame) -> pd.Series:
    disuse_date = pd.to_datetime(df["불용일자"], errors="coerce")
    fallback_date = pd.to_datetime(df["취득일자"], errors="coerce") + pd.to_timedelta(
        (df["실제수명"].astype(float) * 365.25).round().astype("Int64"), unit="D"
    )
    return disuse_date.fillna(fallback_date)


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    trainable = df[df["학습데이터여부"].eq("Y")].copy()
    trainable["event_date"] = make_event_date(trainable)
    trainable = trainable.dropna(subset=["event_date"])
    trainable["event_month"] = trainable["event_date"].dt.to_period("M").dt.to_timestamp()

    monthly = trainable.groupby("event_month").size().rename("actual_count").to_frame()
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0)
    monthly.index.name = "event_month"
    monthly = monthly.reset_index()
    monthly["month"] = monthly["event_month"].dt.month
    monthly["year"] = monthly["event_month"].dt.year
    monthly["trend"] = np.arange(len(monthly))
    return monthly


def add_lag_features(monthly: pd.DataFrame) -> pd.DataFrame:
    out = monthly.copy()
    for lag in [1, 2, 3, 6, 12]:
        out[f"lag_{lag}"] = out["actual_count"].shift(lag)
    out["rolling_mean_3"] = out["actual_count"].shift(1).rolling(3).mean()
    out["rolling_mean_6"] = out["actual_count"].shift(1).rolling(6).mean()
    out["rolling_std_6"] = out["actual_count"].shift(1).rolling(6).std()
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def split_monthly(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(monthly) < 36:
        raise ValueError(f"월별 시계열 길이가 너무 짧습니다: {len(monthly)}개월")
    test_months = min(12, max(6, len(monthly) // 6))
    valid_months = min(6, max(3, len(monthly) // 12))
    train = monthly.iloc[: -(valid_months + test_months)].copy()
    valid = monthly.iloc[-(valid_months + test_months) : -test_months].copy()
    test = monthly.iloc[-test_months:].copy()
    return train, valid, test


def evaluate_series_model(name: str, y_true, y_pred, elapsed_sec: float) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "model": name,
        "monthly_rmse_count": rmse(y_true, y_pred),
        "monthly_mae_count": float(mean_absolute_error(y_true, y_pred)),
        "monthly_smape_pct": smape(y_true, y_pred),
        "monthly_r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "elapsed_sec": elapsed_sec,
    }


def seasonal_naive_predict(train_valid: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    history = train_valid.set_index("event_month")["actual_count"]
    preds = []
    fallback = float(history.tail(12).mean())
    for month in test["event_month"]:
        prev_year = month - pd.DateOffset(years=1)
        preds.append(float(history.get(prev_year, fallback)))
    return np.asarray(preds)


def moving_average_predict(train_valid: pd.DataFrame, horizon: int, window: int = 6) -> np.ndarray:
    mean_value = float(train_valid["actual_count"].tail(window).mean())
    return np.repeat(mean_value, horizon)


def sarimax_predict(train_valid: pd.DataFrame, horizon: int) -> np.ndarray:
    if SARIMAX is None:
        raise RuntimeError("statsmodels가 설치되어 있지 않습니다.")
    fitted = fit_sarimax(train_valid)
    return np.asarray(fitted.forecast(horizon), dtype=float)


def fit_sarimax(train_valid: pd.DataFrame):
    if SARIMAX is None:
        raise RuntimeError("statsmodels가 설치되어 있지 않습니다.")
    model = SARIMAX(
        train_valid["actual_count"].astype(float),
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def fit_xgb_monthly(train_valid_feat: pd.DataFrame, feature_cols: list[str]) -> XGBRegressor:
    train_ready = train_valid_feat.dropna(subset=feature_cols + ["actual_count"]).copy()
    model = XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(train_ready[feature_cols], train_ready["actual_count"])
    return model


def xgb_predict(monthly: pd.DataFrame, train_valid: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, XGBRegressor, list[str]]:
    feat = add_lag_features(monthly)
    feature_cols = [
        "trend",
        "month",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_std_6",
    ]
    train_valid_feat = feat[feat["event_month"].isin(train_valid["event_month"])]
    test_feat = feat[feat["event_month"].isin(test["event_month"])].copy()
    fill_values = train_valid_feat[feature_cols].median(numeric_only=True)
    test_feat[feature_cols] = test_feat[feature_cols].fillna(fill_values)
    model = fit_xgb_monthly(train_valid_feat.fillna(fill_values), feature_cols)
    pred = model.predict(test_feat[feature_cols])
    return pred, model, feature_cols


def hybrid_sarimax_xgb_predict(
    monthly: pd.DataFrame,
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    sarimax_pred: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, XGBRegressor]:
    feat = add_lag_features(monthly)
    train_valid_feat = feat[feat["event_month"].isin(train_valid["event_month"])].copy()
    train_valid_feat = train_valid_feat.dropna(subset=feature_cols + ["actual_count"]).copy()
    if len(train_valid_feat) < 24:
        raise ValueError("잔차 보정 모델을 학습할 월별 데이터가 부족합니다.")

    sarimax_fitted = fit_sarimax(train_valid)
    sarimax_in_sample = np.asarray(sarimax_fitted.fittedvalues, dtype=float)
    residual = train_valid["actual_count"].to_numpy(dtype=float) - sarimax_in_sample
    residual_by_month = pd.Series(residual, index=train_valid["event_month"])
    train_valid_feat["residual"] = train_valid_feat["event_month"].map(residual_by_month)

    fill_values = train_valid_feat[feature_cols].median(numeric_only=True)
    residual_model = XGBRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    residual_model.fit(train_valid_feat[feature_cols].fillna(fill_values), train_valid_feat["residual"])

    test_feat = feat[feat["event_month"].isin(test["event_month"])].copy()
    test_feat[feature_cols] = test_feat[feature_cols].fillna(fill_values)
    residual_pred = residual_model.predict(test_feat[feature_cols])
    return np.clip(sarimax_pred + residual_pred, 0, None), residual_model


def load_stage2_best_model() -> tuple[object, dict, str]:
    results = pd.read_csv(STAGE2_RESULTS_PATH)
    best = results.sort_values("test_rmse_months").iloc[0]
    artifact_dir = PROJECT_ROOT / Path(best["artifact_dir"])
    meta_path = artifact_dir / "model_meta.json"
    model_path = artifact_dir / "model.pkl"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(model_path), meta, str(artifact_dir)


def evaluate_stage2_asset_monthly(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    model, meta, artifact_dir = load_stage2_best_model()
    features = meta["features"]
    test_assets = df[df["데이터세트구분"].eq("Test") & df["학습데이터여부"].eq("Y")].copy()
    test_assets["actual_event_date"] = make_event_date(test_assets)
    test_assets = test_assets.dropna(subset=["actual_event_date", "취득일자"]).copy()
    pred_total_months = np.asarray(model.predict(test_assets[features]), dtype=float)
    acq_date = pd.to_datetime(test_assets["취득일자"], errors="coerce")
    test_assets["pred_event_date"] = acq_date + pd.to_timedelta(np.clip(pred_total_months, 0, None) * 30.4375, unit="D")

    min_month = test_assets["actual_event_date"].dt.to_period("M").dt.to_timestamp().min()
    max_month = test_assets["actual_event_date"].dt.to_period("M").dt.to_timestamp().max()
    month_index = pd.date_range(min_month, max_month, freq="MS")

    actual = (
        test_assets.assign(month=test_assets["actual_event_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
        .reindex(month_index, fill_value=0)
    )
    pred = (
        test_assets.assign(month=test_assets["pred_event_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .size()
        .reindex(month_index, fill_value=0)
    )
    backtest = pd.DataFrame({"event_month": month_index, "actual_count": actual.values, "stage2_asset_pred_count": pred.values})
    metrics = evaluate_series_model("Stage2AssetModelMonthly", backtest["actual_count"], backtest["stage2_asset_pred_count"], 0.0)
    metrics["source_model_dir"] = artifact_dir
    return metrics, backtest


def plot_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(predictions["event_month"], predictions["actual_count"], marker="o", label="Actual")
    for col in predictions.columns:
        if col.endswith("_pred"):
            plt.plot(predictions["event_month"], predictions[col], marker="o", alpha=0.8, label=col.replace("_pred", ""))
    plt.title("Stage 3 Monthly Demand Forecast Backtest")
    plt.xlabel("Month")
    plt.ylabel("Failure / Disposal Count")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def clean_for_json(value):
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    start_all = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    monthly = build_monthly_series(df)
    train, valid, test = split_monthly(monthly)
    train_valid = pd.concat([train, valid], ignore_index=True)

    rows = []
    predictions = test[["event_month", "actual_count"]].copy()

    t0 = time.perf_counter()
    pred = seasonal_naive_predict(train_valid, test)
    predictions["SeasonalNaive_pred"] = np.clip(pred, 0, None)
    rows.append(evaluate_series_model("SeasonalNaive", test["actual_count"], predictions["SeasonalNaive_pred"], time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred = moving_average_predict(train_valid, len(test), window=6)
    predictions["MovingAverage6_pred"] = np.clip(pred, 0, None)
    rows.append(evaluate_series_model("MovingAverage6", test["actual_count"], predictions["MovingAverage6_pred"], time.perf_counter() - t0))

    sarimax_pred = None
    if SARIMAX is not None:
        try:
            t0 = time.perf_counter()
            sarimax_pred = sarimax_predict(train_valid, len(test))
            predictions["SARIMAX_pred"] = np.clip(sarimax_pred, 0, None)
            rows.append(evaluate_series_model("SARIMAX", test["actual_count"], predictions["SARIMAX_pred"], time.perf_counter() - t0))
        except Exception as exc:
            rows.append({"model": "SARIMAX", "error": repr(exc)})

    t0 = time.perf_counter()
    xgb_pred, xgb_model, feature_cols = xgb_predict(monthly, train_valid, test)
    predictions["XGBoostLag_pred"] = np.clip(xgb_pred, 0, None)
    rows.append(evaluate_series_model("XGBoostLag", test["actual_count"], predictions["XGBoostLag_pred"], time.perf_counter() - t0))

    if sarimax_pred is not None:
        try:
            t0 = time.perf_counter()
            hybrid_pred, residual_model = hybrid_sarimax_xgb_predict(monthly, train_valid, test, sarimax_pred, feature_cols)
            predictions["SARIMAX_XGBResidual_pred"] = np.clip(hybrid_pred, 0, None)
            rows.append(
                evaluate_series_model(
                    "SARIMAX_XGBResidual",
                    test["actual_count"],
                    predictions["SARIMAX_XGBResidual_pred"],
                    time.perf_counter() - t0,
                )
            )
            joblib.dump(residual_model, RUN_DIR / "sarimax_xgb_residual_model.pkl")
        except Exception as exc:
            rows.append({"model": "SARIMAX_XGBResidual", "error": repr(exc)})

    asset_metrics, asset_backtest = evaluate_stage2_asset_monthly(df)
    rows.append(asset_metrics)

    metrics_df = pd.DataFrame(rows)
    sort_col = "monthly_rmse_count"
    if sort_col in metrics_df.columns:
        metrics_df = metrics_df.sort_values(sort_col, na_position="last").reset_index(drop=True)

    monthly.to_csv(RUN_DIR / "monthly_demand_series.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RUN_DIR / "monthly_backtest_predictions.csv", index=False, encoding="utf-8-sig")
    asset_backtest.to_csv(RUN_DIR / "stage2_asset_monthly_backtest.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(RUN_DIR / "stage3_timeseries_metrics.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    joblib.dump(xgb_model, RUN_DIR / "xgboost_lag_model.pkl")
    plot_predictions(predictions, RUN_DIR / "monthly_demand_backtest.png")

    report = {
        "run_id": RUN_ID,
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "run_dir": str(RUN_DIR.relative_to(PROJECT_ROOT)),
        "monthly_period": {
            "start": str(monthly["event_month"].min().date()),
            "end": str(monthly["event_month"].max().date()),
            "months": int(len(monthly)),
        },
        "split": {
            "train_months": int(len(train)),
            "valid_months": int(len(valid)),
            "test_months": int(len(test)),
            "test_start": str(test["event_month"].min().date()),
            "test_end": str(test["event_month"].max().date()),
        },
        "best_model": metrics_df.iloc[0].to_dict() if not metrics_df.empty else {},
        "elapsed_sec": time.perf_counter() - start_all,
        "note": "Stage 3 is a monthly demand backtest. It is a support experiment for demand charts, not a direct replacement for rf_final_model.pkl.",
    }
    with open(RUN_DIR / "stage3_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)

    print("Stage 3 monthly demand modeling complete")
    print(f"Run dir: {RUN_DIR}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
