from __future__ import annotations

import ast
import json
import math
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, r2_score
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset" / "create_data" / "data_ml" / "phase4_training_data.csv"
STAGE2_RESULTS_PATH = PROJECT_ROOT / "ai_model" / "experiments" / "stage2_valid_tuning_results.csv"
APP_SERVER_PATH = PROJECT_ROOT / "app" / "ai_server.py"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_ROOT / "ai_model" / "experiments" / "runs" / f"{RUN_ID}_stage4_6_final_validation"
SUMMARY_PATH = PROJECT_ROOT / "ai_model" / "experiments" / "stage4_6_final_validation_results.csv"
FINAL_REPORT_PATH = PROJECT_ROOT / "ai_model" / "experiments" / "final_model_selection_report.md"
RANDOM_STATE = 42
TERM_MONTHS = 6
TODAY = pd.Timestamp("2026-02-10")

FEATURES = [
    "내용연수",
    "취득금액",
    "부서가혹도",
    "가격민감도",
    "장비중요도",
    "리드타임등급",
    "취득월",
    "G2B목록명_Code",
    "물품분류명_Code",
    "운용부서코드_Code",
    "캠퍼스_Code",
]

NUMERIC_BASE_FEATURES = [
    "내용연수",
    "취득금액",
    "부서가혹도",
    "가격민감도",
    "장비중요도",
    "리드타임등급",
    "취득월",
]

CATEGORICAL_FOR_TE = ["G2B목록명", "물품분류명", "운용부서코드", "캠퍼스"]


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_total_life(
    name: str,
    y_true,
    y_pred,
    age_months,
    elapsed_sec: float = 0.0,
    extra: dict | None = None,
) -> dict:
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
    row = {
        "model": name,
        "rmse_months": rmse(y_true, y_pred),
        "mae_months": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "term_precision": float(precision),
        "term_recall": float(recall),
        "term_f1": float(f1),
        "elapsed_sec": float(elapsed_sec),
    }
    if extra:
        row.update(extra)
    return row


def clean_for_json(value):
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def parse_params(value: str) -> dict:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def load_stage2_table() -> pd.DataFrame:
    table = pd.read_csv(STAGE2_RESULTS_PATH)
    table["params"] = table["best_params_json"].apply(parse_params)
    table["artifact_path"] = table["artifact_dir"].apply(lambda p: PROJECT_ROOT / Path(p))
    return table.sort_values("test_rmse_months").reset_index(drop=True)


def load_artifact_model(row: pd.Series):
    model_path = row["artifact_path"] / "model.pkl"
    meta_path = row["artifact_path"] / "model_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(model_path), meta


def build_model(model_name: str, params: dict):
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "XGBoost":
        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=1,
            objective="reg:squarederror",
            **params,
        )
    if model_name == "CatBoost":
        return CatBoostRegressor(
            random_seed=RANDOM_STATE,
            verbose=False,
            loss_function="RMSE",
            allow_writing_files=False,
            **params,
        )
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def add_target_encoding(train_df: pd.DataFrame, apply_df: pd.DataFrame, smoothing_factor: int = 10) -> pd.DataFrame:
    out = apply_df.copy()
    global_mean = train_df["실제수명"].mean()
    if pd.isna(global_mean):
        global_mean = 5.0
    for col in CATEGORICAL_FOR_TE:
        train_col = train_df[col].fillna("Unknown").astype(str)
        apply_col = out[col].fillna("Unknown").astype(str)
        target_mean = train_df.assign(_col=train_col).groupby("_col")["실제수명"].mean()
        category_counts = train_col.value_counts()
        smoothed = (target_mean * category_counts + global_mean * smoothing_factor) / (
            category_counts + smoothing_factor
        )
        out[f"{col}_Code"] = apply_col.map(smoothed).fillna(global_mean)
    return out


def build_acquisition_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    y_df = df[df["학습데이터여부"].eq("Y")].copy()
    y_df["취득일자"] = pd.to_datetime(y_df["취득일자"], errors="coerce")
    y_df = y_df.dropna(subset=["취득일자"]).sort_values("취득일자")
    cutoff = y_df["취득일자"].quantile(0.80)
    train_raw = y_df[y_df["취득일자"] < cutoff].copy()
    holdout_raw = y_df[y_df["취득일자"] >= cutoff].copy()
    train = add_target_encoding(train_raw, train_raw)
    holdout = add_target_encoding(train_raw, holdout_raw)
    for col in NUMERIC_BASE_FEATURES:
        median = train[col].median()
        train[col] = train[col].fillna(median)
        holdout[col] = holdout[col].fillna(median)
    return train, holdout, cutoff


def evaluate_artifact_and_ensembles(df: pd.DataFrame, stage2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df[df["데이터세트구분"].eq("Test") & df["학습데이터여부"].eq("Y")].copy()
    y_test = test["실제수명"].astype(float).to_numpy() * 12
    age_months = test["운용연차"].astype(float).to_numpy() * 12

    pred_table = test[["물품고유번호", "G2B목록명", "운용연차", "실제수명"]].copy()
    pred_table["actual_total_life_months"] = y_test

    rows = []
    preds = {}
    for _, row in stage2.iterrows():
        model, meta = load_artifact_model(row)
        features = meta["features"]
        t0 = time.perf_counter()
        pred = np.asarray(model.predict(test[features]), dtype=float)
        elapsed = time.perf_counter() - t0
        preds[row["model"]] = pred
        pred_table[f"{row['model']}_pred_total_life_months"] = pred
        rows.append(
            evaluate_total_life(
                row["model"],
                y_test,
                pred,
                age_months,
                elapsed,
                {"stage": "stage4_test_artifact", "source_artifact": str(row["artifact_path"])},
            )
        )

    ensemble_specs = {}
    top2_names = stage2.head(2)["model"].tolist()
    top3_names = stage2.head(3)["model"].tolist()
    all_names = stage2["model"].tolist()
    ensemble_specs["AverageTop2"] = np.mean([preds[name] for name in top2_names], axis=0)
    ensemble_specs["AverageTop3"] = np.mean([preds[name] for name in top3_names], axis=0)
    ensemble_specs["AverageAll4"] = np.mean([preds[name] for name in all_names], axis=0)
    inv_weights = np.asarray([1.0 / stage2.loc[stage2["model"].eq(name), "valid_rmse_months"].iloc[0] for name in all_names])
    inv_weights = inv_weights / inv_weights.sum()
    ensemble_specs["WeightedByValidRMSE"] = np.average([preds[name] for name in all_names], axis=0, weights=inv_weights)

    for name, pred in ensemble_specs.items():
        pred_table[f"{name}_pred_total_life_months"] = pred
        rows.append(
            evaluate_total_life(
                name,
                y_test,
                pred,
                age_months,
                0.0,
                {"stage": "stage4_test_ensemble", "source_artifact": "stage2_artifact_predictions"},
            )
        )
    return pd.DataFrame(rows).sort_values("rmse_months").reset_index(drop=True), pred_table


def evaluate_acquisition_holdout(df: pd.DataFrame, stage2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train, holdout, cutoff = build_acquisition_holdout(df)
    y_train = train["실제수명"].astype(float).to_numpy() * 12
    y_holdout = holdout["실제수명"].astype(float).to_numpy() * 12
    age_months = holdout["운용연차"].astype(float).to_numpy() * 12

    pred_table = holdout[["물품고유번호", "G2B목록명", "취득일자", "운용연차", "실제수명"]].copy()
    pred_table["actual_total_life_months"] = y_holdout
    rows = []
    preds = {}

    for _, row in stage2.iterrows():
        model = build_model(row["model"], row["params"])
        t0 = time.perf_counter()
        model.fit(train[FEATURES], y_train)
        pred = np.asarray(model.predict(holdout[FEATURES]), dtype=float)
        elapsed = time.perf_counter() - t0
        preds[row["model"]] = pred
        pred_table[f"{row['model']}_pred_total_life_months"] = pred
        rows.append(
            evaluate_total_life(
                row["model"],
                y_holdout,
                pred,
                age_months,
                elapsed,
                {
                    "stage": "acquisition_recent_holdout",
                    "holdout_cutoff": str(cutoff.date()),
                    "train_rows": len(train),
                    "holdout_rows": len(holdout),
                },
            )
        )

    top2_names = stage2.head(2)["model"].tolist()
    top3_names = stage2.head(3)["model"].tolist()
    all_names = stage2["model"].tolist()
    ensembles = {
        "AverageTop2": np.mean([preds[name] for name in top2_names], axis=0),
        "AverageTop3": np.mean([preds[name] for name in top3_names], axis=0),
        "AverageAll4": np.mean([preds[name] for name in all_names], axis=0),
    }
    inv_weights = np.asarray([1.0 / stage2.loc[stage2["model"].eq(name), "valid_rmse_months"].iloc[0] for name in all_names])
    inv_weights = inv_weights / inv_weights.sum()
    ensembles["WeightedByValidRMSE"] = np.average([preds[name] for name in all_names], axis=0, weights=inv_weights)
    for name, pred in ensembles.items():
        pred_table[f"{name}_pred_total_life_months"] = pred
        rows.append(
            evaluate_total_life(
                name,
                y_holdout,
                pred,
                age_months,
                0.0,
                {
                    "stage": "acquisition_recent_holdout_ensemble",
                    "holdout_cutoff": str(cutoff.date()),
                    "train_rows": len(train),
                    "holdout_rows": len(holdout),
                },
            )
        )
    return pd.DataFrame(rows).sort_values("rmse_months").reset_index(drop=True), pred_table, cutoff


def validate_procurement_smoke(df: pd.DataFrame, best_model, features: list[str]) -> tuple[dict, pd.DataFrame]:
    pred_df = df[df["학습데이터여부"].eq("N")].copy()
    pred_df = pred_df.dropna(subset=features + ["운용연차"]).copy()
    pred_total = np.asarray(best_model.predict(pred_df[features]), dtype=float)
    age_months = pred_df["운용연차"].astype(float).to_numpy() * 12
    raw_rul = pred_total - age_months
    pred_df["예측총수명_개월"] = pred_total
    pred_df["RUL_개월_raw"] = raw_rul
    pred_df["RUL_개월"] = np.clip(raw_rul, 0.5, None)
    pred_df["AI예측고장일"] = TODAY + pd.to_timedelta(pred_df["RUL_개월"] * 30.4375, unit="D")
    pred_df["고장예상월"] = pred_df["AI예측고장일"].dt.to_period("M").dt.to_timestamp()
    monthly = pred_df.groupby("고장예상월").size().rename("predicted_failure_count").reset_index()
    summary = {
        "prediction_rows": int(len(pred_df)),
        "negative_raw_rul_count": int((pred_df["RUL_개월_raw"] < 0).sum()),
        "min_rul_months": float(pred_df["RUL_개월"].min()),
        "median_rul_months": float(pred_df["RUL_개월"].median()),
        "max_rul_months": float(pred_df["RUL_개월"].max()),
        "term_6m_failure_count": int((pred_df["RUL_개월"] <= 6).sum()),
        "term_12m_failure_count": int((pred_df["RUL_개월"] <= 12).sum()),
        "first_failure_month": str(monthly["고장예상월"].min().date()) if not monthly.empty else None,
        "last_failure_month": str(monthly["고장예상월"].max().date()) if not monthly.empty else None,
    }
    return summary, monthly


def parse_app_feature_list() -> list[str]:
    if not APP_SERVER_PATH.exists():
        return []
    text = APP_SERVER_PATH.read_text(encoding="utf-8")
    match = re.search(r"features\s*=\s*(\[[^\]]+\])", text)
    if not match:
        return []
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return []


def check_server_compatibility(meta_features: list[str]) -> dict:
    app_features = parse_app_feature_list()
    return {
        "app_feature_count": len(app_features),
        "model_feature_count": len(meta_features),
        "app_features": app_features,
        "model_features": meta_features,
        "missing_in_app": [f for f in meta_features if f not in app_features],
        "extra_in_app": [f for f in app_features if f not in meta_features],
        "app_uses_model_meta_features": False,
        "app_rul_postprocess_expected": "현재 서버는 predict 결과를 현재일 + 예측수명_월로 바로 더한다. 총수명 모델이면 운용연차*12를 빼는 RUL 계산이 필요하다.",
    }


def plot_metric_bars(metrics: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metrics, x="rmse_months", y="model", hue="stage", dodge=False)
    plt.title(title)
    plt.xlabel("RMSE months")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    table = table.fillna("")
    cols = list(table.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    test_metrics: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    procurement_summary: dict,
    compatibility: dict,
    selected: dict,
    cutoff: pd.Timestamp,
) -> None:
    lines = [
        "# Final Model Selection Report",
        "",
        f"- Run ID: `{RUN_ID}`",
        f"- Run dir: `{RUN_DIR.relative_to(PROJECT_ROOT)}`",
        f"- Acquisition-date holdout cutoff: `{cutoff.date()}`",
        "",
        "## Stage 4. Ensemble Validation",
        "",
        df_to_markdown_simple(test_metrics.head(8)),
        "",
        "## Additional Validation. Recent Acquisition Holdout",
        "",
        df_to_markdown_simple(holdout_metrics.head(8)),
        "",
        "## Stage 5. Procurement Smoke Test",
        "",
        "```json",
        json.dumps(clean_for_json(procurement_summary), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Server Compatibility Check",
        "",
        "```json",
        json.dumps(clean_for_json(compatibility), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Stage 6. Selection",
        "",
        f"- Selected artifact: `{selected['artifact_dir']}`",
        f"- Selected model: `{selected['model']}`",
        "- Deployment was not overwritten by this script.",
        "- Before replacing `rf_final_model.pkl`, align server inference with `model_meta.json` features and RUL postprocessing.",
        "",
    ]
    FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["실제수명_개월"] = df["실제수명"] * 12
    stage2 = load_stage2_table()

    test_metrics, test_predictions = evaluate_artifact_and_ensembles(df, stage2)
    holdout_metrics, holdout_predictions, cutoff = evaluate_acquisition_holdout(df, stage2)

    selected_row = stage2.iloc[0]
    selected_model, selected_meta = load_artifact_model(selected_row)
    procurement_summary, monthly_pred = validate_procurement_smoke(df, selected_model, selected_meta["features"])
    compatibility = check_server_compatibility(selected_meta["features"])

    test_metrics.to_csv(RUN_DIR / "stage4_test_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(RUN_DIR / "stage4_test_predictions.csv", index=False, encoding="utf-8-sig")
    holdout_metrics.to_csv(RUN_DIR / "acquisition_recent_holdout_metrics.csv", index=False, encoding="utf-8-sig")
    holdout_predictions.to_csv(RUN_DIR / "acquisition_recent_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    monthly_pred.to_csv(RUN_DIR / "stage5_prediction_monthly_counts.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat([test_metrics, holdout_metrics], ignore_index=True)
    combined.to_csv(RUN_DIR / "stage4_6_combined_metrics.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    plot_metric_bars(combined, RUN_DIR / "stage4_6_rmse_comparison.png", "Stage 4-6 RMSE Comparison")

    selected = {
        "model": selected_row["model"],
        "artifact_dir": str(selected_row["artifact_path"]),
        "test_rmse_months": float(selected_row["test_rmse_months"]),
        "holdout_rank": int(
            holdout_metrics.reset_index().query("model == @selected_row.model")["index"].iloc[0] + 1
        )
        if selected_row["model"] in set(holdout_metrics["model"])
        else None,
    }
    report = {
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR.relative_to(PROJECT_ROOT)),
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "stage4_best": test_metrics.iloc[0].to_dict(),
        "acquisition_holdout_best": holdout_metrics.iloc[0].to_dict(),
        "selected_stage2_artifact": selected,
        "procurement_smoke_summary": procurement_summary,
        "server_compatibility": compatibility,
        "elapsed_sec": time.perf_counter() - start,
    }
    with open(RUN_DIR / "stage4_6_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2, allow_nan=False)

    # 최종 후보를 별도 폴더에 복사하되, 실제 서버 호환 경로는 덮어쓰지 않는다.
    final_candidate_dir = RUN_DIR / "final_candidate"
    final_candidate_dir.mkdir(exist_ok=True)
    shutil.copy2(selected_row["artifact_path"] / "model.pkl", final_candidate_dir / "model.pkl")
    shutil.copy2(selected_row["artifact_path"] / "model_meta.json", final_candidate_dir / "model_meta.json")

    write_report(test_metrics, holdout_metrics, procurement_summary, compatibility, selected, cutoff)

    print("Stage 4-6 final validation complete")
    print(f"Run dir: {RUN_DIR}")
    print("\n[Stage 4 Test / Ensemble]")
    print(test_metrics.to_string(index=False))
    print("\n[Recent Acquisition Holdout]")
    print(holdout_metrics.to_string(index=False))
    print("\n[Server Compatibility]")
    print(json.dumps(clean_for_json(compatibility), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
