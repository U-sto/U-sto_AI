from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common


PROJECT_ROOT = common.PROJECT_ROOT
DATA_PATH = common.DATA_PATH
EXPERIMENTS_DIR = common.EXPERIMENTS_DIR
OUTPUTS_DIR = common.OUTPUTS_DIR
TABLES_DIR = common.TABLES_DIR
STAGE2_RESULTS_PATH = TABLES_DIR / "stage2_life_model_search_top_test.csv"
CURRENT_MODEL_DIR = PROJECT_ROOT / "ai_model" / "saved_models" / "current"
MONTHLY_MODEL_DIR = PROJECT_ROOT / "ai_model" / "saved_models" / "monthly_demand" / "current"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
PLOT_DIR = PROJECT_ROOT / "ai_model" / "results" / "plots"
SUMMARY_PATH = PROJECT_ROOT / "ai_model" / "results" / "presentation_metrics_summary.csv"


def setup_korean_plot_style() -> None:
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 120
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def unwrap_estimator(model):
    if hasattr(model, "named_steps"):
        return list(model.named_steps.values())[-1]
    return model


def has_monthly_plot_inputs(path: Path) -> bool:
    return (path / "monthly_model_meta.json").exists() and (path / "monthly_backtest_predictions.csv").exists()


def resolve_monthly_model_dir() -> Path:
    env_path = os.environ.get("MONTHLY_DEMAND_MODEL_DIR")
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if has_monthly_plot_inputs(candidate):
            return candidate

    if has_monthly_plot_inputs(MONTHLY_MODEL_DIR):
        return MONTHLY_MODEL_DIR

    candidates = [
        path
        for pattern in ("*_exp9_lgbm_demand_gs", "*_stage3_monthly_model_search")
        for path in RUNS_DIR.glob(pattern)
        if has_monthly_plot_inputs(path)
    ]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        "월별 수요 모델 입력을 찾지 못했습니다. "
        "run_exp9_lgbm_demand_gs.py를 먼저 실행하거나 "
        "ai_model/saved_models/monthly_demand/current에 monthly_model_meta.json과 "
        "monthly_backtest_predictions.csv를 저장하세요."
    )


def load_current_model_and_meta():
    model = joblib.load(CURRENT_MODEL_DIR / "model.pkl")
    meta = load_json(CURRENT_MODEL_DIR / "model_meta.json")
    return model, meta


def current_stage2_result_row() -> dict:
    _, meta = load_current_model_and_meta()
    metrics = meta["metrics"]
    return {
        "model": meta["model_name"],
        "test_rmse_months": metrics["rmse_months"],
        "test_mae_months": metrics["mae_months"],
        "test_r2": metrics["r2"],
        "test_term_f1": metrics.get("term_f1"),
    }


def load_stage2_results() -> pd.DataFrame:
    if STAGE2_RESULTS_PATH.exists():
        return pd.read_csv(STAGE2_RESULTS_PATH)
    return pd.DataFrame([current_stage2_result_row()])


def plot_model_performance() -> pd.DataFrame:
    results = load_stage2_results()
    order = ["ExtraTrees", "RandomForest", "XGBoost", "CatBoost", "GradientBoosting"]
    if "stage" in results.columns:
        plot_df = results[results["stage"].eq("test") & results["model"].isin(order)].copy()
    else:
        plot_df = results[results["model"].isin(order)].copy()
    if plot_df.empty:
        plot_df = results.copy()

    plot_df["order"] = plot_df["model"].map({name: idx for idx, name in enumerate(order)}).fillna(len(order))
    plot_df = plot_df.sort_values("order")
    rmse_col = "rmse_months" if "rmse_months" in plot_df.columns else "test_rmse_months"
    best_rmse = plot_df[rmse_col].min()
    plot_df["rmse_delta_months"] = plot_df[rmse_col] - best_rmse

    plt.figure(figsize=(10.5, 5.4))
    ax = plt.gca()
    colors = sns.color_palette("Set2", n_colors=len(plot_df))
    bars = ax.bar(plot_df["model"], plot_df["rmse_delta_months"], color=colors, width=0.72)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("모델 성능 비교: 총수명 예측 RMSE", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("모델")
    ax.set_ylabel("Best 대비 RMSE 차이 (개월)")
    for bar, row in zip(bars, plot_df.itertuples(index=False)):
        row_rmse = getattr(row, rmse_col)
        label = f"{row_rmse:.2f}개월"
        if row.rmse_delta_months > 0:
            label += f"\n(+{row.rmse_delta_months:.2f})"
        else:
            label += "\n(best)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.text(
        0.99,
        0.96,
        f"Best RMSE: {best_rmse:.2f}개월",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#555555",
    )
    savefig(PLOT_DIR / "01_model_performance_rmse.png")
    return plot_df


def plot_feature_importance(model, meta) -> pd.DataFrame:
    estimator = unwrap_estimator(model)
    if not hasattr(estimator, "feature_importances_"):
        raise AttributeError("feature_importances_를 찾을 수 없습니다.")

    features = meta["features"]
    fi = pd.DataFrame({"feature": features, "importance": estimator.feature_importances_})
    fi = fi.sort_values("importance", ascending=False).head(10)

    plt.figure(figsize=(10.5, 6))
    ax = sns.barplot(data=fi, x="importance", y="feature", palette="viridis", hue="feature", legend=False)
    ax.set_title("Feature Importance: 수명 예측 영향 변수", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("중요도")
    ax.set_ylabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    savefig(PLOT_DIR / "02_feature_importance.png")
    return fi


def plot_monthly_failure_timeseries() -> pd.DataFrame:
    monthly_dir = resolve_monthly_model_dir()
    monthly_meta = load_json(monthly_dir / "monthly_model_meta.json")
    monthly = pd.read_csv(monthly_dir / "monthly_backtest_predictions.csv")

    pred_candidates = [col for col in monthly.columns if col.endswith("_pred")]
    if not pred_candidates:
        raise ValueError("월별 예측 컬럼을 찾지 못했습니다.")
    pred_col = pred_candidates[0]

    monthly["event_month"] = pd.to_datetime(monthly["event_month"])
    recent = monthly.tail(12).copy()
    source_note = f"{monthly_meta['model_name']} {monthly_meta['variant']}"

    plt.figure(figsize=(11.2, 5.6))
    ax = plt.gca()
    ax.bar(
        recent["event_month"],
        recent["actual_count"],
        width=22,
        alpha=0.62,
        label="실제 고장/처분 수량",
        color="#7FB3D5",
    )
    ax.plot(
        recent["event_month"],
        recent[pred_col],
        marker="o",
        linewidth=2.6,
        label=source_note,
        color="#E67E22",
    )
    ax.set_title("월별 고장 예상 수량: 예측이 조달계획으로 연결되는 흐름", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("월")
    ax.set_ylabel("수량")
    ax.text(
        0.99,
        0.95,
        "최근 12개월 기준",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=35)
    savefig(PLOT_DIR / "03_monthly_failure_forecast.png")
    return recent


def plot_rul_distribution(model, meta) -> dict[str, float]:
    df = pd.read_csv(DATA_PATH)
    predict_df = df[df["학습데이터여부"].eq("N")].copy()
    features = meta["features"]
    predict_df = predict_df.dropna(subset=features + ["운용연차"]).copy()

    pred_total_months = np.asarray(model.predict(predict_df[features]), dtype=float)
    age_months = predict_df["운용연차"].astype(float).to_numpy() * 12
    rul_months = np.clip(pred_total_months - age_months, 0.5, None)

    plot_df = pd.DataFrame({"RUL_개월": rul_months})
    urgent_count = int((plot_df["RUL_개월"] <= 6).sum())
    total_count = len(plot_df)
    urgent_pct = urgent_count / total_count * 100 if total_count else 0.0

    bins = [-np.inf, 6, 12, 24, 60, np.inf]
    labels = ["6개월 이하", "6~12개월", "1~2년", "2~5년", "5년 초과"]
    plot_df["구간"] = pd.cut(plot_df["RUL_개월"], bins=bins, labels=labels, right=True)
    summary = plot_df["구간"].value_counts().reindex(labels).reset_index()
    summary.columns = ["구간", "예상 자산 수"]
    summary["비율"] = summary["예상 자산 수"] / total_count * 100

    plt.figure(figsize=(10.8, 5.6))
    colors = ["#E15759", "#F28E2B", "#EDC948", "#76B7B2", "#59A14F"]
    ax = plt.gca()
    bars = ax.bar(summary["구간"].astype(str), summary["예상 자산 수"], color=colors, width=0.72)
    ax.set_title("RUL 분포: 잔여수명 6개월 이하 자산 규모", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("예측 잔여수명 구간")
    ax.set_ylabel("자산 수")
    bar_labels = [f"{int(row['예상 자산 수']):,}건\n({row['비율']:.1f}%)" for _, row in summary.iterrows()]
    ax.bar_label(bars, labels=bar_labels, padding=4, fontsize=10)
    ax.set_ylim(0, max(summary["예상 자산 수"].max() * 1.22, 1))
    ax.text(
        0.99,
        0.94,
        f"6개월 이하: {urgent_count:,}건 ({urgent_pct:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#DDDDDD"},
    )
    savefig(PLOT_DIR / "04_rul_distribution.png")
    return {"urgent_count": urgent_count, "urgent_pct": urgent_pct, "total_count": total_count}


def write_summary(rul_stats: dict[str, float]) -> None:
    stage2_results = load_stage2_results()
    sort_col = "rmse_months" if "rmse_months" in stage2_results.columns else "test_rmse_months"
    best_stage2 = stage2_results.sort_values(sort_col).iloc[0].to_dict()

    monthly_meta = load_json(resolve_monthly_model_dir() / "monthly_model_meta.json")
    monthly_metrics = monthly_meta["metrics"]
    _, current_meta = load_current_model_and_meta()

    summary = {
        "stage2_best_model": best_stage2.get("model"),
        "stage2_test_rmse_months": best_stage2.get("rmse_months", best_stage2.get("test_rmse_months")),
        "stage2_test_mae_months": best_stage2.get("mae_months", best_stage2.get("test_mae_months")),
        "stage2_test_r2": best_stage2.get("r2", best_stage2.get("test_r2")),
        "stage2_term_f1": best_stage2.get("term_f1", best_stage2.get("test_term_f1")),
        "stage3_best_monthly_model": monthly_meta.get("model_name"),
        "stage3_monthly_variant": monthly_meta.get("variant"),
        "stage3_monthly_rmse_count": monthly_metrics.get("monthly_rmse_count"),
        "stage3_monthly_mae_count": monthly_metrics.get("monthly_mae_count"),
        "stage3_monthly_r2": monthly_metrics.get("monthly_r2"),
        "current_model_feature_count": len(current_meta["features"]),
        "rul_urgent_count": rul_stats["urgent_count"],
        "rul_urgent_pct": rul_stats["urgent_pct"],
        "rul_total_count": rul_stats["total_count"],
    }
    pd.DataFrame([summary]).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    setup_korean_plot_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    model, meta = load_current_model_and_meta()
    plot_model_performance()
    plot_feature_importance(model, meta)
    plot_monthly_failure_timeseries()
    rul_stats = plot_rul_distribution(model, meta)
    write_summary(rul_stats)

    print("Exp9 LightGBM 기준 발표용 그래프와 요약 지표를 생성했습니다.")
    print(f"- {PLOT_DIR / '01_model_performance_rmse.png'}")
    print(f"- {PLOT_DIR / '02_feature_importance.png'}")
    print(f"- {PLOT_DIR / '03_monthly_failure_forecast.png'}")
    print(f"- {PLOT_DIR / '04_rul_distribution.png'}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
