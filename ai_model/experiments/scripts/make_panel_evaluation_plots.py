from __future__ import annotations

import json
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
CURRENT_MODEL_DIR = PROJECT_ROOT / "ai_model" / "saved_models" / "current"
MONTHLY_RUN_DIR = PROJECT_ROOT / "ai_model" / "experiments" / "runs" / "20260525_003630_stage3_monthly_model_search"
PLOT_DIR = PROJECT_ROOT / "ai_model" / "results" / "plots"


def setup_korean_plot_style() -> None:
    mpl.rcParams["font.family"] = "Malgun Gothic"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.dpi"] = 130
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_life_model_evaluation_plot() -> Path:
    model = joblib.load(CURRENT_MODEL_DIR / "model.pkl")
    meta = load_json(CURRENT_MODEL_DIR / "model_meta.json")
    features = meta["features"]
    metrics = meta["metrics"]

    _, _, test, _ = common.prepare_life_data()
    y_true = test["실제수명_개월"].astype(float).to_numpy()
    y_pred = np.asarray(model.predict(test[features]), dtype=float)
    error = np.abs(y_pred - y_true)

    plot_df = pd.DataFrame(
        {
            "실제 총수명(개월)": y_true,
            "예측 총수명(개월)": y_pred,
            "절대오차(개월)": error,
        }
    )

    lim_min = max(0, min(y_true.min(), y_pred.min()) - 8)
    lim_max = max(y_true.max(), y_pred.max()) + 8

    plt.figure(figsize=(8.4, 6.4))
    ax = plt.gca()
    scatter = ax.scatter(
        plot_df["실제 총수명(개월)"],
        plot_df["예측 총수명(개월)"],
        c=plot_df["절대오차(개월)"],
        cmap="viridis_r",
        s=30,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#D62728", linewidth=2.0, linestyle="--", label="완전 예측선")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_title("자산 수명 모델 평가", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("실제 총수명 (개월)", fontsize=12)
    ax.set_ylabel("예측 총수명 (개월)", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.22)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("절대오차 (개월)")

    text = (
        f"CatBoost / 중요도 상위 15개 feature\n"
        f"RMSE {metrics['rmse_months']:.2f}개월  |  MAE {metrics['mae_months']:.2f}개월\n"
        f"R² {metrics['r2']:.3f}  |  Test {len(test):,}건"
    )
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#DADADA", "alpha": 0.95},
    )

    output_path = PLOT_DIR / "05_life_model_evaluation.png"
    savefig(output_path)
    return output_path


def make_monthly_model_evaluation_plot() -> Path:
    meta = load_json(MONTHLY_RUN_DIR / "monthly_model_meta.json")
    metrics = meta["metrics"]
    monthly = pd.read_csv(MONTHLY_RUN_DIR / "monthly_backtest_predictions.csv")
    monthly["event_month"] = pd.to_datetime(monthly["event_month"])

    pred_cols = [col for col in monthly.columns if col.endswith("_pred")]
    if not pred_cols:
        raise ValueError("monthly_backtest_predictions.csv에서 예측 컬럼을 찾지 못했습니다.")
    pred_col = pred_cols[0]

    plt.figure(figsize=(9.2, 6.0))
    ax = plt.gca()
    ax.bar(
        monthly["event_month"],
        monthly["actual_count"],
        width=23,
        color="#7FB3D5",
        alpha=0.68,
        label="실제 고장/처분 수량",
    )
    ax.plot(
        monthly["event_month"],
        monthly[pred_col],
        marker="o",
        markersize=6,
        linewidth=2.7,
        color="#E67E22",
        label="월별 수요 모델 예측",
    )
    ax.set_title("월별 수요 모델 평가", fontsize=18, weight="bold", pad=14)
    ax.set_xlabel("Test 기간 월", fontsize=12)
    ax.set_ylabel("고장/처분 수량", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.22)

    text = (
        f"ExtraTrees / 계절성 중심 7개 feature\n"
        f"RMSE {metrics['monthly_rmse_count']:.2f}건  |  MAE {metrics['monthly_mae_count']:.2f}건\n"
        f"R² {metrics['monthly_r2']:.3f}  |  Recursive backtest"
    )
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#DADADA", "alpha": 0.95},
    )

    output_path = PLOT_DIR / "06_monthly_demand_model_evaluation.png"
    savefig(output_path)
    return output_path


def main() -> None:
    setup_korean_plot_style()
    outputs = [
        make_life_model_evaluation_plot(),
        make_monthly_model_evaluation_plot(),
    ]
    print("패널용 평가 그래프를 생성했습니다.")
    for output in outputs:
        print(f"- {output}")


if __name__ == "__main__":
    main()
