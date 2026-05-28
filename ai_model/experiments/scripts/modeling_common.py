from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "dataset" / "create_data" / "data_ml" / "phase4_training_data.csv"
EXPERIMENTS_DIR = PROJECT_ROOT / "ai_model" / "experiments"
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RUNS_DIR = EXPERIMENTS_DIR / "runs"
RANDOM_STATE = 42

LIFE_FEATURES = [
    "내용연수",
    "취득금액",
    "부서가혹도",
    "가격민감도",
    "장비중요도",
    "리드타임등급",
    "취득월",
    "구매배치수량",
    "대량구매여부",
    "동일배치자산수",
    "월평균사용시간",
    "주당사용일수",
    "공용장비여부",
    "수업사용여부",
    "사용강도지수",
    "누적점검수리횟수",
    "누적수리횟수",
    "최근1년수리횟수",
    "최근2년수리횟수",
    "마지막수리후경과개월",
    "누적수리비용",
    "취득금액대비수리비율",
    "최대장애심각도",
    "교체권고횟수",
    "취득학기구분_Code",
    "신학기준비월여부",
    "예산집행월여부",
    "방학기간여부",
    "부서예산등급_Code",
    "부서교체성향",
    "부서관리성향",
    "G2B목록명_Code",
    "물품분류명_Code",
    "운용부서코드_Code",
    "캠퍼스_Code",
]

LEAKAGE_COLUMNS = {
    "불용일자",
    "처분방식",
    "물품상태",
    "불용사유",
    "실제수명",
    "실제잔여수명",
    "예측잔여수명",
    "AI예측고장일",
    "권장발주일",
    "데이터세트구분",
    "학습데이터여부",
    "동일배치운용연차평균",
    "(월별)고장예상수량",
    "안전재고",
    "(월별)필요수량",
    "예측실행일자",
}


def prepare_life_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    missing = [col for col in LIFE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"누락된 feature 컬럼: {missing}")

    leaked = sorted(set(LIFE_FEATURES) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"feature list에 누수 위험 컬럼이 포함됨: {leaked}")

    for col in LIFE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    train = df[df["데이터세트구분"].eq("Train") & df["학습데이터여부"].eq("Y")].copy()
    valid = df[df["데이터세트구분"].eq("Valid") & df["학습데이터여부"].eq("Y")].copy()
    test = df[df["데이터세트구분"].eq("Test") & df["학습데이터여부"].eq("Y")].copy()
    pred = df[df["데이터세트구분"].eq("Prediction") & df["학습데이터여부"].eq("N")].copy()

    fill_values = train[LIFE_FEATURES].median(numeric_only=True)
    for part in [train, valid, test, pred]:
        part[LIFE_FEATURES] = part[LIFE_FEATURES].fillna(fill_values).fillna(0)
        part["실제수명_개월"] = pd.to_numeric(part["실제수명"], errors="coerce") * 12
        part["운용연차_개월"] = pd.to_numeric(part["운용연차"], errors="coerce") * 12

    return train, valid, test, pred


def make_event_date(df: pd.DataFrame) -> pd.Series:
    disuse_date = pd.to_datetime(df["불용일자"], errors="coerce")
    fallback_date = pd.to_datetime(df["취득일자"], errors="coerce") + pd.to_timedelta(
        (pd.to_numeric(df["실제수명"], errors="coerce") * 365.25).round().astype("Int64"),
        unit="D",
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
