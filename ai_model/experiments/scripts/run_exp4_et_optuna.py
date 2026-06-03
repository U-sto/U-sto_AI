from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Optuna 로그 출력 최소화
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# [1] 경로 및 환경 설정
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common

PROJECT_ROOT = common.PROJECT_ROOT
EXPERIMENTS_DIR = SCRIPT_DIR
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
RUN_DIR = EXPERIMENTS_DIR / "runs"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = RUN_DIR / f"{RUN_ID}_exp4_et_optuna"
RESULT_PATH = TABLES_DIR / "exp4_et_optuna_trials.csv"
TOP_TEST_PATH = TABLES_DIR / "exp4_et_optuna_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
N_TRIALS = 30  # Optuna가 탐색할 총 조합의 수

# [통제 변인] 월별 수요 예측 핵심 7개 피처
FEATURES = [
    "trend", "month", "month_sin", "month_cos", 
    "lag_12", "rolling_mean_6", "rolling_std_6"
]
TARGET_COL = "actual_count"  # modeling_common.py의 규격에 따름

# ---------------------------------------------------------
# [2] 유틸리티 및 시계열 평가 함수
# ---------------------------------------------------------
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

def evaluate_ts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
    }

# ---------------------------------------------------------
# [3] 메인 파이프라인
# ---------------------------------------------------------
def main() -> None:
    start_time = time.perf_counter()
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 4] ExtraTrees 고정 7개 피처 + Optuna 월별 수요 최적화")
    print(f"Run ID: {RUN_ID}")
    print("=" * 60)

    # 데이터 준비
    print(" 원본 데이터를 로드하고 시계열 특징량을 생성하는 중...")
    raw_df = pd.read_csv(common.DATA_PATH)
    
    # 1. 월별 이벤트 데이터 집계
    monthly_base = common.build_monthly_series(raw_df)
    # 2. Lag 및 Rolling 피처 추가
    monthly_features = common.add_lag_features(monthly_base)
    
    # [중요] lag_12 특성상 초반 12개월은 과거 데이터가 없어 NaN이 발생하므로 안전하게 제거
    cleaned_monthly = monthly_features.dropna(subset=FEATURES + [TARGET_COL]).copy()
    
    # 3. 시계열 순서에 따른 Train / Valid / Test 분할
    train, valid, test = common.split_monthly(cleaned_monthly)
    
    # 모델에 입력할 무대 세팅
    X_train, y_train = train[FEATURES], train[TARGET_COL].to_numpy()
    X_valid, y_valid = valid[FEATURES], valid[TARGET_COL].to_numpy()
    
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[FEATURES]
    y_train_valid = train_valid[TARGET_COL].to_numpy()
    
    X_test, y_test = test[FEATURES], test[TARGET_COL].to_numpy()

    # Optuna 목적 함수 (ExtraTrees 맞춤형 하이퍼파라미터 튜닝)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        
        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics = evaluate_ts(y_valid, pred_valid)
        
        # 시계열 검증용 RMSE를 최소화하는 방향으로 유도
        return metrics["rmse"]

    print(f"\n [Step 1] Optuna가 {N_TRIALS}번의 탐색을 통해 ExtraTrees 최적 세팅을 찾는 중...")
    
    study = optuna.create_study(direction="minimize")
    
    def print_callback(study, trial):
        print(f" ▹ [Trial {trial.number:02d}] Valid RMSE: {trial.value:.4f} | Best: {study.best_value:.4f}")
        
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])

    best_params = study.best_params
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    
    print(f"\n [Step 1 완료] Optuna 탐색 완료!")
    print(f"  - 최고 Valid RMSE: {study.best_value:.4f}")
    print(f"  - 찾은 최적 파라미터: {study.best_params}")

    # 전체 시도 기록 저장
    trials_df = study.trials_dataframe()
    trials_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n [Step 2] 최적 파라미터로 [Train+Valid] 통합 학습 후 최종 Test 평가...")
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_train_valid, y_train_valid)
    pred_test = final_model.predict(X_test)
    
    test_metrics = evaluate_ts(y_test, pred_test)
    elapsed_total = time.perf_counter() - start_time
    
    # 최종 결과 스코어 저장
    final_row = {
        "model": "ExtraTrees_Optuna",
        "feature_count": len(FEATURES),
        "optuna_trials": N_TRIALS,
        "best_params_json": json.dumps(study.best_params),
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "elapsed_sec": elapsed_total
    }
    pd.DataFrame([final_row]).to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    
    report = {
        "run_id": RUN_ID,
        "features_used": FEATURES,
        "best_params": study.best_params,
        "test_metrics": test_metrics
    }
    with open(CURRENT_RUN_DIR / "exp4_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2)

    # 최종 결과 출력
    print("\n" + "=" * 60)
    print(" [실험 4 최종 스코어보드 - 월별 수요 모델] ")
    print("-" * 60)
    print(f"▶ 사용 모델 : ExtraTrees (Optuna 최적화)")
    print(f"▶ 사용 피처 : 시계열 고정 7개 변수")
    print(f"▶ 탐색 횟수 : {N_TRIALS} Trials")
    print(f"▶ Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"▶ Test MAE  : {test_metrics['mae']:.4f}")
    print(f"▶ Test R²   : {test_metrics['r2']:.4f}")
    print("=" * 60)
    print(f" 기존 Stage 3(팀장님의 기존 월별 수요 모델) 기록을 가뿐히 넘겼는지 확인해 보세요!")

if __name__ == "__main__":
    main()