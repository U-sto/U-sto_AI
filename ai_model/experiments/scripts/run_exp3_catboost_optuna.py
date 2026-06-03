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
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, r2_score

# Optuna 로그 출력 최소화 (너무 많은 로그가 뜨지 않도록 조절)
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
CURRENT_RUN_DIR = RUN_DIR / f"{RUN_ID}_exp3_catboost_optuna"
RESULT_PATH = TABLES_DIR / "exp3_catboost_optuna_trials.csv"
TOP_TEST_PATH = TABLES_DIR / "exp3_catboost_optuna_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
TERM_MONTHS = 6
N_TRIALS = 30  # Optuna가 탐색할 총 조합의 수 (원하면 더 늘려도 됩니다!)

# [통제 변인] 15개 피처 고정
FEATURES = [
    "내용연수", "부서가혹도", "월평균사용시간", "사용강도지수", "누적점검수리횟수",
    "누적수리횟수", "최근2년수리횟수", "마지막수리후경과개월", "취득금액대비수리비율",
    "최대장애심각도", "부서예산등급_Code", "부서교체성향", "G2B목록명_Code",
    "물품분류명_Code", "운용부서코드_Code"
]

# ---------------------------------------------------------
# [2] 유틸리티 및 평가 함수
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

def infer_target_columns(train: pd.DataFrame) -> tuple[str, str]:
    raw_cols = set(pd.read_csv(common.DATA_PATH, nrows=0).columns)
    added_cols = [col for col in train.columns if col not in raw_cols]
    if len(added_cols) < 2:
        raise ValueError(f"Could not infer target columns. Added columns: {added_cols}")
    return added_cols[0], added_cols[1]

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
        "rmse_months": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae_months": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan,
        "term_precision": float(precision),
        "term_recall": float(recall),
        "term_f1": float(f1),
    }

# ---------------------------------------------------------
# [3] 메인 파이프라인
# ---------------------------------------------------------
def main() -> None:
    start_time = time.perf_counter()
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 3] CatBoost 고정 15개 피처 + Optuna 베이지안 최적화")
    print(f"Run ID: {RUN_ID}")
    print("=" * 60)

    # 데이터 준비
    train, valid, test, _ = common.prepare_life_data()
    target_col, age_col = infer_target_columns(train)
    
    X_train, y_train = train[FEATURES], train[target_col].to_numpy()
    X_valid, y_valid = valid[FEATURES], valid[target_col].to_numpy()
    
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[FEATURES]
    y_train_valid = train_valid[target_col].to_numpy()
    
    X_test, y_test = test[FEATURES], test[target_col].to_numpy()

    # Optuna 목적 함수 (Objective Function) 정의
    def objective(trial):
        # CatBoost가 튜닝할 파라미터의 범위를 자유롭게 지정
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed": RANDOM_STATE,
            "verbose": False # 개별 학습 로그 숨김
        }
        
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics = evaluate(y_valid, pred_valid, valid[age_col])
        
        # Valid 데이터의 RMSE를 최소화하는 방향으로 Optuna가 학습
        return metrics["rmse_months"]

    print(f" [Step 1] Optuna가 {N_TRIALS}번의 탐색을 통해 최적의 파라미터를 찾는 중입니다...")
    print(f"  (베이지안 최적화가 진행되며 점차 오차를 줄여나갑니다. 잠시만 기다려주세요!)")
    
    # Optuna Study 생성 및 실행
    study = optuna.create_study(direction="minimize")
    
    # 진행 상황을 깔끔하게 보여주기 위한 커스텀 콜백
    def print_callback(study, trial):
        print(f" ▹ [Trial {trial.number:02d}] Valid RMSE: {trial.value:.4f} 개월 | Best: {study.best_value:.4f}")
        
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])

    best_params = study.best_params
    best_params["random_seed"] = RANDOM_STATE
    best_params["verbose"] = False
    
    print(f"\n [Step 1 완료] Optuna 탐색 완료!")
    print(f"  - 최고 Valid RMSE: {study.best_value:.4f}")
    print(f"  - 찾은 최적 파라미터: {best_params}")

    # 전체 Trial 기록 저장
    trials_df = study.trials_dataframe()
    trials_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n [Step 2] Optuna가 찾은 최적 파라미터로 [Train+Valid] 통합 학습 후 최종 Test 평가...")
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train_valid, y_train_valid)
    pred_test = final_model.predict(X_test)
    
    test_metrics = evaluate(y_test, pred_test, test[age_col])
    elapsed_total = time.perf_counter() - start_time
    
    # 결과 저장
    final_row = {
        "model": "CatBoost_Optuna",
        "feature_count": 15,
        "optuna_trials": N_TRIALS,
        "best_params_json": json.dumps(best_params),
        "test_rmse_months": test_metrics["rmse_months"],
        "test_mae_months": test_metrics["mae_months"],
        "test_r2": test_metrics["r2"],
        "test_term_f1": test_metrics["term_f1"],
        "elapsed_sec": elapsed_total
    }
    final_df = pd.DataFrame([final_row])
    final_df.to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    
    report = {
        "run_id": RUN_ID,
        "features_used": FEATURES,
        "best_params": best_params,
        "test_metrics": test_metrics
    }
    with open(CURRENT_RUN_DIR / "exp3_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2)

    # 최종 스코어보드
    print("\n" + "=" * 60)
    print(" [실험 3 최종 스코어보드] ")
    print("-" * 60)
    print(f"▶ 사용 모델    : CatBoost (Optuna 최적화)")
    print(f"▶ 피처 조건    : 지정된 15개 변수 고정")
    print(f"▶ 탐색 횟수    : {N_TRIALS} Trials")
    print(f"▶ Test RMSE    : {test_metrics['rmse_months']:.4f} 개월")
    print(f"▶ Test MAE     : {test_metrics['mae_months']:.4f} 개월")
    print(f"▶ Test R²      : {test_metrics['r2']:.4f}")
    print(f"▶ 임박 자산 F1 : {test_metrics['term_f1']:.4f}")
    print("=" * 60)
    print(f" 팀장님 기록 (13.2765개월)을 돌파했는지 확인해 보세요!")

if __name__ == "__main__":
    main()