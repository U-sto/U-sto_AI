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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common

CURRENT_RUN_DIR = common.RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp8_xgb_fixed_optuna"
RESULT_PATH = common.TABLES_DIR / "exp8_xgb_fixed_optuna_trials.csv"
TOP_TEST_PATH = common.TABLES_DIR / "exp8_xgb_fixed_optuna_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
N_TRIALS = 30
TARGET_COL = "실제수명_개월"
FEATURES = [
    "내용연수", "취득금액", "부서가혹도", "가격민감도", "장비중요도",
    "리드타임등급", "취득월", "구매배치수량", "동일배치자산수", "월평균사용시간",
    "주당사용일수", "누적점검수리횟수", "누적수리횟수", "최근1년수리횟수", "누적수리비용"
]

def evaluate_life_model(y_true, y_pred, age_months) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) # 🌟 R2 추가
    actual_rul = y_true - age_months
    pred_rul = y_pred - age_months
    y_true_bin = (actual_rul <= 6).astype(int)
    y_pred_bin = (pred_rul <= 6).astype(int)
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    return {"rmse": rmse, "mae": mae, "r2_score": r2, "f1_score": f1} # 🌟 반환값에 R2 추가

def main() -> None:
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 8] XGBoost 고정 15개 피처 + Optuna 최적화")
    print("=" * 60)

    train, valid, test, _ = common.prepare_life_data()
    
    X_train, y_train = train[FEATURES], train[TARGET_COL].to_numpy()
    X_valid, y_valid = valid[FEATURES], valid[TARGET_COL].to_numpy()
    age_valid = valid["운용연차_개월"].to_numpy()
    
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[FEATURES]
    y_train_valid = train_valid[TARGET_COL].to_numpy()
    
    X_test, y_test = test[FEATURES], test[TARGET_COL].to_numpy()
    age_test = test["운용연차_개월"].to_numpy()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "eval_metric": "rmse"
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics = evaluate_life_model(y_valid, pred_valid, age_valid)
        return metrics["rmse"]

    print(f" Optuna 하이퍼파라미터 튜닝 중 ({N_TRIALS} Trials)...")
    
    # Optuna 시드 고정
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[lambda s, t: print(f" ▹ [Trial {t.number:02d}] Valid RMSE: {t.value:.4f} | Best: {s.best_value:.4f}")])

    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "eval_metric": "rmse"})
    
    # 통합 학습 후 최종 Test 평가
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train_valid, y_train_valid)
    pred_test = final_model.predict(X_test)
    test_metrics = evaluate_life_model(y_test, pred_test, age_test)
    
    # 결과 저장
    study.trials_dataframe().to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    final_row = {
        "model": "XGB_Fixed15_Optuna", "feature_count": len(FEATURES),
        "test_rmse": test_metrics["rmse"], "test_mae": test_metrics["mae"], 
        "test_r2": test_metrics["r2_score"], # CSV 저장에 R2 추가
        "test_f1": test_metrics["f1_score"]
    }
    pd.DataFrame([final_row]).to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print(" [실험 8 최종 결과] ")
    print(f"▶ 최적 파라미터 : {study.best_params}")
    print(f"▶ Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"▶ Test MAE  : {test_metrics['mae']:.4f}")
    print(f"▶ Test R²   : {test_metrics['r2_score']:.4f}") # 터미널 출력에 R2 추가
    print(f"▶ 임박자산 F1: {test_metrics['f1_score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()