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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common

CURRENT_RUN_DIR = common.RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp6_lgbm_opt"
RESULT_PATH = common.TABLES_DIR / "exp6_lgbm_opt_trials.csv"
TOP_TEST_PATH = common.TABLES_DIR / "exp6_lgbm_opt_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
N_TRIALS = 30
TARGET_COL = "실제수명_개월"
FEATURES = common.LIFE_FEATURES  # [변인 변경] 35개 전체 피처 다양하게 활용

def evaluate_life_model(y_true, y_pred, age_months) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    actual_rul = y_true - age_months
    pred_rul = y_pred - age_months
    y_true_bin = (actual_rul <= 6).astype(int)
    y_pred_bin = (pred_rul <= 6).astype(int)
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    return {"rmse": rmse, "mae": mae, "r2_score": r2, "f1_score": f1}

def main() -> None:
    start_time = time.perf_counter()
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 6] LightGBM 전체 피처 풀 활용 + Optuna 최적화")
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
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        metrics = evaluate_life_model(y_valid, pred_valid, age_valid)
        return metrics["rmse"]

    print(f" Optuna 하이퍼파라미터 튜닝 중 ({N_TRIALS} Trials)...")
    
    # Optuna 자체의 탐색 경로(시드)를 고정하기 위해 TPESampler 추가
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[lambda s, t: print(f" ▹ [Trial {t.number:02d}] Valid RMSE: {t.value:.4f} | Best: {s.best_value:.4f}")])

    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
    
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X_train_valid, y_train_valid)
    pred_test = final_model.predict(X_test)
    test_metrics = evaluate_life_model(y_test, pred_test, age_test)
    
    study.trials_dataframe().to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    
    # 최종 CSV 결과에 R2 점수 추가 (평가 함수에서 뱉는 키값인 "r2_score" 사용)
    final_row = {
        "model": "LGBM_OptFeatures_Optuna", "feature_count": len(FEATURES),
        "test_rmse": test_metrics["rmse"], "test_mae": test_metrics["mae"], 
        "test_r2": test_metrics["r2_score"],
        "test_f1": test_metrics["f1_score"]
    }
    pd.DataFrame([final_row]).to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print(" [실험 6 최종 결과] ")
    print(f"▶ 최적 파라미터 : {study.best_params}")
    print(f"▶ Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"▶ Test MAE  : {test_metrics['mae']:.4f}")
    print(f"▶ Test R²   : {test_metrics['r2_score']:.4f}")  # 🌟 [수정 3] 터미널 출력 화면에 R2 추가
    print(f"▶ 임박자산 F1: {test_metrics['f1_score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()