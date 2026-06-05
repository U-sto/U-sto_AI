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
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, r2_score # 🌟 r2_score 추가

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# [1] 경로 및 환경 설정
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common

CURRENT_RUN_DIR = common.RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_exp7_xgb_fixed_gs"
TOP_TEST_PATH = common.TABLES_DIR / "exp7_xgb_fixed_gs_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
TARGET_COL = "실제수명_개월"

# [통제 변인] 고정 15개 피처
FEATURES = [
    "내용연수", "취득금액", "부서가혹도", "가격민감도", "장비중요도",
    "리드타임등급", "취득월", "구매배치수량", "동일배치자산수", "월평균사용시간",
    "주당사용일수", "누적점검수리횟수", "누적수리횟수", "최근1년수리횟수", "누적수리비용"
]

# ---------------------------------------------------------
# [2] 자산 수명 모델 전용 평가 함수
# ---------------------------------------------------------
def evaluate_life_model(y_true, y_pred, age_months) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) # 🌟 R2 계산 로직 추가
    
    actual_rul = y_true - age_months
    pred_rul = y_pred - age_months
    
    y_true_bin = (actual_rul <= 6).astype(int)
    y_pred_bin = (pred_rul <= 6).astype(int)
    
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    return {"rmse": rmse, "mae": mae, "r2_score": r2, "f1_score": f1} # 🌟 반환값에 R2 추가

# ---------------------------------------------------------
# [3] 메인 파이프라인
# ---------------------------------------------------------
def main() -> None:
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    common.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 7] XGBoost 고정 15개 피처 + GridSearchCV 최적화")
    print("=" * 60)

    train, valid, test, _ = common.prepare_life_data()
    
    # GridSearchCV를 위해 Train과 Valid를 합쳐서 검증 진행
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[FEATURES]
    y_train_valid = train_valid[TARGET_COL].to_numpy()
    
    X_test, y_test = test[FEATURES], test[TARGET_COL].to_numpy()
    age_test = test["운용연차_개월"].to_numpy()

    # GridSearchCV 탐색 그리드 설정
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8]
    }
    
    print(" GridSearchCV 기본 격자 탐색 중...")
    xgb_base = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="rmse")
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_valid, y_train_valid)
    
    best_params = grid_search.best_params_
    print(f" ▹ [GridSearch 완료] 최적 파라미터 찾음: {best_params}")
    
    # 최종 테스트 데이터 평가
    final_model = grid_search.best_estimator_
    pred_test = final_model.predict(X_test)
    test_metrics = evaluate_life_model(y_test, pred_test, age_test)
    
    # 결과 저장
    final_row = {
        "model": "XGB_Fixed15_GridSearch", "feature_count": len(FEATURES),
        "test_rmse": test_metrics["rmse"], "test_mae": test_metrics["mae"], 
        "test_r2": test_metrics["r2_score"], # CSV 저장에 R2 추가
        "test_f1": test_metrics["f1_score"]
    }
    pd.DataFrame([final_row]).to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print(" [실험 7 최종 결과] ")
    print(f"▶ 최적 파라미터 : {best_params}")
    print(f"▶ Test RMSE : {test_metrics['rmse']:.4f}")
    print(f"▶ Test MAE  : {test_metrics['mae']:.4f}")
    print(f"▶ Test R²   : {test_metrics['r2_score']:.4f}") # 터미널 출력에 R2 추가
    print(f"▶ 임박자산 F1: {test_metrics['f1_score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()