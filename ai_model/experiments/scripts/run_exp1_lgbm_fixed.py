from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, r2_score

# ---------------------------------------------------------
# [1] 경로 및 환경 설정 (서현님 전용 experiments_sh 폴더에 쏙 들어가도록 수정!)
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import modeling_common as common

PROJECT_ROOT = common.PROJECT_ROOT

# 핵심 수정: common.py에 고정된 경로를 무시하고, 현재 파일이 있는 폴더(experiments_sh)를 기준점으로 덮어씁니다.
EXPERIMENTS_DIR = SCRIPT_DIR
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
RUN_DIR = EXPERIMENTS_DIR / "runs" 

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = RUN_DIR / f"{RUN_ID}_exp1_lgbm_fixed"
RESULT_PATH = TABLES_DIR / "exp1_lgbm_fixed_results.csv"
TOP_TEST_PATH = TABLES_DIR / "exp1_lgbm_fixed_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
TERM_MONTHS = 6

# [피처 고정] 15개 피처 리스트
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

def lgbm_model_specs() -> list[dict]:
    return [
        {
            "model": "LightGBM",
            "variant": "lgbm_balanced",
            "params": {"n_estimators": 700, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "subsample": 0.9}
        },
        {
            "model": "LightGBM",
            "variant": "lgbm_shallow_regularized",
            "params": {"n_estimators": 900, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 15, "subsample": 0.85}
        },
        {
            "model": "LightGBM",
            "variant": "lgbm_fast_shallow",
            "params": {"n_estimators": 500, "learning_rate": 0.06, "max_depth": 5, "num_leaves": 31, "subsample": 0.95}
        },
        {
            "model": "LightGBM",
            "variant": "lgbm_deeper_slow",
            "params": {"n_estimators": 800, "learning_rate": 0.035, "max_depth": 7, "num_leaves": 63, "subsample": 0.85}
        }
    ]

# ---------------------------------------------------------
# [3] 메인 파이프라인
# ---------------------------------------------------------
def main() -> None:
    start_time = time.perf_counter()
    
    # 폴더 생성 로직 (없으면 안전하게 알아서 생성합니다)
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 1] LightGBM 고정 15개 피처 성능 평가")
    print(f"Run ID: {RUN_ID}")
    print("=" * 60)

    train, valid, test, _ = common.prepare_life_data()
    target_col, age_col = infer_target_columns(train)
    
    X_train, y_train = train[FEATURES], train[target_col].to_numpy()
    X_valid, y_valid = valid[FEATURES], valid[target_col].to_numpy()
    
    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[FEATURES]
    y_train_valid = train_valid[target_col].to_numpy()
    
    X_test, y_test = test[FEATURES], test[target_col].to_numpy()

    results = []
    specs = lgbm_model_specs()
    
    print(f"\n [Step 1] Valid 데이터를 기반으로 최적의 LightGBM 스펙 탐색 중...")
    for spec in specs:
        model = LGBMRegressor(
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            verbose=-1, 
            **spec["params"]
        )
        
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        pred_valid = model.predict(X_valid)
        elapsed = time.perf_counter() - t0
        
        metrics = evaluate(y_valid, pred_valid, valid[age_col])
        
        res_row = {
            "model": spec["model"],
            "variant": spec["variant"],
            "params_json": json.dumps(spec["params"]),
            "valid_rmse_months": metrics["rmse_months"],
            "valid_mae_months": metrics["mae_months"],
            "valid_r2": metrics["r2"],
            "valid_term_f1": metrics["term_f1"],
            "elapsed_sec": elapsed
        }
        results.append(res_row)
        print(f" ▹ [{spec['variant']}] Valid RMSE: {metrics['rmse_months']:.4f} 개월 ({elapsed:.2f}초)")

    valid_df = pd.DataFrame(results).sort_values("valid_rmse_months")
    valid_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    
    best_spec_row = valid_df.iloc[0]
    print(f"\n Valid 기준 1위 스펙: {best_spec_row['variant']} (RMSE: {best_spec_row['valid_rmse_months']:.4f})")

    print(f"\n [Step 2] 1위 스펙으로 [Train+Valid] 통합 학습 후 최종 Test 평가...")
    best_params = json.loads(best_spec_row["params_json"])
    
    final_model = LGBMRegressor(
        random_state=RANDOM_STATE, 
        n_jobs=-1, 
        verbose=-1, 
        **best_params
    )
    
    final_model.fit(X_train_valid, y_train_valid)
    pred_test = final_model.predict(X_test)
    
    test_metrics = evaluate(y_test, pred_test, test[age_col])
    
    # 결과 저장
    final_row = dict(best_spec_row)
    final_row.update({
        "test_rmse_months": test_metrics["rmse_months"],
        "test_mae_months": test_metrics["mae_months"],
        "test_r2": test_metrics["r2"],
        "test_term_precision": test_metrics["term_precision"],
        "test_term_recall": test_metrics["term_recall"],
        "test_term_f1": test_metrics["term_f1"]
    })
    
    final_df = pd.DataFrame([final_row])
    final_df.to_csv(TOP_TEST_PATH, index=False, encoding="utf-8-sig")

    report = {
        "run_id": RUN_ID,
        "features_used": FEATURES,
        "feature_count": len(FEATURES),
        "best_variant": final_row["variant"],
        "test_metrics": test_metrics
    }
    with open(CURRENT_RUN_DIR / "exp1_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_for_json(report), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(" [실험 1 최종 스코어보드] ")
    print("-" * 60)
    print(f"▶ 사용 모델    : LightGBM ({final_row['variant']})")
    print(f"▶ 피처 조건    : 지정된 15개 변수 고정")
    print(f"▶ Test RMSE    : {test_metrics['rmse_months']:.4f} 개월")
    print(f"▶ Test MAE     : {test_metrics['mae_months']:.4f} 개월")
    print(f"▶ Test R²      : {test_metrics['r2']:.4f}")
    print(f"▶ 임박 자산 F1 : {test_metrics['term_f1']:.4f}")
    print("=" * 60)
    print(f"결과가 서현님의 experiments_sh/outputs/tables 폴더에 쏙 들어갔습니다!")

if __name__ == "__main__":
    main()