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
CURRENT_RUN_DIR = RUN_DIR / f"{RUN_ID}_exp2_lgbm_search"
RESULT_PATH = TABLES_DIR / "exp2_lgbm_search_results.csv"
TOP_TEST_PATH = TABLES_DIR / "exp2_lgbm_search_top_test.csv"

RANDOM_STATE = common.RANDOM_STATE
TERM_MONTHS = 6

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
        {"model": "LightGBM", "variant": "lgbm_balanced", "params": {"n_estimators": 700, "learning_rate": 0.04, "max_depth": 6, "num_leaves": 31, "subsample": 0.9}},
        {"model": "LightGBM", "variant": "lgbm_shallow_regularized", "params": {"n_estimators": 900, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 15, "subsample": 0.85}},
        {"model": "LightGBM", "variant": "lgbm_fast_shallow", "params": {"n_estimators": 500, "learning_rate": 0.06, "max_depth": 5, "num_leaves": 31, "subsample": 0.95}},
        {"model": "LightGBM", "variant": "lgbm_deeper_slow", "params": {"n_estimators": 800, "learning_rate": 0.035, "max_depth": 7, "num_leaves": 63, "subsample": 0.85}}
    ]

def ordered_subset(features: list[str], indexes: list[int]) -> list[str]:
    return [features[i] for i in indexes if 0 <= i < len(features)]

# 핵심: LightGBM 맞춤형 피처 세트 탐색 함수
def build_feature_sets(train: pd.DataFrame, target_col: str) -> dict[str, list[str]]:
    features = list(common.LIFE_FEATURES)
    
    # 도메인 기반 고정 피처셋 유지
    feature_sets = {
        "full_35": features,
        "no_category_codes_31": features[:31],
        "asset_usage_maintenance_25": ordered_subset(features, list(range(0, 7)) + list(range(10, 24)) + [25, 26, 27, 29, 30]),
        "compact_domain_20": ordered_subset(features, [0, 1, 2, 3, 4, 5, 6, 10, 14, 16, 17, 19, 21, 22, 23, 25, 26, 27, 29, 30]),
        "simple_asset_11": ordered_subset(features, list(range(0, 7)) + list(range(31, 35))),
        "no_maintenance_26": ordered_subset(features, list(range(0, 15)) + list(range(24, 35))),
    }

    # 2번 실험의 차별점: ExtraTrees가 아닌 LightGBM이 직접 피처 중요도 채점!
    selector = LGBMRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    selector.fit(train[features], train[target_col])
    importance = pd.Series(selector.feature_importances_, index=features).sort_values(ascending=False)
    
    # LGBM 기준 상위 10, 15, 20, 25개 묶음을 후보에 추가
    for k in [10, 15, 20, 25]:
        feature_sets[f"lgbm_importance_top_{k}"] = list(importance.head(k).index)

    deduped = {}
    seen = set()
    for name, cols in feature_sets.items():
        cols = [col for col in features if col in cols]
        signature = tuple(cols)
        if signature not in seen and len(cols) >= 3:
            deduped[name] = cols
            seen.add(signature)
    return deduped

# ---------------------------------------------------------
# [3] 메인 파이프라인
# ---------------------------------------------------------
def main() -> None:
    start_time = time.perf_counter()
    CURRENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f" [실험 2] LightGBM 맞춤형 피처 탐색 + Grid 스펙 최적화")
    print(f"Run ID: {RUN_ID}")
    print("=" * 60)

    # 데이터 준비
    train, valid, test, _ = common.prepare_life_data()
    target_col, age_col = infer_target_columns(train)
    
    y_train = train[target_col].to_numpy()
    y_valid = valid[target_col].to_numpy()
    
    train_valid = pd.concat([train, valid], ignore_index=True)
    y_train_valid = train_valid[target_col].to_numpy()
    y_test = test[target_col].to_numpy()

    # 피처셋과 스펙 후보군 로드
    feature_sets = build_feature_sets(train, target_col)
    specs = lgbm_model_specs()
    
    print(f"✔ 생성된 피처 조합 후보: {len(feature_sets)}개")
    print(f"✔ 모델 스펙 후보: {len(specs)}개")
    print(f" [Step 1] 총 {len(feature_sets) * len(specs)}번의 시뮬레이션을 통해 1위 조합 탐색 중...\n")
    
    valid_rows = []
    
    # Grid Search (피처셋 x 모델스펙) 루프
    for feature_set_name, features in feature_sets.items():
        for spec in specs:
            model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **spec["params"])
            
            t0 = time.perf_counter()
            model.fit(train[features], y_train)
            pred_valid = model.predict(valid[features])
            elapsed = time.perf_counter() - t0
            
            metrics = evaluate(y_valid, pred_valid, valid[age_col])
            
            valid_rows.append({
                "model": spec["model"],
                "variant": spec["variant"],
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "params_json": json.dumps(spec["params"]),
                "valid_rmse_months": metrics["rmse_months"],
                "valid_mae_months": metrics["mae_months"],
                "valid_r2": metrics["r2"],
                "elapsed_sec": elapsed,
                "features_list": features # 나중에 재학습을 위해 임시 보관
            })
            # 진행상황 간단 로그
            print(f" ▹ [{feature_set_name}] + [{spec['variant']}] -> Valid RMSE: {metrics['rmse_months']:.4f} 개월")

    # 결과 정렬 및 저장
    valid_df = pd.DataFrame(valid_rows).sort_values("valid_rmse_months").reset_index(drop=True)
    best_candidate = valid_df.iloc[0]
    
    # CSV 저장 시에는 리스트 형태인 features_list 컬럼은 제외하고 저장
    save_valid_df = valid_df.drop(columns=["features_list"])
    save_valid_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n [Step 1 완료] Valid 기준 압도적 1위 조합 발견!")
    print(f"  - 최고 모델 세팅: {best_candidate['variant']}")
    print(f"  - 최고 피처 세트: {best_candidate['feature_set']} ({best_candidate['feature_count']}개)")
    print(f"  - Valid RMSE: {best_candidate['valid_rmse_months']:.4f}")

    print(f"\n [Step 2] 1위 조합으로 [Train+Valid] 통합 학습 후 최종 Test 평가...")
    best_params = json.loads(best_candidate["params_json"])
    best_features = best_candidate["features_list"]
    
    final_model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **best_params)
    final_model.fit(train_valid[best_features], y_train_valid)
    pred_test = final_model.predict(test[best_features])
    
    test_metrics = evaluate(y_test, pred_test, test[age_col])
    
    # 4. 결과 출력
    print("\n" + "=" * 60)
    print(" [실험 2 최종 스코어보드] ")
    print("-" * 60)
    print(f"▶ 사용 모델    : LightGBM ({best_candidate['variant']})")
    print(f"▶ 최적 피처    : {best_candidate['feature_set']} ({best_candidate['feature_count']}개)")
    print(f"▶ Test RMSE    : {test_metrics['rmse_months']:.4f} 개월")
    print(f"▶ Test MAE     : {test_metrics['mae_months']:.4f} 개월")
    print(f"▶ Test R²      : {test_metrics['r2']:.4f}")
    print(f"▶ 임박 자산 F1 : {test_metrics['term_f1']:.4f}")
    print("=" * 60)
    print(f" 기존 1번 실험 기록(13.4582)을 넘어섰는지 확인해보세요!")

if __name__ == "__main__":
    main()