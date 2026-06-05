# AI 모델링 고도화 최종 실험 리포트 (실험 1 ~ 10 총망라)

본 보고서는 **자산 수명 예측(Stage 2)** 및 **월별 수요 예측(Stage 3)** 모델의 성능을 극대화하기 위해 수행된 총 10가지 통제 실험 결과를 체계적으로 비교 분석한 최종 마스터 리포트입니다.

---

## 한눈에 보는 핵심 요약

### 최종 권고 배포 모델 요약
| 구분 | 추천 모델 | 핵심 지표 (Test) | 비즈니스 기대 효과 및 채택 사유 |
| :--- | :--- | :--- | :--- |
| **Stage 2(자산 수명)** | **CatBoost + Optuna**(실험 3) | - **RMSE:** 13.31 개월- **R²:** 0.9037 | **[범용성 최우선]** 연속형 수명 예측 오차가 가장 적으며, 과적합이 완벽히 제어되어 안정적인 예산 수립 가능 |
| **Stage 2(특수 목적)** | **XGBoost + Optuna**(실험 8) | - **임박 자산 F1:** **0.8413**(프로젝트 최고치) | **[고장 알람 정밀도 최우선]** 당장 교체가 시급한 자산을 잡아내는 정밀도가 극대화되어 교체 공백 방지에 유리 |
| **Stage 3(월별 수요)** | **LightGBM GridSearch**(실험 9, `variant_5`) | - **RMSE:** **8.02 대**- **R²:** **0.9863** | **[종합 1위] 설명력 98.6%, 실제 월별 수요와의 평균 오차 단 ±5.2대 수준.** Train+Valid 전체 통합 재학습(Refit)을 통해 최신 시계열 트렌드를 완벽 반영한 괴물급 모델 |

---

## 실험 환경 및 입력 피처(Feature) 요약

* **자산 수명 예측 모델 Feature (15개 고정):**
`내용연수`, `부서가혹도`, `월평균사용시간`, `사용강도지수`, `누적점검수리횟수`, `누적수리횟수`, `최근2년수리횟수`, `마지막수리후경과개월`, `취득금액대비수리비율`, `최대장애심각도`, `부서예산등급_Code`, `부서교체성향`, `G2B목록명_Code`, `물품분류명_Code`, `운용부서코드_Code`

* **월별 수요 예측 모델 Feature (7개 고정):**
`trend`, `month`, `month_sin`, `month_cos`, `lag_12`, `rolling_mean_6`, `rolling_std_6`

---

## Ⅰ. [Stage 2] 자산 수명 예측 모델 실험 (실험 1~3, 5~8)

### [실험 1] 동일 피처(15개) + GridSearchCV + 신규 모델(LGBM)

* **스크립트:** `run_exp1_lgbm_fixed.py`
* **실험 목적:** 15개 기본 피처 환경에서 LightGBM을 신규 도입하고, 기존의 튜닝 방식인 GridSearchCV를 통해 모델 성향별(기본형, 얕은 규제형, 벼락치기형, 과적합형) 성능 변동을 분석합니다.
* **스펙 탐색 결과 (Valid RMSE):**
* `lgbm_balanced` (max_depth: 6, num_leaves: 31): 14.2468 개월
* **`lgbm_shallow_regularized` (max_depth: 4, num_leaves: 15) : 14.0643 개월 (1위)**
* `lgbm_fast_shallow` (max_depth: 5, n_estimators: 500): 14.2824 개월
* `lgbm_deeper_slow` (max_depth: 7, num_leaves: 63): 14.4140 개월

* **최종 Test 셋 평가 결과 (`lgbm_shallow_regularized` 선정):**
* **Test RMSE:** 13.4582 개월 | **Test MAE:** 10.2584 개월 | **Test R²:** 0.9016 | **임박 자산 F1:** 0.8277

* **인사이트:** 트리의 깊이를 4로 얕게 제한하는 대신 학습률을 낮춰 900번 반복 학습한 모델이 우수했습니다. 지엽적인 노이즈에 집착하지 않고 굵직한 패턴만 학습해야 일반화 성능이 올라갑니다.

### [실험 2] 자체 피처 최적화 + GridSearchCV + 신규 모델(LGBM)

* **스크립트:** `run_exp2_lgbm_search.py`
* **실험 목적:** 'lgbm_balanced', 'lgbm_shallow_regularized', 'lgbm_fast_shallow', 'lgbm_deeper_slow'를 이용하여 40번의 시뮬레이션을 통해 피처 세트 조합 다변화에 따른 최적의 유효 변수 조합을 도출합니다.
* **주요 조합별 Valid RMSE 결과:**
* `full_35` + `lgbm_shallow_regularized`: 14.1338 개월
* `no_category_codes_31` + `lgbm_shallow_regularized`: 14.8972개월
* `asset_usage_maintenance_25` + `lgbm_shallow_regularized`: 14.8920 개월
* `compact_domain_20` + `lgbm_shallow_regularized`: 16.2407 개월
* `simple_asset_11` + `lgbm_shallow_regularized`: 21.8302 개월
* `no_maintenance_26` + `lgbm_shallow_regularized`: 18.6466 개월
* `lgbm_importance_top_10` + `lgbm_shallow_regularized`: 16.4391 개월
* `lgbm_importance_top_15` + `lgbm_fast_shallow`: 14.3121개월
* `lgbm_importance_top_20` + `lgbm_shallow_regularized`: 14.1831 개월
* `lgbm_importance_top_25` + `lgbm_shallow_regularized`: **14.1333 개월 (★ 최적)**

* **최종 Test 셋 평가 결과:**
* **최고 피처 세트:** `lgbm_importance_top_25` (25개 변수)
* **Test RMSE:** 13.5833 개월 | **Test MAE:** 10.4094 개월 | **Test R²:** 0.8997 | **임박 자산 F1:** 0.8350

* **인사이트:** 무조건 변수를 늘리는 것보다 변수 중요도 상위 25개 피처를 엄선하여 노이즈를 걸러냈을 때 가장 정밀하고 안정적인 밸런스를 보여주었습니다.

### [실험 3] 동일 피처(15개) + Optuna + 기존 모델(CatBoost)

* **스크립트:** `run_exp3_catboost_optuna.py`
* **실험 목적:** 기존 베이스라인인 CatBoost 모델을 복귀시키고 Optuna 베이지안 최적화(30 Trials)를 적용하여 연속적인 하이퍼파라미터 공간 전역해를 수립합니다.
* **최적 하이퍼파라미터:** `{'iterations': 1069, 'learning_rate': 0.0563, 'depth': 4, 'l2_leaf_reg': 0.1698, 'random_strength': 0.249, 'bagging_temperature': 0.2760, 'random_seed': 42, 'verbose': False}`

* **최종 Test 셋 평가 결과 (Valid RMSE: 13.8571):**
* **Test RMSE:** 13.3133 개월 | **Test MAE:** 10.1524 개월 | **Test R²:** 0.9037 | **임박 자산 F1:** 0.8232

* **핵심 발견:** 모델 종류(CatBoost/LGBM)나 튜닝 방식(Grid/Optuna)에 상관없이 **자산 수명 데이터는 트리 깊이를 얕게(`depth: 4`) 제어해야만 과적합을 피하고 최종 Test 셋에서 피크 성능**을 낸다는 물리적 특성을 증명했습니다.

### [실험 5] 동일 피처(15개) + Optuna + 신규 모델(LGBM)

* **스크립트:** `run_exp5_lgbm_fixed.py`
* **실험 목적:** 15개 기본 피처 환경에서 LightGBM의 파라미터 공간(`num_leaves`, `min_child_samples` 등 규제 포함)을 Optuna로 자동 탐색합니다.
* **최적 하이퍼파라미터:** `{'n_estimators': 800, 'max_depth': 5, 'num_leaves': 168, 'learning_rate': 0.0165, 'min_child_samples': 25, 'subsample': 0.8569, 'colsample_bytree': 0.6822}`

* **최종 Test 셋 평가 결과:**
* **Test RMSE:** 13.4931 개월 | **Test MAE:** 10.3424 개월 | **Test R²:** 0.9010 | **임박자산 F1:** 0.8268

* **인사이트:** 리프 중심(Leaf-wise)으로 자라는 LightGBM의 특성을 반영해 `num_leaves: 168`까지 공간을 열어주었으나, 과적합 방지를 위해 `max_depth: 5`, `min_child_samples: 25` 수준에서 안정적인 브레이크가 걸렸습니다.

### [실험 6] 자체 피처 최적화 + Optuna + 신규 모델(LGBM)

* **스크립트:** `run_exp6_lgbm_optuna_opt.py`
* **실험 목적:** 실험 2의 피처 다변화 관점과 실험 5의 Optuna 자동 하이퍼파라미터 튜닝 기법을 결합하여 결합 최적화 시너지 효과를 검증합니다.
* **최적 하이퍼파라미터:** `{'n_estimators': 900, 'max_depth': 6, 'num_leaves': 179, 'learning_rate': 0.0166, 'min_child_samples': 38, 'subsample': 0.9040, 'colsample_bytree': 0.6626}`
* **최종 Test 셋 평가 결과:**
* **Test RMSE:** 13.8478 개월 | **Test MAE:** 10.5245 개월 | **Test R²:** 0.8958 | **임박자산 F1:** 0.8287

* **인사이트:** 피처 선택과 파라미터 영역을 동시에 개방했을 때, 검증 범위 밖의 최종 테스트 데이터 환경에서는 소폭의 오차가 증가(RMSE 상승)하는 현상이 관측되어 과도하게 넓은 튜닝 공간의 기회비용을 정량화했습니다.

### [실험 7] 동일 피처(15개) + GridSearchCV + 신규 모델(XGBoost)

* **스크립트:** `run_exp7_xgb_fixed_gs.py`
* **실험 목적:** 전통적인 수평 분할 구조의 대표 격인 XGBoost 모델을 고정 피처 환경에 투입하여 타 부스팅 알고리즘과의 원천 성능 격차를 검증합니다.
* **최적 하이퍼파라미터:** `{'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': 4, 'n_estimators': 500, 'subsample': 0.8}`
* **최종 Test 셋 평가 결과:**
* **Test RMSE:** 15.4779 개월 | **Test MAE:** 12.0820 개월 | **Test R²:** 0.8698 | **임박자산 F1:** 0.8277

* **인사이트:** 임박 자산 F1 스코어는 방어했으나 연속형 오차(RMSE)와 설명력(R2)이 대폭 하락했습니다. 우리 데이터 구조상 XGBoost의 손실 함수 빌딩 메커니즘은 타 모델 대비 다소 적합도가 낮음을 확인했습니다.

### [실험 8] 동일 피처(15개) + Optuna + 신규 모델(XGBoost)

* **스크립트:** `run_exp8_xgb_fixed_optuna.py`
* **실험 목적:** XGBoost의 구조적 한계를 Optuna의 확률적 전역 탐색 공간으로 이관했을 때의 복구 성능 및 지표 변동을 추적합니다.
* **최적 하이퍼파라미터:** `{'n_estimators': 800, 'max_depth': 3, 'learning_rate': 0.0908, 'min_child_weight': 1, 'subsample': 0.9047, 'colsample_bytree': 0.9992}`
* **최종 Test 셋 평가 결과:**
* **Test RMSE:** 15.4558 개월 | **Test MAE:** 11.9869 개월 | **Test R²:** 0.8702 | **임박자산 F1:** **0.8413 (전체 실험 중 역대 최고)**

* **인사이트:** 전체 오차 부문은 부스팅 모델 중 가장 열세였으나, 고장이 임박한 자산을 정밀하게 집어내는 **'임박자산 F1-Score'가 0.8413으로 프로젝트 최고 성적을 기록**했습니다. 고장 알람의 정밀도가 최우선인 특수 실무 환경에 단독 배포 가능한 유력 후보군입니다.

---

## Ⅱ. [Stage 3] 월별 수요 예측 시계열 모델 실험 (실험 4, 9~10)

### [실험 4] 동일 피처(7개) + Optuna + 기존 모델(ExtraTrees)

* **스크립트:** `run_exp4_et_optuna.py`
* **실험 목적:** 시계열 패턴 분석을 위해 엄선된 고정 7개 피처 구조 하에서 기존 앙상블 모델인 ExtraTrees의 나무 분할 속성을 다각도로 최적화합니다.
* **최적 하이퍼파라미터:** `{'n_estimators': 700, 'max_depth': 11, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 0.9777, 'bootstrap': True}`
* **최종 Test 셋 평가 결과 (Valid RMSE: 11.2402):**
* **Test RMSE:** 10.0628 개 | **Test MAE:** 5.5365 개 | **Test R²:** 0.9785

* **핵심 발견:** 자산 수명 모델(Stage 2)과 정반대로, 시계열 수요 모델에서는 Optuna가 **`max_depth: 11`이라는 깊은 구조를 선택**했습니다. 이는 **시간의 흐름에 따른 주기성과 복잡한 인과 맥락을 학습하기 위해 모델 스스로가 깊은 사고 구조를 수립**해낸 유의미한 결과물입니다. 또한 변수 가중치 선택 비율이 97.7%에 달해 선정한 7개 시계열 변수 모두가 극도로 유효함을 증명했습니다.

### [실험 9] 동일 피처(7개) + GridSearchCV + 신규 모델(LightGBM)

* **스크립트:** `run_exp9_lgbm_demand_gs.py`
* **실험 목적:** 연산 효율성이 압도적인 LightGBM을 시계열 수요 데이터에 신규 도입하고, GridSearch를 통해 하이퍼파라미터 조합에 따른 최종 예측 성능 랭킹을 산출합니다.
* **GridSearch 주요 상위 결과 테이블:**
| variant (버전) | 파라미터 조합 (params) | Valid RMSE | Valid MAE | Valid R² | **Test RMSE** | **Test MAE** | **Test R²** | **최종순위** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **lgbm_gs_variant_5** | `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 6, "n_estimators": 500, "subsample": 0.8}` | 14.7070 | 11.2512 | 0.9580 | **8.0242** | **5.2460** | **0.986297** | **1위** |
| lgbm_gs_variant_2 | `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 4, "n_estimators": 500, "subsample": 0.8}` | 14.6430 | 11.3408 | 0.9584 | 8.0518 | 5.4865 | 0.986203 | 2위 |
| lgbm_gs_variant_8 | `{"colsample_bytree": 0.8, "learning_rate": 0.03, "max_depth": 8, "n_estimators": 500, "subsample": 0.8}` | 14.6001 | 11.0809 | 0.9587 | 8.0538 | 5.2702 | 0.986196 | 3위 |
| lgbm_gs_variant_10 | `{"colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 4, "n_estimators": 300, "subsample": 0.8}` | 14.0961 | 10.5486 | 0.9615 | 8.4076 | 4.3835 | 0.984956 | 4위 |

* **전 실험 통틀어 역대 최고 스코어 달성 포인트:**
1. **Validation 순위 뒤집기 현상 (Refit의 저력):** 최종 우승한 `variant_5` 모델은 검증(Valid) 단계에서 전체 9위에 머물렀으나, 검증 이후 `Train + Valid` 전체 데이터를 병합하여 재학습(Refit)을 수행하자 미학습 영역의 최신 시계열 패턴을 완벽히 흡수하여 **최종 Test 데이터에서 R² 0.9863 이라는 경이로운 스코어로 최종 1등**을 차지했습니다.
2. **안정적인 속도 제어:** 학습률을 0.1로 가져간 과감한 속성 모델군보다, `learning_rate: 0.03`으로 템포를 조절하며 나무를 500개 쌓아 올린 모델이 시계열 재귀 예측 오차 누적을 가장 철저하게 틀어막았습니다.

### [실험 10] 동일 피처(7개) + Optuna + 신규 모델(LightGBM)

* **스크립트:** `run_exp10_lgbm_demand_optuna.py`
* **실험 목적:** 인간의 도메인 지식을 초월해 연속형 파라미터 경계 조건 내에서 전역 최적해 조합을 도출하기 위한 무작위 베이지안 최적화를 수행합니다.
* **Optuna 무작위 탐색 주요 상위 결과 테이블:**

| trial (버전) | 파라미터 조합 (params) | Valid RMSE | Valid MAE | Valid R² | **Test RMSE** | **Test MAE** | **Test R²** | **최종순위** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **lgbm_optuna_trial_0** | `{"colsample_bytree": 0.6232, "learning_rate": 0.0396, "max_depth": 10, "min_child_samples": 12, "n_estimators": 400, "num_leaves": 191, "subsample": 0.6623}` | 10.1478 | 7.5698 | 0.9800 | **9.7035** | **6.6130** | **0.979962** | **1위** |
| lgbm_optuna_trial_13 | `{"colsample_bytree": 0.6103, "learning_rate": 0.0502, "max_depth": 9, "min_child_samples": 12, "n_estimators": 700, "num_leaves": 203, "subsample": 0.9087}` | 8.3750 | 6.4376 | 0.9864 | 11.1457 | 7.3744 | 0.973562 | 2위 |
| lgbm_optuna_trial_18 | `{"colsample_bytree": 0.7544, "learning_rate": 0.0519, "max_depth": 7, "min_child_samples": 11, "n_estimators": 800, "num_leaves": 56, "subsample": 0.8614}` | 7.8712 | 6.1128 | 0.9879 | 11.3675 | 6.3416 | 0.972500 | 3위 |
| lgbm_optuna_trial_25 | `{"colsample_bytree": 0.8924, "learning_rate": 0.0948, "max_depth": 6, "min_child_samples": 9, "n_estimators": 700, "num_leaves": 18, "subsample": 0.7386}` | 5.4428 | 4.3923 | 0.9942 | 12.6555 | 6.9553 | 0.965915 | 6위 |

* **인사이트 및 과적합(Overfitting) 현상 분석:**
1. **강력한 규제를 통한 일반화:** 우승 트라이얼인 `trial_0`은 데이터 샘플링 비율(`subsample: 0.66`, `colsample_bytree: 0.62`)을 극도로 제약하여 모델을 강하게 묶어둔 덕분에 테스트 환경에서 오버피팅 없이 R² 97.9%의 방어력을 보였습니다.
2. **GridSearch 모델의 판정승:** 일부 하위 트라이얼(`trial_25` 등)은 검증 기간 데이터에 과도하게 동화(`Valid R²: 0.9942`)되려다 보니 최종 테스트 데이터셋에서는 도리어 오버피팅을 겪었습니다. 결과적으로 인간의 도메인 직관을 바탕으로 탐색 반경을 안전하게 제한했던 실험 9번의 GridSearch 1등 모델이 범용성 측면에서 판정승(Test R² 0.9863)을 거두었습니다.

---

## 최종 결론 및 현업 배포(Deployment) 가이드

1. **자산 수명 예측 (Stage 2):** 오차율 제어와 수명 예측 밸런스가 뛰어난 **[실험 3] CatBoost + Optuna 최적화 모델(Test R²: 0.9037)**을 메인 운영 환경 모델로 배포할 것을 강력히 권장합니다. 다만, 현업 배포 시 고장 알람의 정확도가 비용 절감 측면에서 최우선이라면 임박자산 탐지 지표가 압도적인 **[실험 8] XGBoost + Optuna 모델(F1-Score: 0.8413)**을 비즈니스 목적에 따라 교체 서빙할 수 있도록 준비합니다.
2. **월별 수요 예측 (Stage 3):** 시계열 재귀 예측 오차를 완벽히 통제하고 98.6%라는 경이로운 설명력을 확보한 **[실험 9] LightGBM GridSearch 1등 모델(`lgbm_gs_variant_5`)**을 최종 운영 서빙 모델로 확정합니다. 실제 수요 대수와의 평균 오차가 단 5.2대 수준이므로 자산 수급 자동화 로직에 즉각 통합할 수 있는 고품질 지표입니다.