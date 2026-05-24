# 사용주기 AI 모델 실험 계획

## 1. 목표

현재 채택된 RandomForest 모델을 기준선으로 두고, 같은 데이터와 같은 산출물 계약 위에서 여러 모델을 비교한다. 최종 모델은 서버 코드 수정 없이 교체할 수 있도록 `model_interface_contract.md`의 규칙을 따른다.

핵심 목표는 단순히 RMSE가 낮은 모델을 찾는 것이 아니라, 실제 조달 화면에서 쓸 수 있는 고장 시점, 월별 수요, 권장 발주일이 안정적으로 나오는 모델을 고르는 것이다.

## 2. 현재 기준선

기준 노트북은 다음 파일이다.

```text
ai_model/notebooks/02_Modeling_RF.ipynb
```

현재 기준 모델은 `RandomForestRegressor`이다. 기존 `02_Modeling_RF.ipynb`에 저장되어 있던 출력에는 아래 성능이 남아 있었다.

- Baseline RF RMSE: 약 `3.09`개월
- Tuned RF RMSE: 약 `3.06`개월
- Tuned RF MAE: 약 `1.50`개월
- Tuned RF R2: 약 `0.92`

다만 2026-05-24 기준 로컬 `venv`와 현재 `phase4_training_data.csv`로 기존 RF 학습 조건을 재현했을 때 위 값은 재현되지 않았다. 현재 재현 가능한 RF 기준선은 아래와 같다.

- Default RF RMSE: 약 `3.72`개월
- 기존 tuned parameter RF RMSE: 약 `3.80`개월
- GridSearch 재실행 RF RMSE: 약 `3.91`개월

따라서 이후 실험의 1차 비교 기준은 새 벤치마크 노트북에서 생성하는 `ai_model/experiments/model_comparison.csv`를 기준으로 한다.

## 3. 실행 환경

VSCode에서는 프로젝트 루트의 `venv`를 사용한다.

```powershell
.\venv\Scripts\python.exe
```

노트북 커널은 `U-sto_AI (venv)`로 선택한다.

설치 완료된 실험용 패키지는 다음과 같다.

```text
ipykernel
scikit-learn
joblib
matplotlib
seaborn
scipy
statsmodels
pmdarima
xgboost
lightgbm
catboost
optuna
```

동일 환경을 다시 구성할 때는 아래 명령을 사용한다.

```powershell
.\venv\Scripts\python.exe -m pip install ipykernel scikit-learn joblib matplotlib seaborn scipy statsmodels pmdarima xgboost lightgbm catboost optuna
.\venv\Scripts\python.exe -m ipykernel install --user --name u-sto-ai --display-name "U-sto_AI (venv)"
```

## 4. 공통 데이터와 feature

모든 실험은 같은 CSV를 사용한다.

```text
dataset/create_data/data_ml/phase4_training_data.csv
```

기본 feature는 아래 목록으로 고정한다.

```python
FEATURES = [
    "내용연수",
    "취득금액",
    "부서가혹도",
    "가격민감도",
    "장비중요도",
    "리드타임등급",
    "취득월",
    "G2B목록명_Code",
    "물품분류명_Code",
    "운용부서코드_Code",
    "캠퍼스_Code",
]
```

타깃은 총 수명 개월이다.

```python
TARGET = "실제수명_개월"
```

모든 모델은 `실제수명 * 12`로 만든 총 수명 개월을 예측해야 한다. `RUL_개월`은 후처리에서 계산한다.

## 5. 실험 후보 모델

1단계에서는 빠르게 여러 모델을 같은 조건에서 돌려 기준선을 만든다.

- `DummyRegressor`: 평균 또는 중앙값 예측 기준선
- `LinearRegression`
- `Ridge`, `Lasso`, `ElasticNet`
- `KNeighborsRegressor`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`

2단계에서는 성능이 기대되는 부스팅 모델을 튜닝한다.

- `XGBRegressor`
- `LGBMRegressor`
- `CatBoostRegressor`

3단계에서는 시계열 관점의 보조 실험을 분리해서 진행한다.

- `ARIMA` 또는 `auto_arima`
- `ARIMA + XGBoost` 잔차 보정
- 월별 총 고장 수요 예측 모델

시계열 모델은 자산별 총 수명 예측 모델과 목적이 다르므로, 최종 서버 교체 후보로 바로 비교하기보다 월별 수요 그래프 보정용 보조 모델로 평가한다.

4단계에서는 상위 모델을 조합한다.

- `VotingRegressor`
- `StackingRegressor`
- 상위 2개 모델 평균 앙상블

앙상블은 RMSE가 낮아도 추론 시간이 길거나 설명력이 떨어지면 최종 배포 후보에서 제외할 수 있다.

## 6. 평가 방식

기본 평가는 기존 분할을 그대로 따른다.

- Train: `데이터세트구분 in ["Train", "Valid"]`
- Test: `데이터세트구분 == "Test"`
- Prediction: `데이터세트구분 == "Prediction"`

추가 검증으로 시간 기반 holdout을 검토한다. 예를 들어 취득일자 또는 불용일자 기준으로 최근 기간을 검증 구간으로 두면 실제 운영 환경에 가까운 성능을 볼 수 있다.

## 7. 평가 지표

모델 자체 성능은 아래 지표로 비교한다.

- `RMSE_months`: 1차 선택 기준
- `MAE_months`: 평균 오차 체감 기준
- `R2`: 설명력 확인
- `MAPE`: 타깃이 0에 가까운 경우 왜곡될 수 있으므로 보조 지표로만 사용

서비스 품질 관점에서는 아래 지표를 추가한다.

- 학기 내 고장예상 여부의 precision, recall, F1
- 월별 예상고장수량 MAE
- 월별 필요수량 MAE
- 권장발주기한이 과도하게 늦거나 빠른 케이스 수
- 품목별 상위 위험 장비 ranking 안정성

최종 선택은 `RMSE_months`가 가장 낮은 모델을 우선하되, 조달 권고안이 불안정하거나 특정 품목에 과하게 치우치는 모델은 제외한다.

## 8. 튜닝 전략

빠른 1차 실험에서는 기본 파라미터 또는 작은 grid를 사용한다.

2차 후보 모델은 `RandomizedSearchCV` 또는 `Optuna`로 튜닝한다.

권장 튜닝 순서는 다음과 같다.

1. RandomForest와 ExtraTrees의 tree 수, depth, leaf 조건 조정
2. XGBoost의 `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree` 조정
3. LightGBM의 `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples` 조정
4. CatBoost의 `depth`, `learning_rate`, `iterations`, `l2_leaf_reg` 조정
5. 상위 2개 모델로 stacking 또는 simple average 검토

## 9. 산출물 규칙

각 실험은 아래 구조로 저장한다.

```text
ai_model/experiments/runs/{run_id}_{model_name}/
  model.pkl
  model_meta.json
  metrics.json
  test_predictions.csv
  feature_importance.png
```

`metrics.json` 예시는 다음과 같다.

```json
{
  "run_id": "20260523_235900_lgbm",
  "model_name": "LGBMRegressor",
  "rmse_months": 0.0,
  "mae_months": 0.0,
  "r2": 0.0,
  "features": [],
  "target": "실제수명_개월"
}
```

최종 채택 모델은 서버 호환을 위해 아래 경로에도 복사한다.

```text
ai_model/saved_models/random_forest/rf_final_model.pkl
ai_model/saved_models/random_forest/model_meta.json
```

폴더명이 `random_forest`여도 당장은 호환성을 위해 유지한다. 추후 서버 코드에서 active model 경로를 설정값으로 분리하면 `ai_model/saved_models/active_model/` 같은 이름으로 바꾸는 것이 더 좋다.

## 10. 개발 단계

### Step 1. 노트북 정리

`02_Modeling_RF.ipynb`의 학습 부분을 함수화한다.

- `load_training_data()`
- `build_train_test_split()`
- `train_model()`
- `evaluate_model()`
- `save_artifacts()`

### Step 2. 공통 벤치마크 스크립트 작성

반복 실험을 위해 아래 스크립트를 만든다.

```text
ai_model/experiments/run_benchmark.py
```

역할은 모델 후보를 순회 학습하고, 동일한 test set으로 metrics와 산출물을 저장하는 것이다.

### Step 3. 1차 모델 비교

작은 파라미터로 전체 후보를 돌려 성능표를 만든다.

```text
ai_model/experiments/model_comparison.csv
```

이 단계의 목표는 최종 모델 결정이 아니라 튜닝할 가치가 있는 상위 3개 모델을 고르는 것이다.

### Step 4. 상위 모델 튜닝

상위 3개 모델에 대해 Optuna 또는 RandomizedSearchCV를 적용한다. 튜닝 시간은 모델별로 제한하고, 같은 random seed를 사용한다.

### Step 5. 후처리 파이프라인 검증

후보 모델별로 `run_procurement_pipeline()`를 실행해 아래 결과가 정상 생성되는지 확인한다.

- `df_matrix_asset`
- `df_ts_demand`
- `df_plan`

특히 `AI예측고장일`, `권장발주기한`, `추정예산`의 값 범위가 실무적으로 이상하지 않은지 확인한다.

### Step 6. 최종 모델 채택

최종 모델은 아래 조건을 모두 만족해야 한다.

- Test RMSE가 RF 기준선보다 낮다.
- MAE와 R2가 같이 악화되지 않는다.
- 학기 내 고장예상 분류가 과하게 보수적이거나 공격적이지 않다.
- `model_meta.json`과 `rf_final_model.pkl`만 교체해도 후처리 결과가 생성된다.
- VSCode 로컬 환경과 서버 실행 환경에서 동일하게 로드된다.

## 11. 우선순위

가장 먼저 할 일은 공통 벤치마크 스크립트 작성이다. 지금처럼 노트북마다 모델 코드가 분리되어 있으면 성능 비교가 흔들리기 쉽다.

추천 진행 순서는 다음과 같다.

1. `model_interface_contract.md` 기준으로 RF 노트북의 입출력 계약을 고정한다.
2. 공통 벤치마크 스크립트를 만든다.
3. RF, ExtraTrees, XGBoost, LightGBM, CatBoost를 같은 split으로 비교한다.
4. 상위 2개 모델만 튜닝한다.
5. 최종 모델을 기존 배포 파일명으로 저장하고 서버/UI에서 한 번 검증한다.

## 12. 진행 기록

### 1차 벤치마크

2026-05-24 기준 `03_Modeling_Experiment_Benchmark.ipynb`에서 1차 모델 비교를 실행했다. 결과 파일은 `ai_model/experiments/model_comparison.csv`이다.

상위 결과는 다음과 같다.

- ExtraTrees: Test RMSE `3.5318`개월, MAE `1.5726`개월, R2 `0.8935`
- XGBoost: Test RMSE `3.7151`개월, MAE `1.4461`개월, R2 `0.8822`
- CatBoost: Test RMSE `3.7371`개월, MAE `1.4677`개월, R2 `0.8808`

### 2단계 튜닝

2026-05-24 기준 `04_Modeling_Stage2_Tuning.ipynb` 구조로 2단계 튜닝을 진행했다. 3-fold CV 기반 튜닝은 Test 성능이 1차 ExtraTrees보다 좋아지지 않았기 때문에, 기존 데이터의 `Train`, `Valid`, `Test` 분할 의미를 살려 Valid 기반 선택 절차를 추가했다.

최종 절차는 `Train`으로 파라미터 후보를 비교하고, `Valid` RMSE로 파라미터를 선택한 뒤, `Train+Valid` 전체로 재학습하고 `Test`를 최종 평가하는 방식이다.

2단계 결과 파일은 `ai_model/experiments/stage2_valid_tuning_results.csv`이며, 최상위 모델은 CatBoost이다.

- CatBoost: Test RMSE `3.4684`개월, MAE `1.4099`개월, R2 `0.8973`
- XGBoost: Test RMSE `3.8071`개월, MAE `1.5028`개월, R2 `0.8762`
- RandomForest: Test RMSE `3.9139`개월, MAE `1.5801`개월, R2 `0.8692`
- ExtraTrees: Test RMSE `3.9447`개월, MAE `1.6052`개월, R2 `0.8671`

현재 기준으로는 1차 ExtraTrees보다 2단계 CatBoost가 더 좋은 최종 후보이다. 다만 서버 배포 경로는 아직 덮어쓰지 않았다.

### 3단계 월별 수요 예측 보조 실험

2026-05-24 기준 `05_Modeling_Stage3_TimeSeries_Demand.ipynb`와 `run_stage3_timeseries_demand.py`를 추가했다. 3단계는 자산별 총수명 예측 모델을 직접 교체하는 단계가 아니라, 월별 고장/처분 수요 그래프와 조달 수량 산정의 안정성을 확인하는 보조 실험이다.

최신 결과 파일은 `ai_model/experiments/stage3_timeseries_demand_results.csv`이며, 실행 산출물은 `ai_model/experiments/runs/20260524_120335_stage3_timeseries_demand`에 저장했다.

- MovingAverage6: 월별 수요 RMSE `10.2347`건, MAE `8.1667`건
- SARIMAX: 월별 수요 RMSE `10.8389`건, MAE `9.0640`건
- XGBoostLag: 월별 수요 RMSE `12.6105`건, MAE `10.3315`건
- SARIMAX + XGBoost Residual: 월별 수요 RMSE `13.1349`건, MAE `10.6182`건
- SeasonalNaive: 월별 수요 RMSE `14.8015`건, MAE `12.2500`건
- Stage2AssetModelMonthly: 월별 수요 RMSE `15.4765`건, MAE `12.0000`건

현재 합성 데이터에서는 복잡한 시계열 모델보다 최근 6개월 이동평균이 월별 총량 예측에는 가장 안정적이었다. 따라서 3단계 결과는 최종 `rf_final_model.pkl` 교체 후보라기보다, UI의 월별 수요 그래프를 보정하거나 조달 계획의 월별 총량 기준선을 제시하는 보조 로직으로 해석한다.

### 4~6단계 최종 검증 및 취득일자 holdout

2026-05-24 기준 `06_Modeling_Stage4_6_Final_Validation.ipynb`와 `run_stage4_6_final_validation.py`를 추가했다. 4단계에서는 2단계 산출물과 단순 앙상블을 비교했고, 추가 검증으로 `취득일자` 기준 최근 20% 구간을 holdout으로 분리해 재학습/검증했다. 5단계에서는 Prediction 대상에 대해 RUL, 예측 고장월 smoke test를 수행했고, 6단계에서는 최종 후보와 서버 호환성을 점검했다.

최신 실행 산출물은 `ai_model/experiments/runs/20260524_121229_stage4_6_final_validation`에 저장했다. 요약 결과는 `ai_model/experiments/stage4_6_final_validation_results.csv`, 최종 리포트는 `ai_model/experiments/final_model_selection_report.md`이다.

기존 Test split 기준 상위 결과는 다음과 같다.

- ExtraTrees: RMSE `17.7531`개월, MAE `13.5592`개월, R2 `0.8395`, 학기 내 고장 F1 `0.8738`
- AverageTop2: RMSE `17.8912`개월, MAE `13.7364`개월, R2 `0.8370`, 학기 내 고장 F1 `0.8798`
- AverageTop3: RMSE `17.9909`개월, MAE `13.8287`개월, R2 `0.8352`, 학기 내 고장 F1 `0.8832`

취득일자 기준 최근 holdout 검증의 cutoff는 `2016-11-13`이며, 오래된 취득건 5,738건으로 학습하고 최근 취득건 1,436건으로 검증했다. 상위 결과는 다음과 같다.

- ExtraTrees: RMSE `15.9289`개월, MAE `12.5267`개월, R2 `0.5042`, 학기 내 고장 F1 `0.6838`
- AverageTop2: RMSE `15.9344`개월, MAE `12.5529`개월, R2 `0.5039`, 학기 내 고장 F1 `0.6844`
- AverageAll4: RMSE `15.9522`개월, MAE `12.5696`개월, R2 `0.5027`, 학기 내 고장 F1 `0.6838`

결론적으로 현재 데이터에서는 앙상블이 단일 ExtraTrees보다 RMSE를 낮추지 못했다. 최종 후보는 ExtraTrees로 유지하되, 실제 서버 배포 전에는 서버 입력 feature와 후처리 로직을 맞춰야 한다. 현재 서버는 feature 9개를 하드코딩하고 있으나 최종 모델은 11개 feature를 사용하며, 서버는 총수명 예측값에서 `운용연차 * 12`를 뺀 RUL 계산 없이 현재일에 바로 더하고 있다.
