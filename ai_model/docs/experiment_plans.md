# 사용주기 AI 모델 실험 계획

작성일: 2026-05-25

## 1. 현재 목표

사용주기 AI는 하나의 모델로 모든 값을 직접 계산하지 않고, 목적이 다른 두 모델과 규칙 기반 조달 계산을 결합한다.

| 구분 | 목적 | 현재 선택 |
| --- | --- | --- |
| 자산 수명 모델 | 개별 자산의 총수명/잔여수명 예측 | CatBoost |
| 월별 수요 모델 | 월별 고장/처분 예상 수량 예측 | ExtraTrees |
| 조달 계산 로직 | 안전재고, 필요수량, ROP, 권장발주기한 산출 | 서버 규칙 로직 |

이 구조를 선택한 이유는 개별 자산 수명 예측과 월별 발생량 예측의 목적이 다르기 때문이다. 자산 수명 모델은 “이 물품이 언제쯤 수명을 다하는가”에 강하고, 월별 수요 모델은 “특정 월에 얼마나 몰리는가”를 설명하는 데 더 적합하다.

## 2. 데이터 기준

현재 실험 데이터는 `dataset/create_data/phase1_acquisition.py`부터 `phase4_ml_prep.py`까지 재생성한 합성 데이터다.

데이터 생성에는 다음 패턴을 반영했다.

- 품목별 내용연수와 수명 변동성
- 부서별 가혹도, 교체 성향, 관리 성향
- 구매 배치와 대량 구매 효과
- 사용량/가동률, 공용 장비, 수업 사용 여부
- 점검/수리 이력과 수리 비용 누적
- 학사 일정, 방학, 신학기 준비, 예산 집행월
- Train/Valid/Test/Prediction 분리

모델 입력에는 결과를 직접 알려주는 컬럼을 넣지 않는다. 예를 들어 `불용일자`, `실제수명`, `실제잔여수명`, `AI예측고장일`, `권장발주일`, 월별 필요수량 컬럼은 feature에서 제외한다.

## 3. 활성 실험 스크립트

현재 활성 스크립트는 아래 4개다.

| 파일 | 역할 |
| --- | --- |
| `ai_model/experiments/scripts/modeling_common.py` | 공통 경로, feature list, 데이터 전처리, 월별 시계열 생성 |
| `ai_model/experiments/scripts/run_stage2_life_model_search.py` | 자산 수명 모델 탐색 및 배포 모델 갱신 |
| `ai_model/experiments/scripts/run_stage3_monthly_model_search.py` | 월별 수요 모델 탐색 및 월별 모델 artifact 생성 |
| `ai_model/experiments/scripts/make_presentation_plots_current.py` | 발표용 그래프 4종 및 요약 지표 생성 |

이전 벤치마크 노트북과 구형 실험 스크립트는 `ai_model/archive/obsolete/20260525_modeling_cleanup`로 옮겼다.

## 4. Stage 2: 자산 수명 모델 탐색

### 4.1. 예측 대상

자산 수명 모델은 개별 자산의 `실제수명_개월`, 즉 총수명을 예측한다.

서버에서는 예측 총수명에서 현재 운용연차를 빼서 잔여수명(RUL)을 계산하고, 이를 바탕으로 고장예측일을 산출한다.

### 4.2. 검증 방식

- `Train`으로 학습
- `Valid`로 모델/피처셋 후보 선택
- `Train + Valid`로 재학습
- `Test`로 최종 성능 확인

성능 지표는 다음을 사용한다.

- RMSE(months): 총수명 오차의 크기
- MAE(months): 평균 절대 오차
- R2: 설명력
- term precision/recall/F1: 잔여수명 6개월 이하 임박 자산 탐지 품질

### 4.3. 비교한 모델

- CatBoost
- XGBoost
- ExtraTrees
- RandomForest
- GradientBoosting

각 모델은 여러 하이퍼파라미터 조합으로 비교했다.

### 4.4. 비교한 피처셋

- `full_35`: 전체 35개 feature
- `no_category_codes_31`: 카테고리 코드 제외
- `asset_usage_maintenance_25`: 자산/사용/수리 중심
- `compact_domain_20`: 도메인 핵심 feature 중심
- `simple_asset_11`: 단순 자산 기준 feature
- `no_maintenance_26`: 점검/수리 feature 제외
- `importance_top_10/15/20/25`: ExtraTrees 중요도 기반 상위 feature

### 4.5. 현재 최종 결과

| 항목 | 값 |
| --- | --- |
| 최종 모델 | CatBoost |
| variant | `cb_fast_shallow` |
| feature set | `importance_top_15` |
| feature 수 | 15개 |
| Test RMSE | 13.2765개월 |
| Test MAE | 10.1507개월 |
| Test R2 | 0.9042 |
| 임박 자산 F1 | 0.8314 |

최종 모델은 `ai_model/saved_models/current/model.pkl`에 배포되어 있으며, 서버는 `model_meta.json`에 저장된 feature list를 읽어 입력 순서를 맞춘다.

## 5. Stage 3: 월별 수요 모델 탐색

### 5.1. 예측 대상

월별 수요 모델은 월별 고장/처분 발생량을 예측한다. UI의 `고장 예상 수량` 그래프는 이 모델의 예측 흐름을 기준으로 사용한다.

### 5.2. 검증 방식

월별 수요는 시간 순서가 중요하므로, 테스트 구간의 실제 미래 lag 값을 직접 쓰지 않는다. 이전 달 예측값을 다음 달 lag로 넣는 recursive backtest 방식으로 검증한다.

### 5.3. 비교한 모델

- XGBoost
- RandomForest
- ExtraTrees
- GradientBoosting
- Ridge
- Seasonal Naive, Moving Average 계열 기준선

### 5.4. 비교한 피처셋

- `all_12`: trend, month, sin/cos, lag, rolling 전체
- `lag_short_8`: 단기 lag 중심
- `seasonal_7`: 계절성 + 전년 lag + 6개월 rolling 중심
- `lag_only_8`: lag/rolling만 사용
- `compact_6`: 최소 계절/lag feature
- `no_trend_11`: trend 제외

### 5.5. 현재 최종 결과

| 항목 | 값 |
| --- | --- |
| 최종 모델 | ExtraTrees |
| variant | `et_regularized` |
| feature set | `seasonal_7` |
| feature 수 | 7개 |
| Test RMSE | 9.7765건 |
| Test MAE | 5.3774건 |
| Test R2 | 0.9797 |

월별 모델 artifact는 최신 run 폴더의 `monthly_demand_model.pkl`에 저장된다.

현재 활성 run:

- `ai_model/experiments/runs/20260525_003630_stage3_monthly_model_search`

## 6. 서버 적용 방식

`app/ai_server.py`는 다음 순서로 값을 만든다.

1. 필터 조건으로 대상 자산을 좁힌다.
2. 자산 수명 모델로 개별 자산의 총수명을 예측한다.
3. 운용연차를 빼서 RUL과 고장예측일을 계산한다.
4. 월별 수요 모델로 월별 고장 예상 수량을 보정한다.
5. 리스크 성향에 따라 안전재고를 더해 필요수량을 계산한다.
6. 리드타임과 버퍼를 반영해 ROP와 권장발주기한을 산출한다.
7. LLM이 전략적 조달 가이드 문장을 생성한다.

즉, UI에 보이는 값은 `모델 예측값 + 규칙 기반 조달 계산 + LLM 설명`의 결합 결과다.

## 7. 현재 산출물

| 산출물 | 경로 |
| --- | --- |
| 자산 수명 모델 탐색 전체 결과 | `ai_model/experiments/outputs/tables/stage2_life_model_search_results.csv` |
| 자산 수명 모델 최종 후보 결과 | `ai_model/experiments/outputs/tables/stage2_life_model_search_top_test.csv` |
| 월별 수요 모델 탐색 전체 결과 | `ai_model/experiments/outputs/tables/stage3_monthly_model_search_results.csv` |
| 월별 수요 모델 최종 후보 결과 | `ai_model/experiments/outputs/tables/stage3_monthly_model_search_top_test.csv` |
| 월별 수요 모델 발표용 요약 | `ai_model/experiments/outputs/tables/stage3_monthly_demand_results.csv` |
| 현재 배포 자산 수명 모델 | `ai_model/saved_models/current/model.pkl` |
| 현재 발표용 요약 지표 | `ai_model/results/presentation_metrics_summary.csv` |
| 발표용 그래프 | `ai_model/results/plots` |

## 8. 다음 개선 후보

현재 단계에서는 모델 구조를 더 복잡하게 늘리기보다, 발표와 운영 관점에서 다음을 우선한다.

- 합성 데이터 생성 근거를 문서화해 “왜 이런 패턴을 넣었는지” 설명 가능하게 만들기
- 자산 수명 모델과 월별 수요 모델의 목적 차이를 명확히 설명하기
- 교수님 발표에서는 RMSE만이 아니라 권장발주일/안전재고/필요수량으로 이어지는 업무 활용성을 함께 보여주기
- 실제 학교 데이터가 확보되면 같은 스크립트로 재학습 가능하다는 점을 강조하기
