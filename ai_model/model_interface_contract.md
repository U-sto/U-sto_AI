# 사용주기 AI 모델 교체 계약서

이 문서는 `02_Modeling_RF.ipynb`에서 만든 사용주기 예측 모델을 다른 모델로 교체할 때 서버, 후처리, UI 연동부를 최대한 수정하지 않기 위해 유지해야 하는 변수명과 산출물 규칙을 정리한 것이다.

기준 파일: `ai_model/notebooks/02_Modeling_RF.ipynb`

## 1. 로컬 실행 환경

VSCode에서는 프로젝트 루트의 `venv`를 사용한다.

```powershell
.\venv\Scripts\python.exe
```

노트북 커널은 다음 이름으로 등록되어 있다.

```text
U-sto_AI (venv)
```

설치 확인이 끝난 주요 패키지는 다음과 같다.

- `pandas`, `numpy`
- `scikit-learn`, `joblib`, `scipy`
- `matplotlib`, `seaborn`
- `statsmodels`, `pmdarima`
- `xgboost`, `lightgbm`, `catboost`
- `optuna`

## 2. 입력 데이터 계약

기본 데이터 경로는 아래를 유지한다.

```python
DATA_PATH = "dataset/create_data/data_ml/phase4_training_data.csv"
```

모델 학습과 예측에는 `데이터세트구분` 컬럼을 사용한다.

- 학습: `Train`, `Valid`
- 평가: `Test`
- 실제 예측 대상: `Prediction`

모델이 입력으로 받는 feature 컬럼은 다음 목록을 기준으로 통일한다.

```python
initial_features = [
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

서버 교체 안정성을 위해 새 모델도 기본적으로 이 feature 목록을 그대로 받는 구조가 가장 안전하다. feature를 줄이거나 늘릴 경우에는 반드시 `model_meta.json`의 `features` 값을 함께 갱신해야 한다.

## 3. 타깃 계약

현재 RF 노트북은 총 수명을 개월 단위로 학습한다.

```python
df["실제수명_개월"] = df["실제수명"] * 12
target = "실제수명_개월"
```

중요한 점은 모델이 직접 예측해야 하는 값이 `RUL_개월`이 아니라 `예상 총 수명 개월`이라는 것이다. 후처리 함수 `predict_assets()`가 아래 계산을 수행한다.

```python
RUL_개월 = 모델이_예측한_총수명_개월 - 운용연차 * 12
```

따라서 새 모델이 잔여수명 `RUL_개월`을 직접 예측하도록 만들면 기존 후처리에서 운용기간이 한 번 더 빠져 결과가 틀어진다. RUL 직접 예측 모델을 쓰고 싶다면 서버 교체 없이 쓸 수 있도록 `predict()` 단계에서 다시 총 수명 개월 형태로 감싸는 wrapper가 필요하다.

## 4. 모델 산출물 계약

서버와 후처리 코드를 수정하지 않으려면 최종 배포 모델은 기존 파일명 규칙을 유지한다.

```text
ai_model/saved_models/random_forest/rf_final_model.pkl
ai_model/saved_models/random_forest/model_meta.json
```

`rf_final_model.pkl`에는 `joblib.load()`로 읽을 수 있고, `predict(X)` 메서드를 가진 객체를 저장한다. RandomForest가 아니더라도 이 파일명으로 저장하면 기존 호출부를 그대로 사용할 수 있다.

`model_meta.json`에는 최소한 아래 키가 필요하다.

```json
{
  "features": ["내용연수", "취득금액"],
  "rmse_months": 3.06
}
```

권장 확장 키는 다음과 같다.

```json
{
  "model_name": "LightGBMRegressor",
  "target": "실제수명_개월",
  "features": [],
  "rmse_months": 0.0,
  "mae_months": 0.0,
  "r2": 0.0,
  "trained_at": "YYYY-MM-DD HH:mm:ss",
  "data_path": "dataset/create_data/data_ml/phase4_training_data.csv",
  "split_rule": "Train+Valid train, Test evaluation, Prediction inference"
}
```

기존 `run_procurement_pipeline()`는 `rmse_months`, `rmse`, `rmse_days` 순서로 오차 값을 읽는다. 단위 혼선을 막기 위해 앞으로는 `rmse_months`를 표준으로 쓴다.

## 5. 예측 함수 계약

모델 교체 후에도 아래 흐름은 유지되어야 한다.

```python
model = load_model(model_path)
X_pred = df_target[features]
df_target["RUL_개월"] = predict_assets(model, X_pred, df_target["운용연차"])
df_target["AI예측고장일"] = df_target["RUL_개월"].apply(lambda x: calculate_failure_date(x, today_date))
df_target["포트폴리오_영역"] = df_target.apply(lambda r: assign_matrix_zone(r["RUL_개월"], r["장비중요도"]), axis=1)
df_target["고장예상여부"] = df_target["RUL_개월"].apply(lambda x: classify_term_failure(x, term_months))
```

`model.predict(X_pred)`의 반환값은 입력 행 개수와 같은 길이의 numeric array여야 한다.

## 6. 후처리 입력 변수

LLM 또는 UI에서 넘어오는 분석 조건은 아래 변수명으로 매핑한다.

```python
today_date = "YYYY-MM-DD"
target_term = "1학기" | "여름계절학기" | "2학기" | "겨울계절학기"
service_factor = 1.0 이상 float
service_level = 0.95
```

현재 학기별 기간 매핑은 다음과 같다.

```python
term_map = {
    "1학기": 3,
    "여름계절학기": 6,
    "2학기": 9,
    "겨울계절학기": 12,
}
```

## 7. 출력 데이터프레임 계약

`run_procurement_pipeline()`는 세 개의 데이터프레임을 반환한다.

```python
df_matrix_asset, df_ts_demand, df_plan = run_procurement_pipeline(...)
```

`df_matrix_asset`는 자산 단위 결과이며, 최소한 아래 파생 컬럼이 필요하다.

- `RUL_개월`
- `AI예측고장일`
- `포트폴리오_영역`
- `고장예상여부`

`df_ts_demand`는 시계열 그래프용 월별 집계 결과이다.

- `고장예상연월`
- `예상고장수량`
- `필요수량`
- `발주권장연월`

`df_plan`은 조달 권고안 표에 표시되는 결과이다.

- `품목명`
- `권장구매수량(필요수량)`
- `추정예산`
- `권장발주기한`
- `AI코멘트`

## 8. 누수 방지 규칙

모델 학습 feature에는 예측 시점에 알 수 없는 사후 결과 컬럼을 넣지 않는다.

사용 금지 예시는 다음과 같다.

- `불용일자`
- `처분방식`
- `물품상태`
- `불용사유`
- `실제수명`
- `실제수명_개월`
- `실제잔여수명`
- `예측잔여수명`
- `(월별)고장예상수량`
- `안전재고`
- `(월별)필요수량`
- `AI예측고장일`
- `안전버퍼`
- `권장발주일`

## 9. 모델 교체 체크리스트

1. 새 모델이 `initial_features`와 같은 입력 컬럼을 받는지 확인한다.
2. 새 모델의 `predict()`가 총 수명 개월을 반환하는지 확인한다.
3. `rf_final_model.pkl`을 `joblib.dump()`로 저장한다.
4. `model_meta.json`에 `features`와 `rmse_months`를 저장한다.
5. `run_procurement_pipeline()`를 한 번 실행해 세 결과값이 생성되는지 확인한다.
6. 기존 UI가 기대하는 컬럼명과 날짜 포맷이 바뀌지 않았는지 확인한다.
