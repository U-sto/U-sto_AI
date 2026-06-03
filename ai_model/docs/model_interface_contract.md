# 모델-서버 인터페이스 계약서

작성일: 2026-05-25

## 1. 목적

이 문서는 모델을 교체해도 `app/ai_server.py`와 UI 요청/응답 흐름이 깨지지 않도록, 현재 서버가 기대하는 모델 artifact와 입력/출력 규칙을 정리한 문서다.

현재 구조는 두 모델을 함께 사용한다.

| 모델 | 역할 | artifact |
| --- | --- | --- |
| 자산 수명 모델 | 개별 자산 총수명 예측 | `ai_model/saved_models/current/model.pkl` |
| 월별 수요 모델 | 월별 고장/처분 예상 수량 예측 | 최신 `*_stage3_monthly_model_search/monthly_demand_model.pkl` |

## 2. 자산 수명 모델 계약

### 2.1. 파일 위치

- 모델: `ai_model/saved_models/current/model.pkl`
- 메타데이터: `ai_model/saved_models/current/model_meta.json`

### 2.2. 현재 모델

- 모델: CatBoost
- variant: `cb_fast_shallow`
- feature set: `importance_top_15`
- target: `실제수명_개월`
- target 의미: 개별 자산의 총수명(months)

### 2.3. 현재 입력 feature

`model_meta.json`의 `features` 값을 서버가 읽는다. 현재 feature는 다음 15개다.

1. 내용연수
2. 부서가혹도
3. 월평균사용시간
4. 사용강도지수
5. 누적점검수리횟수
6. 누적수리횟수
7. 최근2년수리횟수
8. 마지막수리후경과개월
9. 취득금액대비수리비율
10. 최대장애심각도
11. 부서예산등급_Code
12. 부서교체성향
13. G2B목록명_Code
14. 물품분류명_Code
15. 운용부서코드_Code

서버는 이 순서에 맞춰 DataFrame을 구성해야 한다. feature가 없으면 0 또는 전처리 기본값으로 채우되, 가능하면 `phase4_training_data.csv`가 같은 컬럼을 갖도록 유지한다.

### 2.4. 출력값

모델 출력은 총수명 개월 수다.

서버 후처리:

```text
예측잔여수명 = 예측총수명개월 - 운용연차개월
AI예측고장일 = 취득일자 + 예측총수명개월
```

따라서 모델 자체가 권장발주일이나 안전재고를 직접 예측하지 않는다.

## 3. 월별 수요 모델 계약

### 3.1. 파일 위치

월별 수요 모델은 최신 run 폴더에서 자동 탐색한다.

- 모델: `ai_model/experiments/runs/*_stage3_monthly_model_search/monthly_demand_model.pkl`
- 메타데이터: `monthly_model_meta.json`

서버는 최신 수정 시간을 기준으로 가장 최신 월별 모델을 읽는다.

### 3.2. 현재 모델

- 모델: ExtraTrees
- variant: `et_regularized`
- feature set: `seasonal_7`
- target: 월별 고장/처분 발생량

### 3.3. 현재 입력 feature

현재 월별 수요 모델 feature는 다음 7개다.

1. trend
2. month
3. month_sin
4. month_cos
5. lag_12
6. rolling_mean_6
7. rolling_std_6

서버는 `monthly_model_meta.json`의 `features` 값을 읽어서 입력을 맞춘다. 추후 월별 모델을 XGBoost나 Ridge로 바꿔도 meta의 feature list만 맞으면 서버 수정 없이 동작하도록 설계한다.

### 3.4. 검증 기준

월별 모델 검증은 recursive backtest를 기준으로 한다.

테스트 구간의 실제 미래 lag를 직접 넣지 않고, 이전 예측값을 다음 달 lag로 사용한다. 이렇게 해야 실제 서비스 상황과 유사하다.

## 4. 서버 계산값

UI에 표시되는 주요 값은 아래처럼 만들어진다.

| UI/응답 값 | 산출 방식 |
| --- | --- |
| 예측잔여수명 | 자산 수명 모델 출력 - 운용연차 |
| AI예측고장일 | 취득일자 + 예측 총수명 |
| 고장 예상 수량 | 월별 수요 모델 예측 및 필터링 결과 |
| 안전재고 | 리스크 성향별 서비스 계수/버퍼 적용 |
| 필요수량 | 고장 예상 수량 + 안전재고 |
| 발주시점(ROP) | 수요 피크와 리드타임을 역산 |
| 권장발주기한 | 설치 완료 목표일 - 리드타임 - 버퍼 |

즉, 모델 artifact가 직접 반환하는 값과 UI에 표시되는 값은 다르다. UI 값은 모델 출력 후 서버의 조달 계산 로직을 거쳐 만들어진다.

## 5. 필터 입력

초기 검색/분석 조건에서 서버가 받는 주요 필터는 다음과 같다.

- 년도
- 캠퍼스
- 운용부서
- 물품분류명
- 리스크 성향

필터는 모델 입력 feature라기보다 대상 자산과 월별 집계 범위를 좁히는 조건이다. 단, `운용부서코드_Code`, `물품분류명_Code`처럼 인코딩된 값은 자산 수명 모델 feature로도 쓰일 수 있다.

## 6. 누수 방지 원칙

다음 컬럼은 모델 feature로 사용하지 않는다.

- 불용일자
- 처분방식
- 물품상태
- 불용사유
- 실제수명
- 실제잔여수명
- 예측잔여수명
- AI예측고장일
- 권장발주일
- 데이터세트구분
- 학습데이터여부
- 월별 고장/필요수량 결과 컬럼

이 컬럼들은 학습 정답이거나, 예측 이후에 생성되는 결과값이다. feature에 포함되면 데이터 누수가 발생한다.

## 7. 모델 교체 체크리스트

새 모델로 교체할 때는 아래 조건을 만족해야 한다.

1. `model.pkl`과 `model_meta.json`을 함께 저장한다.
2. `model_meta.json`에 `features`, `target`, `metrics`를 기록한다.
3. 서버가 읽는 feature 이름과 데이터셋 컬럼명이 일치해야 한다.
4. 출력 단위는 자산 수명 모델의 경우 개월(months)이어야 한다.
5. 월별 모델은 `monthly_model_meta.json`의 feature list와 실제 모델 입력 순서가 같아야 한다.
6. 누수 컬럼이 feature에 포함되지 않았는지 확인한다.
7. `app/ai_server.py` import smoke test를 통과해야 한다.
