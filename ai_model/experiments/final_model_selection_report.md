# Final Model Selection Report

- Run ID: `20260524_121229`
- Run dir: `ai_model\experiments\runs\20260524_121229_stage4_6_final_validation`
- Acquisition-date holdout cutoff: `2016-11-13`

## Stage 4. Ensemble Validation

| model | rmse_months | mae_months | r2 | term_precision | term_recall | term_f1 | elapsed_sec | stage | source_artifact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees | 17.7531 | 13.5592 | 0.8395 | 1.0000 | 0.7758 | 0.8738 | 0.1240 | stage4_test_artifact | C:\Users\Hwang_Yulim\U-sto_AI\ai_model\experiments\runs\20260524_115653_stage2_valid_tuning\extratrees |
| AverageTop2 | 17.8912 | 13.7364 | 0.8370 | 1.0000 | 0.7853 | 0.8798 | 0.0000 | stage4_test_ensemble | stage2_artifact_predictions |
| AverageTop3 | 17.9908 | 13.8287 | 0.8352 | 1.0000 | 0.7908 | 0.8832 | 0.0000 | stage4_test_ensemble | stage2_artifact_predictions |
| AverageAll4 | 18.0760 | 13.9040 | 0.8336 | 1.0000 | 0.7921 | 0.8840 | 0.0000 | stage4_test_ensemble | stage2_artifact_predictions |
| WeightedByValidRMSE | 18.0801 | 13.9088 | 0.8335 | 1.0000 | 0.7935 | 0.8848 | 0.0000 | stage4_test_ensemble | stage2_artifact_predictions |
| XGBoost | 18.2357 | 14.0508 | 0.8307 | 1.0000 | 0.7976 | 0.8874 | 0.0085 | stage4_test_artifact | C:\Users\Hwang_Yulim\U-sto_AI\ai_model\experiments\runs\20260524_115653_stage2_valid_tuning\xgboost |
| RandomForest | 18.2867 | 14.1131 | 0.8297 | 1.0000 | 0.7962 | 0.8865 | 0.0613 | stage4_test_artifact | C:\Users\Hwang_Yulim\U-sto_AI\ai_model\experiments\runs\20260524_115653_stage2_valid_tuning\randomforest |
| CatBoost | 18.5055 | 14.3118 | 0.8256 | 1.0000 | 0.8030 | 0.8907 | 0.0047 | stage4_test_artifact | C:\Users\Hwang_Yulim\U-sto_AI\ai_model\experiments\runs\20260524_115653_stage2_valid_tuning\catboost |

## Additional Validation. Recent Acquisition Holdout

| model | rmse_months | mae_months | r2 | term_precision | term_recall | term_f1 | elapsed_sec | stage | holdout_cutoff | train_rows | holdout_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees | 15.9288 | 12.5267 | 0.5042 | 1.0000 | 0.5195 | 0.6838 | 0.7531 | acquisition_recent_holdout | 2016-11-13 | 5738 | 1436 |
| AverageTop2 | 15.9344 | 12.5529 | 0.5039 | 1.0000 | 0.5202 | 0.6844 | 0.0000 | acquisition_recent_holdout_ensemble | 2016-11-13 | 5738 | 1436 |
| AverageAll4 | 15.9522 | 12.5696 | 0.5027 | 1.0000 | 0.5195 | 0.6838 | 0.0000 | acquisition_recent_holdout_ensemble | 2016-11-13 | 5738 | 1436 |
| WeightedByValidRMSE | 15.9524 | 12.5701 | 0.5027 | 1.0000 | 0.5202 | 0.6844 | 0.0000 | acquisition_recent_holdout_ensemble | 2016-11-13 | 5738 | 1436 |
| AverageTop3 | 15.9671 | 12.5788 | 0.5018 | 1.0000 | 0.5160 | 0.6808 | 0.0000 | acquisition_recent_holdout_ensemble | 2016-11-13 | 5738 | 1436 |
| CatBoost | 16.0529 | 12.6204 | 0.4964 | 1.0000 | 0.5195 | 0.6838 | 3.7028 | acquisition_recent_holdout | 2016-11-13 | 5738 | 1436 |
| RandomForest | 16.1428 | 12.7236 | 0.4908 | 1.0000 | 0.5077 | 0.6734 | 0.4023 | acquisition_recent_holdout | 2016-11-13 | 5738 | 1436 |
| XGBoost | 16.1537 | 12.7403 | 0.4901 | 1.0000 | 0.5160 | 0.6808 | 0.4441 | acquisition_recent_holdout | 2016-11-13 | 5738 | 1436 |

## Stage 5. Procurement Smoke Test

```json
{
  "prediction_rows": 9126,
  "negative_raw_rul_count": 4176,
  "min_rul_months": 0.5,
  "median_rul_months": 6.504468046570466,
  "max_rul_months": 182.59997007277886,
  "term_6m_failure_count": 4533,
  "term_12m_failure_count": 4845,
  "first_failure_month": "2026-02-01",
  "last_failure_month": "2041-04-01"
}
```

## Server Compatibility Check

```json
{
  "app_feature_count": 9,
  "model_feature_count": 11,
  "app_features": [
    "내용연수",
    "취득금액",
    "부서가혹도",
    "가격민감도",
    "장비중요도",
    "G2B목록명_Code",
    "물품분류명_Code",
    "운용부서코드_Code",
    "캠퍼스_Code"
  ],
  "model_features": [
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
    "캠퍼스_Code"
  ],
  "missing_in_app": [
    "리드타임등급",
    "취득월"
  ],
  "extra_in_app": [],
  "app_uses_model_meta_features": false,
  "app_rul_postprocess_expected": "현재 서버는 predict 결과를 현재일 + 예측수명_월로 바로 더한다. 총수명 모델이면 운용연차*12를 빼는 RUL 계산이 필요하다."
}
```

## Stage 6. Selection

- Selected artifact: `C:\Users\Hwang_Yulim\U-sto_AI\ai_model\experiments\runs\20260524_115653_stage2_valid_tuning\extratrees`
- Selected model: `ExtraTrees`
- Deployment was not overwritten by this script.
- Before replacing `rf_final_model.pkl`, align server inference with `model_meta.json` features and RUL postprocessing.
