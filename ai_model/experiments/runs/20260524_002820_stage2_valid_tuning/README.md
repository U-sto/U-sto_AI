# Stage 2 Valid Tuning Results

Best model: CatBoost

| Model | Valid RMSE | Test RMSE | Test MAE | Test R2 | Term F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 9.2729 | 3.4684 | 1.4099 | 0.8973 | 0.9929 |
| XGBoost | 9.2295 | 3.8071 | 1.5028 | 0.8762 | 0.9901 |
| RandomForest | 9.9537 | 3.9139 | 1.5801 | 0.8692 | 0.9893 |
| ExtraTrees | 9.8016 | 3.9447 | 1.6052 | 0.8671 | 0.9893 |

Selection rule: Train split was used for parameter search, then the selected model was retrained on Train+Valid and evaluated once on Test.

