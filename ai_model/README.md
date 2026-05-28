# ai_model

AI 팀 모델링 작업의 현재 기준 폴더 구조다.

- `docs/` - 실험 계획과 서버 인터페이스 계약 문서
- `notebooks/` - 현재 참고용 노트북 workflow
- `experiments/scripts/` - 현재 활성 모델링/그래프 생성 스크립트
- `experiments/outputs/tables/` - 최신 CSV 성능표
- `experiments/outputs/reports/` - 최신 markdown 결과 요약
- `experiments/runs/` - 현재 활성 run artifact
- `results/plots/` - 발표용 그래프 이미지
- `saved_models/current/` - 서버가 읽는 자산 수명 모델
- `archive/obsolete/` - 이전 실험 파일과 대체된 산출물

현재 모델링 기준은 `자산 수명 모델 + 월별 수요 모델 + 서버 조달 계산 로직` 분리 구조다.
