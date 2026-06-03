# RAG 평가 플롯 해석 리포트

생성일: 2026-05-26

## 실행 상태 요약

- `evaluation/generate_rag_panel_plots.py`: 정상 완료. `results/rag_panel`의 5개 plot과 summary/csv가 갱신됨.
- `evaluation/generate_llm_judge_eval.py --force`: 64개 중 23개 평가 후 OpenAI `insufficient_quota`로 중단됨.
- `results/llm_judge_partial`: 중단 전 생성된 23개 raw 결과를 별도 partial plot으로 정리함.
- `evaluation/run_chain_diagnostics.py`: OpenAI quota가 필요한 운영 체인 진단 평가라 이번 실행에서는 보류함.

## 2026-05-26 추가 실행 기록

원문 매뉴얼(`dataset/input`), FAQ(`dataset/FAQ`), domain guide를 포함하도록 `scripts_/create_vector_db.py`를 수정한 뒤 새 Chroma DB 재생성을 시도했습니다.

새 DB에 들어갈 예정인 문서 구성:

- QA 문서: 64개
- 원문 매뉴얼 문서: 77개
- FAQ 문서: 10개
- 도메인 구분 가이드 문서: 3개
- 총 문서 수: 154개

실행 결과:

- Chroma DB 재생성은 OpenAI embedding API 단계에서 `insufficient_quota`로 실패함.
- DB 생성 스크립트는 임시 DB를 먼저 만든 뒤 성공 시에만 기존 DB와 교체하도록 수정되어 있어, 기존 `chroma_db`는 보존됨.
- 따라서 현재 `results/rag_panel`은 새 원문/FAQ 포함 DB가 아니라 기존 268개 QA DB 기준 결과임.
- `evaluation/generate_llm_judge_eval.py --sample-size 1`도 embedding API 단계에서 동일한 quota 오류로 실패함.
- `evaluation/run_chain_diagnostics.py --limit 1`은 실행 파일은 생성했지만, tool routing/chat/query embedding 모두 quota 오류를 받아 실제 검색/생성 진단값은 비어 있고 기술 오류 응답만 기록됨.

quota가 해소되면 아래 순서로 다시 실행해야 새 코드 개선 효과를 평가할 수 있습니다.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' scripts_\create_vector_db.py
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' evaluation\generate_rag_panel_plots.py
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' evaluation\generate_llm_judge_eval.py --force
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' evaluation\run_chain_diagnostics.py --limit 64
```

주의: `results/llm_judge`의 기존 `llm_judge_summary.md`, `llm_judge_results.csv`는 이전 64개 전체 평가 결과이며, 이번 `--force` 재실행은 23개에서 중단되었습니다. 따라서 이번 리포트에서는 `rag_panel`은 최신 전체 결과, `llm_judge_partial`은 최신 부분 결과로 구분해 해석합니다.

## 1. RAG Panel Plots

대상 폴더: `results/rag_panel`

### 01_retrieval_consistency_at_k.png

최신 수치:

- Category Hit@1: 88.1%
- Category Hit@5: 95.9%
- Category Precision@5: 78.1%
- Chapter Hit@5: 94.8%

해석:

상위 5개 검색 결과 안에 같은 category 문서가 포함될 확률이 95.9%로 높습니다. 즉, 사용자가 물품 반납/취득/불용 같은 특정 업무 범주의 질문을 했을 때 최소 1개 이상의 같은 범주 문서를 찾는 능력은 안정적인 편입니다.

Category Precision@k가 k가 커질수록 내려가는 것은 정상입니다. 검색 개수를 늘릴수록 관련 문서뿐 아니라 인접 주제 문서도 같이 들어오기 때문입니다. 다만 최종 답변 context에는 top-k 전체가 아니라 더 압축된 문서를 넣어야 하므로, 최근 코드에서 적용한 adaptive filtering과 context diversity가 이 지점에 대응합니다.

패널 해석 문구:

> 상위 5개 검색 후보 기준으로 동일 업무 category 문서 적중률이 95.9%로 높아, 지식베이스의 1차 검색 구조는 안정적입니다. 다만 precision은 k 증가에 따라 하락하므로 최종 context 압축과 reranking 품질이 답변 품질을 좌우합니다.

### 02_knowledge_base_category_composition.png

최신 수치:

- Embedded QA documents: 268개
- Knowledge categories: 13개
- Source manual files: 8개

해석:

현재 Chroma DB에는 268개의 QA 기반 문서가 들어 있습니다. 이 그래프는 실제 LLM judge의 답변 category와는 다른 metadata category 분포를 보여줍니다. 즉, 이 plot은 "어떤 질문 유형에서 hallucination이 발생했는가"보다 "지식베이스가 어떤 문서 라벨로 구성되어 있는가"를 설명하는 용도입니다.

패널 해석 문구:

> 지식베이스는 8개 원천 매뉴얼에서 생성된 268개 QA 문서로 구성되어 있으며, 특정 절차/시스템 중심 문서 비중이 높습니다. 향후 원문 chunk를 함께 저장하면 QA 생성 과정에서 빠진 세부 근거를 보완할 수 있습니다.

### 03_semantic_distance_distribution.png

해석:

초록색 분포가 "같은 category 문서 중 가장 가까운 문서와의 거리", 주황색 분포가 "다른 category 문서 중 가장 가까운 문서와의 거리"입니다. x축은 cosine distance라 낮을수록 가깝습니다.

같은 category 문서가 다른 category 문서보다 더 왼쪽에 몰려 있으면 임베딩 공간에서 같은 업무 문서끼리 잘 모였다는 뜻입니다. 최신 결과에서도 검색 hit 지표가 높게 나왔으므로, 임베딩 구조 자체는 어느 정도 유효하다고 볼 수 있습니다.

다만 category silhouette score가 0.085로 높지는 않습니다. 이는 업무 category가 완전히 분리된 군집이라기보다 서로 의미가 겹치는 문서가 많다는 뜻입니다. 물품 반납/불용/처분처럼 실제 업무상 연결되는 주제들이 섞이는 것은 자연스럽지만, 답변 생성 단계에서는 잘못 섞이지 않도록 context 압축이 중요합니다.

패널 해석 문구:

> 같은 category 문서끼리의 거리가 다른 category 문서보다 전반적으로 가깝게 나타나 검색 기반은 유효합니다. 다만 silhouette score는 0.085로 낮아 업무 주제가 완전히 분리되지는 않으므로 reranking과 context selection이 필요합니다.

### 04_embedding_space_pca.png

해석:

1536차원 임베딩을 2차원으로 축소한 시각화입니다. x축과 y축 자체에 업무적 의미는 없습니다. 색이 비슷한 점들이 어느 정도 모이면 category별 의미 구조가 잡혀 있다는 신호이고, 색이 섞인 구역은 실제 업무 절차가 서로 연결되어 있거나 라벨 기준이 겹친다는 뜻입니다.

이 plot은 성능 점수로 직접 읽기보다 "지식베이스가 완전히 무작위로 흩어지지는 않았는가"를 보여주는 보조 자료로 쓰는 것이 좋습니다.

### 05_rag_pipeline_funnel.png

해석:

현재 설정 구조는 vector retrieval 25개, rerank candidate 15개, final context 10개로 표시됩니다. 다만 최근 운영 코드에서는 최종 context를 내부적으로 최대 6개 수준으로 압축하고, adaptive filter와 diversity selection을 적용하도록 수정했습니다.

따라서 이 plot은 config 기반 구조 설명에 가깝고, 실제 운영 체인에서 최종 몇 개 문서가 들어갔는지는 `run_chain_diagnostics.py`로 확인하는 것이 더 정확합니다.

패널 해석 문구:

> RAG pipeline은 넓게 후보를 가져온 뒤 reranking과 context 압축으로 답변 근거를 좁히는 구조입니다. 최신 운영 코드에서는 잡음 context를 줄이기 위해 final context를 더 보수적으로 선택하도록 개선했습니다.

## 2. Partial LLM Judge Plots

대상 폴더: `results/llm_judge_partial`

이번 재평가는 API quota 문제로 23개 샘플까지만 완료되었습니다. 따라서 아래 수치는 전체 성능 대표값이 아니라, 최신 코드 수정 후 일부 샘플에서 관찰된 중간 결과로 해석해야 합니다.

최신 부분 수치:

- Evaluated samples: 23개
- Faithfulness: 4.61 / 5
- Answer Relevance: 4.70 / 5
- Reference Alignment: 3.35 / 5
- Hallucination Rate: 4.3%
- Retrieved Source Hit Rate: 91.3%
- Retrieved Chapter Hit Rate: 56.5%

기존 전체 64개 기준 참고 수치:

- Faithfulness: 4.67 / 5
- Answer Relevance: 4.77 / 5
- Reference Alignment: 3.88 / 5
- Hallucination Rate: 6.2%
- Retrieved Source Hit Rate: 96.9%
- Retrieved Chapter Hit Rate: 71.9%

### 01_partial_llm_judge_metric_scores.png

해석:

부분 평가 기준으로 Faithfulness 92.2%, Answer relevance 93.9%, Non-hallucination 95.7% 수준입니다. 생성 답변이 검색 문맥에 근거하고 질문에 직접 응답하는 성향은 유지되고 있습니다.

반면 Reference alignment는 67.0% 수준으로 낮습니다. 이는 답변이 완전히 틀렸다는 뜻이라기보다, 생성 답변이 정답 예시의 세부 항목을 충분히 포함하지 못했거나 표현 범위가 좁아졌을 가능성이 큽니다. 최근 코드가 context를 더 보수적으로 압축하도록 바뀌었기 때문에, hallucination은 줄 수 있지만 reference answer의 세부 항목을 덜 포괄할 수 있습니다.

### 02_partial_llm_judge_score_distribution.png

해석:

Faithfulness와 answer relevance는 대부분 높은 점수에 몰려 있습니다. 다만 일부 질문에서 2~3점이 발생해 검색 context가 질문 의도를 충분히 담지 못했거나, 정제 질의가 원 질문의 세부 의미를 덜 반영했을 가능성이 있습니다.

Reference alignment의 분산이 더 큰 편이므로, 향후 개선은 "근거 없는 말을 줄이는 것"과 동시에 "정답 예시의 필수 체크리스트를 빠뜨리지 않는 것"을 같이 봐야 합니다.

### 03_partial_hallucination_rate_by_category.png

부분 평가 category별 결과:

- 물품 반납 관리 (n=7): 14.3%, 1건
- 물품 취득 관리 (n=11): 0%
- 절차 (n=4): 0%
- 식별 (n=1): 0%

해석:

이번 부분 평가에서 hallucination은 물품 반납 관리 1건입니다. 표본이 7개뿐이라 14.3%로 보이지만 실제로는 1건입니다. 전체 재평가가 완료되기 전까지는 category별 위험도를 단정하기 어렵습니다.

패널 해석 문구:

> 부분 재평가 23건 기준 hallucination은 1건이며, 물품 반납 관리에서 발생했습니다. 표본 수가 작으므로 category별 위험 진단은 전체 64건 재평가 완료 후 확정하는 것이 안전합니다.

## 3. 현재 결과에서 보이는 개선 방향

이번 코드 수정 후 가장 먼저 확인해야 할 지점은 다음입니다.

- RAG panel 기준 검색 hit는 충분히 높으므로, 검색 후보 생성보다 최종 context selection 품질이 더 중요합니다.
- partial LLM judge에서 hallucination rate는 낮지만 reference alignment가 낮으므로, context 압축이 과도하게 세부 근거를 줄이지 않는지 확인해야 합니다.
- Chapter hit가 partial 기준 56.5%로 낮게 나왔으므로, chapter metadata boost 또는 원문 chunk 추가가 다음 개선 후보입니다.
- 물품 반납/불용/처분처럼 의미가 겹치는 category는 hybrid retrieval 또는 metadata 기반 reranking이 필요합니다.

## 4. 다음 실행 권장 사항

OpenAI quota가 해소되면 아래 순서로 다시 실행하는 것이 좋습니다.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' evaluation\generate_llm_judge_eval.py --force
& 'C:\Users\Hwang_Yulim\U-sto_AI\.venv\Scripts\python.exe' evaluation\run_chain_diagnostics.py --limit 64
```

그 후 확인할 파일:

- `results/llm_judge/01_llm_judge_metric_scores.png`
- `results/llm_judge/02_llm_judge_score_distribution.png`
- `results/llm_judge/03_hallucination_rate_by_category.png`
- `results/chain_diagnostics/chain_diagnostics.csv`
- `results/chain_diagnostics/chain_diagnostics.jsonl`

특히 `chain_diagnostics.csv`의 `retrieved_count`, `filtered_count`, `final_context_count`, `attribution_doc_ids`를 보면 최근 수정한 adaptive filtering과 context diversity가 실제 질문별로 어떻게 작동했는지 확인할 수 있습니다.
