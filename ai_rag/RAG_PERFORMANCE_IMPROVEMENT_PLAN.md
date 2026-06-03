# RAG 성능 개선 방향 정리

## 1. 현재 코드 진단 요약

현재 `ai_rag` 코드는 크게 `ingestion -> embedding/vectorstore -> retrieval/rerank -> prompt/generation -> evaluation` 흐름으로 구성되어 있다. 구조 자체는 단순하고 이해하기 좋지만, 성능 테스트 결과에서 아쉬움이 남는 이유는 검색 모델 하나의 문제라기보다 다음 요소들이 함께 영향을 주는 것으로 보인다.

- 질문 분류기가 RAG가 필요한 업무 절차 질문을 `NO_RAG`로 보낼 가능성이 있다.
- query refinement 프롬프트가 사실상 `...`만 있어 검색어 개선 효과가 안정적이지 않다.
- similarity threshold가 `10.0`으로 매우 넉넉해서 실제 필터 역할을 거의 하지 않는다.
- rerank 이후 최종 context가 10개까지 들어가며, 중복/잡음 문서가 답변에 섞일 수 있다.
- 지식베이스가 LLM이 생성한 QA 문서 중심이라 원문 매뉴얼의 세부 절차나 예외 조건이 빠질 수 있다.
- 평가 스크립트는 운영 RAG 체인을 그대로 평가하지 않고 별도의 간이 retrieval/generation 루프를 사용한다.
- 테스트 코드 일부가 현재 구현과 맞지 않아 성능 개선 전후의 회귀 확인 수단으로 쓰기 어렵다.

따라서 개선 우선순위는 `프롬프트 라우팅 안정화 -> 검색 품질 개선 -> context 압축/근거 강화 -> 평가 체계 정비` 순서가 적절하다.

## 2. 우선 개선해야 할 코드 포인트

### 2.1 질문 분류기 기준 재정의

파일: `rag/prompt.py`, `rag/chain.py`

현재 `build_question_classifier_prompt()`는 질문을 "시스템 데이터 조회"와 "일반 설명/절차/정책 질문"으로 나누라고 하지만, 어떤 경우가 `NEED_RAG`인지 명확하지 않다. 실제로 물품 반납, 불용, 취득, 처분 같은 업무 절차 질문은 매뉴얼 기반 RAG가 필요한데, 모델이 이를 `NO_RAG`로 판단하면 context 없이 답하게 된다.

개선 방향:

- `NEED_RAG`: 물품 취득/반납/불용/처분/등록/관리/규정/절차/시스템 사용법 등 업무 지식 질문
- `NO_RAG`: 인사말, 잡담, 챗봇 사용법, 질문 재작성 요청 등 도메인 문서가 필요 없는 질문
- 출력값은 JSON 또는 단일 토큰으로 고정하고, `classification.strip().upper()`처럼 방어적으로 파싱
- 분류 실패 시 기본값은 `NEED_RAG`로 두는 것이 안전함

예상 효과:

- 절차/반납/불용 질문에서 context 없는 답변이 줄어 hallucination 위험 감소
- 업무 도메인 질문의 답변 일관성 향상

### 2.2 Query refinement 프롬프트 보강

파일: `rag/prompt.py`

현재 `build_query_refine_prompt()`의 본문에 `...`가 들어 있어 검색어 변환 규칙이 없다. query rewrite 단계가 오히려 원 질문의 핵심 키워드를 누락하거나, 불필요한 설명을 붙여 검색 성능을 떨어뜨릴 수 있다.

개선 방향:

- 원 질문의 핵심 명사와 업무 용어를 보존
- 반납/불용/처분/취득/등록/검수/관리전환 등 도메인 용어를 표준화
- 출력은 검색 질의 한 줄만 허용
- 자산번호, G2B 목록번호, 물품고유번호는 삭제하지 않음
- 모호한 경우 원 질문을 최대한 유지

추가로, query refinement가 항상 도움이 되지는 않으므로 다음 3개 검색을 함께 비교하는 실험이 필요하다.

- 원 질문 검색
- refined query 검색
- 원 질문 + refined query 앙상블 검색

### 2.3 Similarity threshold 재설정

파일: `app/config.py`, `rag/chain.py`

현재 `SIMILARITY_SCORE_THRESHOLD = 10.0`이다. Chroma `similarity_search_with_score()`의 score는 보통 낮을수록 가까운 거리 점수인데, 10.0이면 대부분의 문서가 통과할 가능성이 높다. 결과적으로 threshold filter가 성능 방어막 역할을 하지 못한다.

개선 방향:

- 실제 검색 score 분포를 로그/CSV로 수집
- 정답 문서가 포함된 경우와 오답 문서만 검색된 경우의 score 분포를 비교
- 고정 threshold 대신 다음 중 하나를 선택
  - 보수적 고정 threshold
  - top-1과 top-k의 score gap 기반 필터
  - reranker score 기반 필터
  - 최소 근거 수 미달 시 abstain 처리

예상 효과:

- 관련 없는 문서가 context에 들어가는 비율 감소
- 근거 부족 질문에서 억지 답변 감소

### 2.4 Context 개수와 구성 방식 조정

파일: `rag/chain.py`, `app/config.py`

reranking을 사용할 때 최종 context는 `RERANK_TOP_N = 10`개이다. 현재 지식베이스가 QA 문서 268개 규모라 top 10은 꽤 많은 편이며, 서로 비슷한 문서 또는 약간 다른 절차 문서가 함께 들어가 답변이 흔들릴 수 있다.

개선 방향:

- 최종 context 후보를 4개, 6개, 8개, 10개로 실험
- 같은 `source/chapter/category`의 중복 문서는 diversity 기준으로 제한
- context block에 `doc_id`, `title`, `chapter`, `category`, retrieval score, rerank score를 함께 넣기
- 답변 생성 프롬프트에서 "근거 문서에 없는 세부 절차는 답하지 말 것"을 더 강하게 지시
- 답변 끝에 사용 근거 doc_id 또는 title을 내부 attribution으로 유지

예상 효과:

- 잡음 context 감소
- reference alignment 개선
- 답변 근거 추적성 향상

### 2.5 지식베이스 구축 방식 개선

파일: `ingestion/loader.py`, `ingestion/qa_converter.py`, `scripts_/create_vector_db.py`

현재 DB에는 LLM이 생성한 QA 형태의 문서가 들어간다. 이 방식은 사용자 질문과 유사한 QA를 찾기 쉬운 장점이 있지만, 원문 매뉴얼의 표, 예외 조건, 단계 순서, 조건문이 QA 생성 과정에서 손실될 수 있다.

개선 방향:

- QA 문서만 저장하지 말고 원문 chunk도 함께 저장
- 각 원문 chunk에 `source`, `chapter`, `title`, `category`, `section_path`, `chunk_index`를 부여
- QA 문서는 "질문 매칭용", 원문 chunk는 "근거 생성용"으로 역할 분리
- 검색은 QA와 원문 chunk를 모두 대상으로 하되, 최종 답변 context에는 원문 chunk를 우선 포함
- 카테고리 라벨은 LLM 자유 생성 대신 고정 taxonomy로 정규화

추천 구조:

- `doc_type=qa`: 질문 의도 매칭용
- `doc_type=manual_chunk`: 답변 근거용
- `doc_type=faq`: FAQ 보강용

예상 효과:

- 세부 절차 누락 감소
- 정답 예시와 표현이 다르더라도 핵심 근거가 더 잘 살아남음
- reference alignment 개선 가능성

### 2.6 Hybrid retrieval 추가

파일: `vectorstore/retriever.py`, 신규 retriever 모듈

현재 검색은 dense vector similarity 중심이다. 한국어 행정 문서에서는 "불용", "반납", "관리전환", "내용연수", "G2B"처럼 키워드 자체가 강한 신호인 경우가 많다. dense retrieval만 쓰면 의미는 비슷하지만 다른 절차 문서가 섞일 수 있다.

개선 방향:

- dense vector 검색 + BM25/키워드 검색을 함께 사용
- 결과를 RRF(Reciprocal Rank Fusion)로 합치기
- 도메인 키워드가 있는 경우 category/chapter metadata boost 적용
- "반납 vs 불용", "취득 vs 등록", "처분 vs 불용"처럼 헷갈리는 쌍은 룰 기반 boost 또는 negative hint 추가

예상 효과:

- 용어가 명확한 질문에서 top-k precision 향상
- 카테고리 혼동 감소

### 2.7 FAQ 검색 방식 개선

파일: `rag/faq_service.py`

현재 FAQ는 키워드 포함 여부로만 매칭한다. 작고 고정된 FAQ에는 괜찮지만, 표현이 조금 달라지면 놓치고, 반대로 키워드 하나만 맞아도 과하게 들어갈 수 있다.

개선 방향:

- FAQ도 embedding 대상에 포함
- 키워드 매칭은 high precision shortcut으로 유지
- FAQ가 context에 들어갈 때는 source를 `faq`로 명확히 표시
- FAQ와 매뉴얼이 충돌하면 매뉴얼 또는 최신 source 우선순위를 명시

### 2.8 Tool routing과 RAG routing 분리

파일: `rag/chain.py`, `rag/prompt.py`

현재 실행 흐름은 먼저 tool call 가능 LLM을 호출하고, tool call이 없으면 RAG 분류기로 넘어간다. 이 구조는 괜찮지만, 최종 답변 프롬프트에 `Function Calling 판단 기준`까지 들어가면 생성 모델이 답변 대신 "함수 호출이 필요함" 같은 판단 문구를 섞을 위험이 있다.

개선 방향:

- tool routing 전용 prompt와 답변 생성 prompt를 완전히 분리
- 답변 생성 단계에서는 function decision prompt를 제거하거나 최소화
- tool 결과 + RAG context를 함께 써야 하는 혼합 질문을 명시적으로 지원
- tool call 실패 시 RAG로 넘어갈지, 사용자에게 조회 실패를 안내할지 정책 분리

## 3. 평가 체계 개선

### 3.1 운영 체인 기준 end-to-end 평가 추가

파일: `evaluation/generate_llm_judge_eval.py`

현재 LLM judge 평가는 별도의 `_retrieve_context()`와 `_generate_rag_answer()`를 사용한다. 운영 코드의 `run_rag_chain()`에 있는 분류기, query refinement, threshold, reranker, prompt 조합이 평가에 반영되지 않는다.

개선 방향:

- `run_rag_chain()`을 직접 호출하는 평가 스크립트 추가
- 각 샘플마다 다음 로그를 저장
  - classification
  - refined_query
  - retrieved doc ids and scores
  - filtered doc ids
  - reranked doc ids and scores
  - final context doc ids
  - generated answer
  - attribution
- 기존 간이 평가와 운영 체인 평가를 분리해서 보고

### 3.2 Retrieval 지표 확장

현재 retrieval consistency 그래프는 같은 category/chapter 문서가 top-k에 들어오는지 보여준다. 다음 지표를 추가하면 개선 실험 판단이 쉬워진다.

- Recall@k: 정답 source/chapter/doc_id가 top-k에 들어왔는지
- MRR: 정답 문서가 몇 번째에 나왔는지
- nDCG@k: 관련도 순위 품질
- Context Precision: 최종 context 중 실제로 답변에 필요한 문서 비율
- Abstention Accuracy: 근거 부족 질문에서 답변을 거절했는지

### 3.3 카테고리별 표본 확대

현재 hallucination category 그래프는 일부 카테고리 표본 수가 작다. 절차 카테고리의 25%는 실제 1건이므로, 개선 판단용으로는 표본을 늘려야 한다.

개선 방향:

- 카테고리별 최소 20개 이상 평가셋 구성
- 반납/불용/취득/처분/사용주기 예측처럼 혼동 가능성이 높은 카테고리 집중 보강
- paraphrase 질문, 짧은 구어체 질문, 복합 질문, 근거 부족 질문을 별도 세트로 구성

## 4. 테스트 코드 정비

파일: `tests/test_chain.py`

현재 테스트는 구현과 맞지 않는 패치 대상이 있다. 예를 들어 `rag.chain.query_refinement_chain`, `rag.chain.question_classifier_chain`을 패치하지만 실제 코드에는 해당 전역 객체가 없다. 또한 `run_rag_chain(user_query)`처럼 현재 함수 시그니처와 맞지 않는 호출도 있다.

개선 방향:

- `PromptTemplate` 패치 대신 classifier/refiner를 함수로 분리하고 그 함수를 테스트에서 패치
- `retrieve_docs`, `CrossEncoderReranker`, `llm.invoke`, `llm.bind_tools`를 명확히 mock
- 최소 테스트 세트:
  - 절차 질문은 RAG로 라우팅된다
  - 인사말은 NO_RAG로 라우팅된다
  - threshold 미달 시 NO_CONTEXT_RESPONSE 반환
  - reranker 사용 시 최종 context가 rerank 결과 순서를 따른다
  - tool 조회 질문은 RAG를 건너뛴다
  - tool + 절차 혼합 질문은 tool 결과와 RAG context를 함께 쓴다

## 5. 권장 실험 순서

### Phase 1. 빠른 성능 안정화

1. 질문 분류 프롬프트 수정
2. query refinement 프롬프트 수정
3. classification/refined query/retrieved score 로그 저장
4. threshold를 실제 score 분포 기반으로 재조정
5. 최종 context 수를 10에서 4~6 사이로 비교 실험

목표:

- hallucination rate 감소
- 절차/반납/불용 질문의 안정성 개선

### Phase 2. 검색 품질 개선

1. 원문 manual chunk를 vector DB에 함께 저장
2. QA 문서와 원문 문서에 `doc_type` metadata 부여
3. hybrid retrieval 또는 metadata boost 추가
4. reranker score 저장 및 threshold 적용
5. context diversity 적용

목표:

- Reference alignment 개선
- Chapter hit rate 개선
- 답변 근거 추적성 개선

### Phase 3. 평가/운영 품질 개선

1. 운영 체인 end-to-end 평가 스크립트 작성
2. 카테고리별 평가셋 확대
3. 실패 케이스 자동 리포트 생성
4. 테스트 코드 정비
5. 실험 설정별 결과를 CSV/Markdown으로 비교

목표:

- 개선 여부를 정량적으로 판단 가능
- 코드 수정 후 회귀 위험 감소

## 6. 가장 먼저 고칠 추천 항목

가장 먼저 손댈 항목은 다음 5개다.

1. `build_question_classifier_prompt()`를 도메인 업무 질문은 기본적으로 `NEED_RAG`가 되도록 수정
2. `build_query_refine_prompt()`의 `...`를 실제 검색어 재작성 규칙으로 교체
3. `SIMILARITY_SCORE_THRESHOLD = 10.0`을 score 분포 기반으로 재설정
4. rerank 최종 context 수를 10에서 4~6으로 줄이는 실험
5. 운영 체인의 retrieval 로그를 저장해서 실패 케이스를 직접 볼 수 있게 만들기

이 다섯 가지는 구현 난이도 대비 효과가 클 가능성이 높다. 특히 1번과 2번은 RAG 성능의 앞단을 잡는 작업이라, 모델이나 DB를 크게 바꾸기 전에 먼저 적용하는 것이 좋다.
