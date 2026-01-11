import traceback
from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수

from rag.prompt import assemble_prompt  # 프롬프트 생성
from rag.reranker import CrossEncoderReranker
from app.config import (
    NO_CONTEXT_RESPONSE, TECHNICAL_ERROR_RESPONSE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, RETRIEVER_TOP_K,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_K,
    RERANK_TOP_N,
    USE_RERANKING,
    RERANK_DEBUG
)

def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
    retriever_top_k: int = RETRIEVER_TOP_K
):

    # 1️. Retrieval
    # Chroma VectorStore 사용
    retrieved_docs = retrieve_docs(
        vectordb=vectordb,     # 벡터 DB
        query=user_query,      # 사용자 질문
        top_k=retriever_top_k  # 후보 문서 수
    )

    # 검색 실패 시 fallback
    if not retrieved_docs:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "attribution": []
        }

    # 2️. 유사도 점수 score 기반 필터링
    # threshold 이하 문서만 유지
    filtered_docs = [
        (doc, retrieved_score)
        for doc, retrieved_score in retrieved_docs
        if retrieved_score <= SIMILARITY_SCORE_THRESHOLD
    ]

     # threshold 통과 문서가 없는 경우 fallback
    if not filtered_docs:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "attribution": []
        }

    # reranking 직전 (retrieval 결과 확인)
    if USE_RERANKING:
        if RERANK_DEBUG:
            print("[DEBUG] Retrieval 결과 (정렬 전):")
            for i, (doc, score) in enumerate(filtered_docs[:5]):
                print(f"  [{i}] doc_id={doc.metadata.get('doc_id')} | score={score}")
    
    # 기본값 None으로 명시
    rerank_candidates = None

    # 정렬 먼저
    filtered_docs.sort(key=lambda x: x[1])  

    # Re-ranking 대상 후보 수 제한
    if USE_RERANKING:
        rerank_candidates = filtered_docs[:RERANK_CANDIDATE_K]
    
    # 3️. 유사도 기반 score 기준 정렬
    # distance 기준이므로 오름차순 정렬
    filtered_docs.sort(key=lambda x: x[1])

    if USE_RERANKING:
        if RERANK_DEBUG:
            print("[DEBUG] Re-ranking 적용")

        # 3.5 Re-ranking
        reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)
        top_docs = reranker.rerank(
            query=user_query,
            docs_with_scores=rerank_candidates,
            top_n=RERANK_TOP_N
        )
    else:
        # score 버리고 Document만 유지
        top_docs = [doc for doc, _ in filtered_docs[:TOP_N_CONTEXT]]

    # reranking 이후 (최종 선택 결과 확인)

    if USE_RERANKING and RERANK_DEBUG:
        print("[DEBUG] Re-ranking 후 doc_id:")
        for i, doc in enumerate(top_docs):
            print(f"  [{i}] {doc.metadata.get('doc_id')}")


    # 4. Context 구성
    # re-ranking 이후에는 Document 리스트만 사용
    context = "\n\n".join([
        doc.page_content for doc in top_docs 
    ])

    # 5. Chunk Attribution 구성
    attribution = [
    {
        "doc_id": doc.metadata.get("doc_id")
    }
    for doc in top_docs
]


    # 6. 프롬프트 생성
    prompt = assemble_prompt(
        context=context,
        question=user_query
    )
    
    # 7. LLM 호출 (sampling 파라미터는 LLM 생성 시 설정됨)
    # LLM 호출
    try:
        response = llm.invoke(
            [
                HumanMessage(content=prompt)               # 모든 규칙이 prompt에 포함됨
            ]
        )

    except Exception as e:
        # 1. 에러가 났다는 사실과 내용을 출력 (터미널에서 확인 가능)
        print(f"[ERROR] LLM Chain failed: {e}")
    
        # 문제를 일으킨 질문 내용 로깅 (디버깅용)
        print(f"[ERROR] Failed query: {user_query}")

        # 2. 상세 에러 위치(몇 번째 줄인지) 출력
        print(traceback.format_exc())

        # 3. 프로그램이 죽지 않도록 안전한 기본값(Fallback) 반환
        return {
            "answer": TECHNICAL_ERROR_RESPONSE,
            "attribution": []
        }

    # 9. 최종 반환
    return {
        "answer": response.content,   # LLM 답변
        "attribution": attribution    # 사용 문서
    }


"""
RAG Chain 구성
- Retrieval: Chroma(HNSW) 기반 후보 문서 검색
- Filtering: similarity score threshold 기반 후보 필터링
- Re-ranking: CrossEncoder 기반 의미론적 재채점 및 문서 재정렬
- Context Selection: 상위 chunk 선별
- Generation: context 기반 LLM 응답 생성
- Chunk Attribution: 사용 chunk metadata 추적
"""
