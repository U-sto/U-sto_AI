from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수

from rag.prompt import build_prompt  # 프롬프트 생성
from app.config import (
    NO_CONTEXT_RESPONSE,LLM_TEMPERATURE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, Top_p, RETRIEVER_TOP_K
)
# ==============================
# FAISS score 기반 RAG Chain
# ==============================


def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
     retriever_top_k: int = RETRIEVER_TOP_K
):

    # 1️. Retrieval
    # Chroma VectorStore 사용
    # 내부적으로 FAISS 인덱스가 검색 수행
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
        (doc, score)
        for doc, score in retrieved_docs
        if score <= SIMILARITY_SCORE_THRESHOLD
    ]

    # threshold 통과 문서가 없는 경우 fallback
    if not filtered_docs:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "attribution": []
        }

    # 3️. FAISS score 기준 정렬
    # distance 기준이므로 오름차순 정렬
    filtered_docs.sort(key=lambda x: x[1])

    # 상위 N개 문서 선택
    top_docs = filtered_docs[:TOP_N_CONTEXT]

    # 4. Context 구성
    context = "\n\n".join([
        doc.page_content  # 문서 본문
        for doc, _ in top_docs
    ])

    # 5. Chunk Attribution 구성
    attribution = [
        {
            "doc_id": doc.metadata.get("doc_id"),
            "score": float(score)  # numpy 타입 제거
        }
        for doc, score in top_docs
    ]

    # 6. 프롬프트 생성
    prompt = build_prompt(
        context=context,
        question=user_query
    )

    # 7. LLM 호출 (top-p + top-k 상한 조합)
    response = llm.invoke(
        [
            SystemMessage(content=
            (
                "당신은 대학교 행정 업무를 지원하는 전문 AI 어시스턴트입니다." 
                "반드시 근거가 있는 내용만 답변하십시오." 
                "불확실한 경우 모른다고 명시하십시오." 
                "존댓말(하십시오체)만 사용하십시오." )
            ),
            HumanMessage(content=prompt)
        ],
        # top_p=Top_p,  # 다양성 조절
        # temperature=LLM_TEMPERATURE  # 응답 다양성 조절
    )

    # 8. 최종 응답 반환
    return {
        "answer": response.content,
        "attribution": attribution
    }


"""
RAG Chain 구성
- Retrieval: Chroma VectorStore (내부 FAISS 검색)
- Filtering: 유사도 점수 threshold 기반 문서 선별
- Sorting: FAISS score 오름차순 정렬
- Generation: LLM 응답 생성
- Attribution: 사용된 문서 메타데이터 반환
"""
