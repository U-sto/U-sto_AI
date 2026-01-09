from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수

from rag.prompt import build_prompt  # 프롬프트 생성
from app.config import (
    NO_CONTEXT_RESPONSE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, RETRIEVER_TOP_K
)

# RAG 시스템 페르소나 정의 (유지보수를 위해 상수로 분리)
RAG_SYSTEM_PROMPT = """
당신은 대학교 행정 업무를 지원하는 전문적인 AI 어시스턴트입니다.
제공된 문맥(Context)을 바탕으로 사용자의 질문에 답변하세요.

[답변 가이드라인]
1. 답변 끝에 불필요한 이모지나 사족을 달지 마세요.
2. 반드시 격식 있고 정중한 존댓말(하십시오체)을 사용하세요.
3. 모르는 내용은 솔직히 모른다고 답변하세요.
"""

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
    # 3️. 유사도 기반 score 기준 정렬
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
    
    # 7. LLM 호출 (sampling 파라미터는 LLM 생성 시 설정됨)
    # LLM 호출
    try:
        response = llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),  # 시스템 프롬프트
                HumanMessage(content=prompt)               # 유저 프롬프트
            ]
        )

    except Exception:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "attribution": []
        }

    # 9. 최종 반환
    return {
        "answer": response.content,   # LLM 답변
        "attribution": attribution    # 사용 문서
    }


"""
RAG Chain 구성
- Retrieval: Chroma VectorStore (HNSW 기반 벡터 인덱싱)
- Filtering: 유사도 점수 threshold 기반 문서 선별
- Sorting: 유사도 점수 score 오름차순 정렬
- Generation: LLM 응답 생성
- Attribution: 사용된 문서 메타데이터 반환
"""
