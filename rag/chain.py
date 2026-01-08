from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수
from rag.prompt import build_prompt  # 프롬프트 생성
from app.config import (
    NO_CONTEXT_RESPONSE,
    CROSS_ENCODER_MODEL_NAME,
    RERANK_TOP_N
)
from vectorstore.cross_encoder import CrossEncoderReranker
# Chunk Attribution 포함 RAG Chain
# chain.py

def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
    retriever_top_k: int = 60
):

    # 1️. Retrieval
    # Chroma VectorStore 사용
    # 내부적으로 FAISS 인덱스가 검색 수행
    retrieved_docs = retrieve_docs(
        vectordb=vectordb,
        query=user_query,
        top_k=retriever_top_k
    )

    # 검색 실패 시 fallback
    if not retrieved_docs:
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "attribution": []
        }

    # 2️. Cross-Encoder Re-ranking 초기화
    reranker = CrossEncoderReranker(CROSS_ENCODER_MODEL_NAME)

    # Re-ranking 수행
    reranked_docs = reranker.rerank(
        query=user_query,
        docs_with_scores=retrieved_docs,
        top_n=RERANK_TOP_N
    )

    # 3️. Context 구성
    context = "\n\n".join([
        doc.page_content  # 문서 본문
        for doc, _ in reranked_docs
    ])

    # 4. Chunk Attribution 구성
    attribution = [
        {
            "doc_id": doc.metadata.get("doc_id"),
            "score": score
        }
        for doc, score in reranked_docs
    ]

    # 5. 프롬프트 생성
    prompt = build_prompt(context, user_query)

    # 6. LLM 호출 (top-p + top-k 상한 조합)
    response = llm.invoke(
        [
            SystemMessage(content=
                "당신은 대학교 행정 업무를 지원하는 전문 AI 어시스턴트입니다."
                "반드시 근거가 있는 내용만 답변하십시오."
                "불확실한 경우 모른다고 명시하십시오."
                "존댓말(하십시오체)만 사용하십시오."
            ),
            HumanMessage(content=prompt)
        ],
        top_p=0.9,        # 누적 확률 기반 샘플링
        temperature=0.3  # 안정성 중심
    )

    # 9️⃣ 최종 응답 반환
    return {
        "answer": response.content,
        "attribution": attribution
    }


# 1. Retrieval → Re-ranking → LLM 연결 로직
# 2. Chunk Attribution 기능 구현
