import traceback
from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수

from rag.prompt import build_prompt  # 프롬프트 생성
from rag.reranker import CrossEncoderReranker
from app.config import (
    NO_CONTEXT_RESPONSE, TECHNICAL_ERROR_RESPONSE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, RETRIEVER_TOP_K,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_K,
    RERANK_TOP_N,
    USE_RERANKING,
    RERANK_DEBUG
)

import logging
logger = logging.getLogger(__name__)

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate  # [NEW] 프롬프트 템플릿 추가
from langchain_core.output_parsers import StrOutputParser # [NEW] 문자열 파싱 추가

from vectorstore.retriever import retrieve_docs
from rag.prompt import build_prompt
from rag.reranker import CrossEncoderReranker
from app.config import (
    NO_CONTEXT_RESPONSE, TECHNICAL_ERROR_RESPONSE, SIMILARITY_SCORE_THRESHOLD, TOP_N_CONTEXT, RETRIEVER_TOP_K,
    RERANKER_MODEL_NAME,
    RERANK_CANDIDATE_K,
    RERANK_TOP_N,
    USE_RERANKING,
    RERANK_DEBUG
)

# RAG 시스템 페르소나 정의
RAG_SYSTEM_PROMPT = """
당신은 대학교 행정 업무를 지원하는 전문적인 AI 어시스턴트입니다.
제공된 문맥(Context)을 바탕으로 사용자의 질문에 답변하세요.

[답변 가이드라인]
1. 답변 끝에 불필요한 이모지나 사족을 달지 마세요.
2. 반드시 격식 있고 정중한 존댓말(하십시오체)을 사용하세요.
3. 모르는 내용은 솔직히 모른다고 답변하세요.
"""

# [NEW] 질문 정제(Refinement)용 프롬프트 정의
QUERY_REFINE_PROMPT = """
당신은 대학 행정 시스템 검색 전문가입니다.
사용자의 질문을 시스템 매뉴얼(DB)에서 검색하기 가장 적합한 '공식 행정 용어'와 '키워드 중심'의 문장으로 변환하세요.
답변에는 변환된 질문만 딱 한 문장으로 출력하세요. (설명 금지)

[변환 예시]
- 사용자: "장비 갖다 버리는 법" -> "물품 불용 신청 및 처분 절차"
- 사용자: "이거 등록하고 저장하면 끝이야?" -> "물품 취득 등록 후 승인 요청 절차"
- 사용자: "AI 예측 믿을만 해?" -> "사용주기 AI 예측 정확도 및 신뢰성"

사용자 질문: {question}
변환된 질문:"""

def run_rag_chain(
    llm,
    vectordb,
    user_query: str,
    retriever_top_k: int = RETRIEVER_TOP_K
):
    try:
        # ------------------------------------------------------------------
        # [STEP 0] 질문 정제 (Query Refinement) - 처방전 2 적용
        # ------------------------------------------------------------------
        # 사용자의 질문이 모호할 경우를 대비해, LLM을 통해 검색용 쿼리로 변환합니다.
        
        refine_prompt = PromptTemplate.from_template(QUERY_REFINE_PROMPT)
        refine_chain = refine_prompt | llm | StrOutputParser()
        
        # LLM에게 검색어 변환 요청
        refined_query = refine_chain.invoke({"question": user_query})
        
        # 로그 확인용 - logging 모듈 사용
        logger.info(f"[Query Refinement] 원본: '{user_query}' -> 변환: '{refined_query}'")

        # ------------------------------------------------------------------
        # [STEP 1] Retrieval (검색)
        # ------------------------------------------------------------------
        # 중요: 검색은 'refined_query(변환된 질문)'로 수행합니다!
        
        retrieved_docs = retrieve_docs(
            vectordb=vectordb,
            query=refined_query,   # [CHANGED] user_query -> refined_query
            top_k=retriever_top_k
        )

        # 검색 실패 시 fallback
        if not retrieved_docs:
            return {
                "answer": NO_CONTEXT_RESPONSE,
                "attribution": []
            }

        # ------------------------------------------------------------------
        # [STEP 2] 유사도 점수 필터링
        # ------------------------------------------------------------------
        filtered_docs = [
            (doc, retrieved_score)
            for doc, retrieved_score in retrieved_docs
            if retrieved_score <= SIMILARITY_SCORE_THRESHOLD
        ]

        if not filtered_docs:
            return {
                "answer": NO_CONTEXT_RESPONSE,
                "attribution": []
            }

        # reranking 직전 디버깅
        if USE_RERANKING and RERANK_DEBUG:
            print("[DEBUG] Retrieval 결과 (정렬 전):")
            for i, (doc, score) in enumerate(filtered_docs[:5]):
                print(f"  [{i}] doc_id={doc.metadata.get('doc_id')} | score={score:.4f}")
        
        # 정렬 (Chroma는 score가 낮을수록 유사함 -> 오름차순 정렬)
        filtered_docs.sort(key=lambda x: x[1])  

        # Re-ranking 대상 후보 수 제한
        rerank_candidates = filtered_docs
        if USE_RERANKING:
            rerank_candidates = filtered_docs[:RERANK_CANDIDATE_K]
        
        # ------------------------------------------------------------------
        # [STEP 3] Re-ranking (선택 사항)
        # ------------------------------------------------------------------
        if USE_RERANKING:
            if RERANK_DEBUG:
                print("[DEBUG] Re-ranking 적용")

            reranker = CrossEncoderReranker(RERANKER_MODEL_NAME)
            
            # 중요: Re-ranking도 '변환된 질문(refined_query)'과 문서를 비교해야 정확합니다.
            top_docs = reranker.rerank(
                query=refined_query,  # [CHANGED] user_query -> refined_query
                docs_with_scores=rerank_candidates,
                top_n=RERANK_TOP_N
            )
        else:
            # Reranking 안 쓰면 상위 N개만 선택
            top_docs = [doc for doc, _ in filtered_docs[:TOP_N_CONTEXT]]

        # ------------------------------------------------------------------
        # [STEP 4] Context 구성
        # ------------------------------------------------------------------
        context = "\n\n".join([
            doc.page_content for doc in top_docs 
        ])

        # ------------------------------------------------------------------
        # [STEP 5] Chunk Attribution 구성
        # ------------------------------------------------------------------
        attribution = [
            {"doc_id": doc.metadata.get("doc_id")}
            for doc in top_docs
        ]

        # ------------------------------------------------------------------
        # [STEP 6] 프롬프트 생성
        # ------------------------------------------------------------------
        # 중요: 최종 답변 생성은 '원래 사용자의 질문(user_query)'에 대해 답해야 합니다.
        # 문맥(Context)은 변환된 질문으로 잘 찾아왔으니, 이제 사용자에게 친절하게 답할 차례입니다.
        
        prompt = build_prompt(
            context=context,
            question=user_query  # [KEEP] 여기는 원본 질문 유지!
        )
        
        # ------------------------------------------------------------------
        # [STEP 7] LLM 답변 생성
        # ------------------------------------------------------------------
        response = llm.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
        )

        return {
            "answer": response.content,
            "attribution": attribution
        }

    except Exception as e:
        print(f"[ERROR] LLM Chain failed: {e}")
        print(f"[ERROR] Failed query: {user_query}")
        print(traceback.format_exc())

        return {
            "answer": TECHNICAL_ERROR_RESPONSE,
            "attribution": []
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