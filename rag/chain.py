from langchain_core.messages import SystemMessage, HumanMessage  # 메시지 타입
from vectorstore.retriever import retrieve_docs  # 검색 함수
from rag.prompt import build_prompt  # 프롬프트 생성
from app.config import NO_CONTEXT_RESPONSE

# RAG 시스템 페르소나 정의 (유지보수를 위해 상수로 분리)
RAG_SYSTEM_PROMPT = """
당신은 대학교 행정 업무를 지원하는 전문적인 AI 어시스턴트입니다.
제공된 문맥(Context)을 바탕으로 사용자의 질문에 답변하세요.

[답변 가이드라인]
1. 답변 끝에 불필요한 이모지나 사족을 달지 마세요.
2. 반드시 격식 있고 정중한 존댓말(하십시오체)을 사용하세요.
3. 모르는 내용은 솔직히 모른다고 답변하세요.
"""

def run_rag_chain(llm, vectordb, user_query: str):
    # 문서 검색
    docs = retrieve_docs(vectordb, user_query)

    if not isinstance(docs, list) or not docs:
        print(f"[Debug] 검색된 문서가 없거나 타입이 잘못되었습니다. (Query: {user_query})")
        return NO_CONTEXT_RESPONSE

    # 컨텍스트 결합
    context = "\n\n".join([doc.page_content for doc in docs])

    # 프롬프트 생성
    prompt_content = build_prompt(context, user_query)

    # LLM 호출
    try:
        response = llm.invoke([
            SystemMessage(content=RAG_SYSTEM_PROMPT), # 상수로 정의된 시스템 프롬프트 사용
            HumanMessage(content=prompt_content)
        ])
        return response.content

    except Exception as e:
        print(f"[Error] LLM 답변 생성 중 오류 발생: {e}")
        # 에러 발생 시 사용자에게 보여줄 기본 메시지 반환
        return "죄송합니다. 답변을 생성하는 도중 내부 오류가 발생했습니다."