import os
import sys
import io
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
# =====
# 역할
# - 유저 입력
# - retriever 호출
# - LLM 응답 출력
# ==========================================
# 🔇 [화면 출력 인코딩 설정]
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# ==========================================

# # 🚨 [필수] API 키 입력 (본인 키로 변경!) -> 키 가져오기로 바꿨습니다.
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
load_dotenv()  # .env 파일을 os.environ에 로드 (정석 방법)
# 벡터 DB 경로 (방금 만든 폴더 이름과 일치해야 함)
DB_PATH = 'chroma_db'

def run_chat():
    print("🤖 챗봇 시스템 로딩 중... (DB 연결)")

    # 1. DB 로드 (LangChain 방식)
    if not os.path.exists(DB_PATH):
        print(f"❌ 오류: '{DB_PATH}' 폴더가 없습니다. create_vector_db.py를 먼저 실행하세요.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        # LangChain으로 저장한 DB 불러오기
        vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        print("✅ 지식 데이터베이스 연결 성공!")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return

    # 2. AI 모델 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

    print("="*50)
    print("🎓 대학 물품 관리 AI 챗봇이 준비되었습니다!")
    print("('종료'라고 입력하면 꺼집니다)")
    print("="*50)

    # 3. 채팅 루프
    while True:
        user_input = input("\n🙋 질문하세요: ")
        
        if user_input.strip() == "종료":
            print("시스템을 종료합니다. 안녕히 가세요!")
            break
        
        if not user_input.strip():
            continue

        print("Thinking...", end="", flush=True)

        # === 검색 단계 ===
        # 질문과 관련된 문서 3개 찾기
        docs = vectordb.similarity_search(user_input, k=3)
        
        if not docs:
            print("\r⚠️ 관련 정보를 매뉴얼에서 찾을 수 없습니다.")
            continue

        # 검색된 내용을 합치기
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 출처 확인 (메타데이터 활용)
        sources = set([doc.metadata.get('source', '알 수 없음') for doc in docs])

        # === 디버깅 출력 (사용자가 좋아했던 기능) ===
        print(f"\r🔍 [참고 자료] {', '.join(sources)} 에서 정보를 찾았습니다.")

        # === 프롬프트 작성 ===
        system_instruction = f"""
        너는 대학 물품 관리 시스템의 'AI 챗봇'이야.
        아래 [참고 자료]를 바탕으로 사용자의 질문에 친절하고 정확하게 답변해줘.
        
        [규칙]
        1. 반드시 아래 제공된 [참고 자료] 내용에 기반해서 답변해.
        2. [참고 자료]에 없는 내용은 "죄송합니다, 매뉴얼에 해당 내용이 없어 답변드리기 어렵습니다."라고 말해.
        3. 답변은 한국어로 하고, 이해하기 쉽게 요약해서 말해줘.
        
        [참고 자료]
        {context_text}
        """

        # === AI 답변 생성 ===
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=system_instruction + f"\n\n질문: {user_input}")
        ])

        print(f"🤖 AI 답변: {response.content}\n")
        print("-" * 50)

if __name__ == "__main__":
    run_chat()