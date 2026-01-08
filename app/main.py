import sys
import io
import os
from dotenv import load_dotenv                      # 환경 변수 로드
from langchain_openai import ChatOpenAI             # LLM
from ingestion.embedder import get_embedding_model  # 임베딩
from vectorstore.chroma_store import load_chroma_db # DB 로드
from rag.chain import run_rag_chain                 # RAG 체인
from app.config import VECTOR_DB_PATH, LLM_MODEL_NAME, LLM_TEMPERATURE

# ==========================================
# 🔇 Windows 한글 깨짐 방지용 출력 인코딩 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")
# ==========================================

def main():
    load_dotenv()  # .env 로드

    print("🤖 챗봇 시스템 로딩 중... (DB 연결)")

    # DB 존재 여부 확인
    if not os.path.exists(VECTOR_DB_PATH):
        print(f"❌ 오류: '{VECTOR_DB_PATH}' 폴더가 없습니다.")
        print("👉 create_vector_db.py를 먼저 실행하세요.")
        return

    # 임베딩 모델 로드
    embeddings = get_embedding_model()


    # 벡터 DB 로드
    try:
        vectordb = load_chroma_db(embeddings, VECTOR_DB_PATH)
        print("✅ 지식 데이터베이스 연결 성공!")
    except Exception as e:
        print("❌ DB 연결 실패")
        return

    llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE
)
    print("=" * 50)
    print("🎓 대학 물품 관리 AI 챗봇이 준비되었습니다!")
    print("👉 질문을 입력하세요. ('종료' 입력 시 종료)")
    print("=" * 50)

    # 채팅 루프
    while True:
        user_input = input("\n🙋 질문하세요: ").strip()

        if user_input == "종료":
            print("👋 시스템을 종료합니다. 안녕히 가세요!")
            break

        if not user_input:
            continue

        print("🤔 Thinking...", end="", flush=True)

        # # RAG 실행 (기존 코드)
        # answer = run_rag_chain(llm, vectordb, user_input)

        # RAG 실행 (정확한 인자 순서)
        answer = run_rag_chain(
            llm=llm,
            vectordb=vectordb,
            user_query=user_input
        )


        # 출력 정리
        print("\r🤖 AI 답변:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
