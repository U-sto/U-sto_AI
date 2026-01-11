import os
import json
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 환경 설정 로드
load_dotenv()

# [경로 설정]
# 실제 JSON 파일이 있는 경로를 정확히 적어야 합니다.
DATA_FILE_PATH = "dataset/qa_output/manual_qa_final.json"
# 챗봇이 읽을 DB가 저장될 경로 (app/config.py의 경로와 같아야 함)
DB_PATH = "./chroma_db" 

def ingest_data():
    print("🚀 데이터베이스 구축(Ingestion)을 시작합니다...")

    # 1. 파일 존재 확인
    if not os.path.exists(DATA_FILE_PATH):
        print(f"❌ 오류: 데이터 파일이 없습니다! ({DATA_FILE_PATH})")
        return

    # 2. JSON 로드
    with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"📄 JSON 파일 로드 완료: 총 {len(qa_data)}개의 지식 데이터")

    # 3. Document 객체로 변환 (검색 최적화)
    documents = []
    for item in qa_data:
        # 질문과 답변을 합쳐서 검색 텍스트를 만듭니다.
        # 이렇게 해야 "질문"과 유사해도 찾고, "답변" 내용으로도 찾습니다.
        combined_text = f"""
        [분류] {item.get('category', '일반')}
        [제목] {item.get('title', '제목없음')}
        [질문] {item.get('question')}
        [답변] {item.get('answer')}
        """
        
        # 메타데이터에는 출처와 원본 질문/답변을 따로 저장해둡니다.
        metadata = {
            "source": item.get("source", "manual"),
            "original_question": item.get("question"),
            "original_answer": item.get("answer")
        }
        
        documents.append(Document(page_content=combined_text.strip(), metadata=metadata))

    # 4. 기존 DB 삭제 (깨끗하게 새로 만들기 위해)
    if os.path.exists(DB_PATH):
        print("🗑️  기존 DB 삭제 중...")
        shutil.rmtree(DB_PATH)

    # 5. 임베딩 및 DB 저장
    print("💾 벡터 DB 굽는 중... (잠시만 기다려주세요)")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 비용 저렴, 성능 우수
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("✅ DB 구축 완료! 모든 데이터가 저장되었습니다.")
    print(f"👉 저장 경로: {DB_PATH}")

if __name__ == "__main__":
    ingest_data()