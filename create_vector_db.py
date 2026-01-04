import os
import json
import glob
import shutil
import sys
import io
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ==========================================
# 🔇 [화면 출력 인코딩 설정] (Windows 한글 깨짐 방지)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# ==========================================

# 🚨 [필수] API 키 입력 -> 키 가져오기로 바꿨습니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# 경로 설정
INPUT_FOLDER = 'dataset/qa_output'
DB_PATH = 'chroma_db'  # 벡터 DB가 저장될 폴더명

def create_vector_db():
    print("🚀 벡터 DB 생성 작업을 시작합니다...")

    # 1. 기존 DB 삭제 (중복 방지)
    if os.path.exists(DB_PATH):
        print(f"🔄 기존 DB 폴더('{DB_PATH}')를 초기화합니다...")
        shutil.rmtree(DB_PATH)
    
    # 2. QA 데이터 불러오기
    json_files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    if not json_files:
        print("❌ 저장된 Q/A 데이터 파일이 없습니다. (2단계 먼저 실행해주세요)")
        return

    documents = []
    print(f"📂 총 {len(json_files)}개의 파일에서 데이터를 로드합니다.")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for item in data:
                question = item.get("question", "")
                answer = item.get("answer", "")
                source = item.get("source", "unknown")

                if question and answer:
                    # 💡 [핵심] 검색할 텍스트 내용을 정의합니다.
                    # 질문과 답변을 합쳐서 임베딩해야 검색 정확도가 높아집니다.
                    page_content = f"질문: {question}\n답변: {answer}"
                    
                    # 메타데이터: 나중에 출처를 밝히거나 원본 답변을 보여줄 때 사용
                    metadata = {
                        "source": source,
                        "question": question,  # 원본 질문 따로 저장
                        "answer": answer       # 원본 답변 따로 저장
                    }
                    
                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)
                    
        except Exception as e:
            print(f"⚠️ 파일 읽기 오류 ({file_path}): {e}")

    print(f"✅ 총 {len(documents)}개의 데이터 조각(Document)을 준비했습니다.")

    # 3. 임베딩 및 DB 저장
    if documents:
        print("🧠 데이터를 벡터로 변환하고 저장하는 중... (시간이 조금 걸릴 수 있습니다)")
        
        # OpenAI의 최신 임베딩 모델 사용 (가격 저렴, 성능 우수)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # ChromaDB 생성 및 저장
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        # 강제 저장 (버전에 따라 자동 저장되지만 확실히 하기 위해)
        try:
            vectordb.persist() 
        except:
            pass # 최신 버전에서는 자동 저장됨

        print(f"🎉 모든 작업 완료! '{DB_PATH}' 폴더에 데이터베이스가 생성되었습니다.")
    else:
        print("❌ 변환할 데이터가 없습니다.")

if __name__ == "__main__":
    create_vector_db()