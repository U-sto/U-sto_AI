import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tqdm import tqdm  # [추가] 진행바 라이브러리

# 1. 환경 변수 로드 (API 키 등)
load_dotenv()

# 2. 프로젝트 루트 경로 추가 (ingestion 폴더를 찾기 위해 필수!)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 3. 도구들 가져오기 (Import)
try:
    from ingestion.loader import load_json_files
    from ingestion.qa_converter import convert_to_qa
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    print("팁: 터미널 위치가 프로젝트 최상위(AI_Project)인지 확인하세요.")
    sys.exit(1)

def main():
    # [설정] 경로 지정
    INPUT_FOLDER = os.path.join(root_dir, 'dataset/input') 
    
    # 결과 저장 폴더
    OUTPUT_FOLDER = os.path.join(root_dir, 'dataset/qa_output')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("QA 데이터 생성을 시작합니다...")

    # 1. 데이터 로드
    if not os.path.exists(INPUT_FOLDER):
        print(f"원본 데이터 폴더가 존재하지 않습니다: {INPUT_FOLDER}")
        return

    documents = load_json_files(INPUT_FOLDER)
    print(f"원본 문서 {len(documents)}개를 로드했습니다.")

    # 2. LLM 설정 (여기서 직접 os.getenv로 키를 가져옵니다)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("오류: .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        return

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)

    generated_data = []

    # 3. 변환 작업
    print(" AI가 문서를 읽고 질문을 생성 중입니다...")

    for doc in tqdm(documents, desc="QA 변환 진행"):
        # [수정] 변수명 변경 (qa_pair -> qa_data)
        qa_data = convert_to_qa(doc, llm)
        
        # 기본적인 데이터 검증 (Validation)
        # qa_data가 없거나, 필수 필드(질문, 답변)가 비어있으면 스킵
        if not qa_data:
            # 변환 실패 (이미 converter 내부에서 처리됨)
            continue
            
        if not qa_data.get("question") or not qa_data.get("answer"):
            # tqdm을 쓸 때는 print 대신 tqdm.write를 써야 진행바가 안 깨집니다.
            tqdm.write(f"[Skip] 불완전한 데이터 형식 제외됨: {doc.get('title', 'Untitled')}")
            continue

        generated_data.append(qa_data)

    # 4. 저장
    output_file = os.path.join(OUTPUT_FOLDER, 'manual_qa_final.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"작업 완료! 총 {len(generated_data)}개의 QA 쌍이 생성되었습니다.")
    print(f"저장 위치: {output_file}")

if __name__ == "__main__":
    main()