# scripts_/create_vector_db.py

import os
import sys
import io
import json
import shutil
import time
import gc
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ==========================================
# [화면 출력 인코딩 설정]
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
# ==========================================

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(root_dir)
for path in (root_dir, project_root):
    if path not in sys.path:
        sys.path.append(path)

import app.config as config
from rag.dictionaries import normalize_category, normalize_doc_id_part


def stringify_value(value, indent=0):
    prefix = " " * indent
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        lines = []
        for key, sub_value in value.items():
            rendered = stringify_value(sub_value, indent + 2)
            if rendered:
                lines.append(f"{prefix}- {key}: {rendered.strip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            rendered = stringify_value(item, indent + 2)
            if rendered:
                lines.append(f"{prefix}- {rendered.strip()}")
        return "\n".join(lines)
    return str(value)


def derive_section_path(item):
    section_path = item.get("section_path")
    if isinstance(section_path, list):
        return " > ".join(str(part).strip() for part in section_path if str(part).strip())
    if section_path:
        return str(section_path).strip()

    chapter = str(item.get("chapter", "")).strip()
    title = str(item.get("title", "")).strip()
    return " > ".join(part for part in (chapter, title) if part) or "root"


def build_manual_content(item):
    priority_fields = [
        "summary",
        "process_summary",
        "location",
        "trigger_condition",
        "process_steps",
        "buttons_and_functions",
        "input_fields",
        "search_conditions",
        "list_display_items",
        "terms_dictionary",
        "date_comparison",
        "warnings",
        "notes",
        "content",
    ]
    lines = [
        f"문서 유형: 원문 매뉴얼",
        f"장/절: {item.get('chapter', '')}",
        f"제목: {item.get('title', '')}",
        f"분류: {item.get('category', '')}",
        f"섹션 경로: {derive_section_path(item)}",
        f"키워드: {', '.join(item.get('keywords', [])) if isinstance(item.get('keywords'), list) else item.get('keywords', '')}",
    ]
    for field in priority_fields:
        rendered = stringify_value(item.get(field))
        if rendered:
            lines.append(f"{field}:\n{rendered}")
    return "\n\n".join(line for line in lines if str(line).strip())


def build_qa_documents(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"QA 파일 형식이 잘못되었습니다. 기대 형식: list, 실제: {type(data)}")

    documents = []
    for idx, item in enumerate(data):
        q = item.get("question", "")
        a = item.get("answer", "")
        if not q or not a:
            continue

        category = normalize_category(
            item.get("category"),
            item.get("chapter", ""),
            item.get("title", ""),
            q,
            a,
        )
        content = (
            "문서 유형: QA 매칭 문서\n"
            "역할: 사용자 질문 의도 매칭용\n"
            f"문서 주제: {category}\n"
            f"관련 메뉴: {item.get('title', '')}\n"
            f"사용자 질문: {q}\n"
            f"답변 요약: {a}"
        )
        metadata = {
            "doc_type": "qa",
            "source": item.get("source", "manual_qa_final.json"),
            "title": item.get("title", ""),
            "chapter": item.get("chapter", ""),
            "category": category,
            "section_path": item.get("section_path") or derive_section_path(item),
            "question": q,
            "doc_id": item.get("doc_id") or f"qa_{idx:04d}",
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_manual_documents(input_folder):
    documents = []
    if not os.path.exists(input_folder):
        print(f"경고: 원문 매뉴얼 폴더가 없습니다 -> {input_folder}")
        return documents

    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"경고: 원문 매뉴얼 파일 형식이 list가 아니어서 건너뜁니다 -> {file_name}")
            continue

        for idx, item in enumerate(data):
            title = item.get("title", "")
            chapter = item.get("chapter", "")
            category = normalize_category(
                item.get("category"),
                chapter,
                title,
                item.get("summary", ""),
                item.get("keywords", ""),
            )
            item = {**item, "category": category}
            content = build_manual_content(item)
            if not content.strip():
                continue
            section_path = derive_section_path(item)
            doc_id = f"manual_{normalize_doc_id_part(file_name)}_{normalize_doc_id_part(chapter)}_{idx:03d}"
            metadata = {
                "doc_type": "manual_chunk",
                "source": file_name,
                "title": title,
                "chapter": chapter,
                "category": category,
                "section_path": section_path,
                "doc_id": doc_id,
                "chunk_index": idx,
            }
            documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_faq_documents(faq_file):
    documents = []
    if not os.path.exists(faq_file):
        print(f"경고: FAQ 파일이 없습니다 -> {faq_file}")
        return documents

    with open(faq_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"경고: FAQ 파일 형식이 list가 아니어서 건너뜁니다 -> {faq_file}")
        return documents

    for idx, item in enumerate(data):
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question or not answer:
            continue
        keywords = item.get("keywords", [])
        category = normalize_category(item.get("category") or "FAQ", "FAQ", question, answer, keywords)
        faq_source = item.get("source", "")
        content = (
            "문서 유형: FAQ\n"
            "source=faq\n"
            f"faq_source: {faq_source}\n"
            f"FAQ 분류: {category}\n"
            f"질문: {question}\n"
            f"답변: {answer}\n"
            f"키워드: {', '.join(keywords) if isinstance(keywords, list) else keywords}"
        )
        metadata = {
            "doc_type": "faq",
            "source": "faq",
            "faq_source": faq_source,
            "title": question,
            "chapter": "FAQ",
            "category": category,
            "section_path": "FAQ",
            "doc_id": item.get("id") or f"faq_{idx:04d}",
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def main():
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("오류: .env 파일이 없거나 OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    # [설정] 경로 지정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # [중요] 프로젝트 루트는 ai_rag의 상위 폴더입니다.
    target_file = os.path.join(project_root, "dataset", "qa_output", "manual_qa_final.json")
    manual_input_folder = os.path.join(project_root, "dataset", "input")
    faq_file = os.path.join(project_root, "dataset", "FAQ", "faq_data.json")
    
    # DB 저장 경로
    DB_PATH = os.path.join(project_root, config.VECTOR_DB_PATH)

    print("벡터 DB 생성 작업을 시작합니다...")

    temp_db_path = f"{DB_PATH}_building_{int(time.time())}"
    backup_db_path = f"{DB_PATH}_backup_{int(time.time())}"

    # 1. 데이터 로드
    if not os.path.exists(target_file):
        print(f"오류: 파일이 없습니다 -> {target_file}")
        print("   먼저 generate_qa.py를 실행해서 데이터를 만들어주세요.")
        sys.exit(1)

    try:
        qa_documents = build_qa_documents(target_file)
        manual_documents = build_manual_documents(manual_input_folder)
        faq_documents = build_faq_documents(faq_file)
    except Exception as e:
        print(f"문서 변환 실패: {e}")
        sys.exit(1)

    documents = qa_documents + manual_documents + faq_documents

    # 변환된 데이터가 하나도 없는 경우 처리
    if not documents:
        print("변환할 데이터가 없습니다.")
        sys.exit(1)

    print(f"QA 문서: {len(qa_documents)}개")
    print(f"원문 매뉴얼 문서: {len(manual_documents)}개")
    print(f"FAQ 문서: {len(faq_documents)}개")
    print(f"총 {len(documents)}개의 지식을 준비했습니다.")

    # 2. 임베딩 및 DB 저장
    print("Embedding 모델을 준비 중입니다...")
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)

    # 임베딩 진행 상황 구체적 명시
    print(f"Chroma DB 저장 시작... (총 {len(documents)}개 벡터 변환, 임시 저장 위치: {temp_db_path})")
    print("(데이터 양에 따라 시간이 조금 걸릴 수 있습니다...)")
    
    try:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=temp_db_path
        )
        try:
            vectordb.persist()
        except Exception:
            pass

        try:
            chroma_client = getattr(vectordb, "_client", None)
            chroma_system = getattr(chroma_client, "_system", None)
            if chroma_system is not None and hasattr(chroma_system, "stop"):
                chroma_system.stop()
        except Exception as e:
            print(f"경고: Chroma client 정리 중 오류가 발생했지만 DB 생성은 계속합니다: {e}")

        del vectordb
        gc.collect()
        time.sleep(2)

    except Exception as e:
        print(f"[Fatal Error] 임베딩 및 DB 저장 중 실패: {e}")
        if os.path.exists(temp_db_path):
            shutil.rmtree(temp_db_path, ignore_errors=True)
        sys.exit(1)

    # 3. 성공한 경우에만 기존 DB 교체
    try:
        if os.path.exists(DB_PATH):
            print(f"기존 DB를 백업합니다: {backup_db_path}")
            shutil.move(DB_PATH, backup_db_path)

        print(f"새 DB를 활성 경로로 이동합니다: {DB_PATH}")
        shutil.move(temp_db_path, DB_PATH)

        if os.path.exists(backup_db_path):
            shutil.rmtree(backup_db_path, ignore_errors=True)
    except Exception as e:
        print(f"[Fatal Error] 새 DB 교체 중 실패: {e}")
        if os.path.exists(temp_db_path):
            shutil.rmtree(temp_db_path, ignore_errors=True)
        if os.path.exists(backup_db_path) and not os.path.exists(DB_PATH):
            shutil.move(backup_db_path, DB_PATH)
        sys.exit(1)

    print("-" * 30)
    print("DB 생성 완료!")
    print(f"이제 '{DB_PATH}' 폴더에 AI의 지식이 저장되었습니다.")

if __name__ == "__main__":
    main()
