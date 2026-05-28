# scripts_/create_vector_db.py

import os
import sys
import io
import json
import shutil
import re
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

def infer_category(*texts):
    joined = " ".join(str(text or "") for text in texts)
    if any(keyword in joined for keyword in ["반납", "반환"]):
        return "물품 반납 관리"
    if any(keyword in joined for keyword in ["불용", "사용 중단"]):
        return "물품 불용 관리"
    if any(keyword in joined for keyword in ["처분", "폐기", "매각", "멸실", "도난"]):
        return "물품 처분 관리"
    if any(keyword in joined for keyword in ["취득", "등록", "검수", "G2B", "목록번호"]):
        return "물품 취득 관리"
    if any(keyword in joined for keyword in ["보유", "현황", "조회"]):
        return "물품 보유 현황"
    if any(keyword in joined for keyword in ["AI", "예측", "사용주기", "내용연수"]):
        return "사용주기 AI 예측"
    return "일반"


def normalize_doc_id(value):
    value = str(value or "").strip()
    value = re.sub(r"[^0-9a-zA-Z가-힣_.-]+", "_", value)
    return value.strip("_") or "doc"


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

        category = item.get("category") or infer_category(q, a, item.get("title", ""))
        content = (
            "문서 유형: QA 매칭 문서\n"
            f"문서 주제: {category}\n"
            f"관련 메뉴: {item.get('title', '')}\n"
            f"사용자 질문: {q}\n"
            f"상세 답변: {a}"
        )
        metadata = {
            "doc_type": "qa",
            "source": item.get("source", "manual_qa_final.json"),
            "title": item.get("title", ""),
            "chapter": item.get("chapter", ""),
            "category": category,
            "doc_id": item.get("doc_id", f"qa_{idx:04d}"),
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
            category = item.get("category") or infer_category(title, item.get("summary", ""), item.get("keywords", ""))
            content = build_manual_content(item)
            if not content.strip():
                continue
            doc_id = f"manual_{normalize_doc_id(file_name)}_{normalize_doc_id(chapter)}_{idx:03d}"
            metadata = {
                "doc_type": "manual_chunk",
                "source": file_name,
                "title": title,
                "chapter": chapter,
                "category": category,
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
        category = item.get("category") or infer_category(question, answer, keywords)
        content = (
            "문서 유형: FAQ\n"
            f"FAQ 분류: {category}\n"
            f"질문: {question}\n"
            f"답변: {answer}\n"
            f"키워드: {', '.join(keywords) if isinstance(keywords, list) else keywords}"
        )
        metadata = {
            "doc_type": "faq",
            "source": item.get("source", "faq_data.json"),
            "title": question,
            "chapter": "FAQ",
            "category": category,
            "doc_id": item.get("id", f"faq_{idx:04d}"),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_domain_guide_documents():
    guides = [
        {
            "doc_id": "domain_guide_return_disuse_disposal",
            "title": "반납, 불용, 처분 구분 가이드",
            "category": "개념비교",
            "keywords": "반납, 불용, 처분, 폐기, 매각, 차이, 순서",
            "content": (
                "반납은 사용 부서가 더 이상 사용하지 않는 물품을 관리 부서로 돌려보내는 절차입니다. "
                "불용은 물품을 계속 사용할 수 없거나 사용할 필요가 없다고 판단하여 사용 중단을 결정하는 단계입니다. "
                "처분은 불용 확정 이후 실제로 폐기, 매각, 멸실, 도난 등으로 자산을 정리하는 단계입니다. "
                "일반적인 흐름은 반납 필요 발생 -> 반납 처리 -> 불용 신청/확정 -> 처분 등록/확정 순서로 이해하면 됩니다. "
                "단, 질문이 특정 메뉴나 버튼 조작을 묻는 경우에는 해당 원문 매뉴얼의 절차를 우선 근거로 삼아야 합니다."
            ),
        },
        {
            "doc_id": "domain_guide_acquisition_approval",
            "title": "취득 등록과 취득 확정 구분 가이드",
            "category": "물품 취득 관리",
            "keywords": "취득, 등록, 확정, 승인요청, 고유번호, G2B",
            "content": (
                "취득 등록은 신규 물품의 기본 정보와 취득 내역을 시스템에 입력하고 저장하는 단계입니다. "
                "취득 확정은 관리자가 등록된 취득 요청을 검토해 최종 승인하는 단계입니다. "
                "물품 고유 번호는 일반적으로 관리자의 취득 확정 이후 생성됩니다. "
                "G2B 목록번호, G2B 목록명, 내용연수, 취득금액 등은 취득 등록 단계의 핵심 입력 또는 자동 매핑 정보입니다."
            ),
        },
        {
            "doc_id": "domain_guide_status_edit_delete",
            "title": "승인 상태별 수정 삭제 가능 여부 가이드",
            "category": "절차",
            "keywords": "대기, 반려, 확정, 수정, 삭제, 승인상태",
            "content": (
                "취득, 반납, 불용, 처분 업무에서 수정과 삭제는 보통 승인 상태가 대기 또는 반려일 때 가능합니다. "
                "이미 확정된 건은 일반 사용자가 임의로 수정하거나 삭제하기 어렵고 관리자 확인이 필요합니다. "
                "질문에 특정 업무 단계가 포함되어 있으면 해당 단계의 원문 매뉴얼 절차를 우선 확인해야 합니다."
            ),
        },
    ]

    documents = []
    for guide in guides:
        content = (
            "문서 유형: 도메인 구분 가이드\n"
            f"제목: {guide['title']}\n"
            f"키워드: {guide['keywords']}\n"
            f"내용: {guide['content']}"
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "doc_type": "domain_guide",
                    "source": "generated_domain_guides",
                    "title": guide["title"],
                    "chapter": "domain_guide",
                    "category": guide["category"],
                    "doc_id": guide["doc_id"],
                },
            )
        )
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
    project_root = os.path.dirname(root_dir)
    target_file = os.path.join(project_root, "dataset", "qa_output", "manual_qa_final.json")
    manual_input_folder = os.path.join(project_root, "dataset", "input")
    faq_file = os.path.join(project_root, "dataset", "FAQ", "faq_data.json")
    
    # DB 저장 경로
    DB_PATH = os.path.join(project_root, "chroma_db")

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
        domain_guide_documents = build_domain_guide_documents()
    except Exception as e:
        print(f"문서 변환 실패: {e}")
        sys.exit(1)

    documents = qa_documents + manual_documents + faq_documents + domain_guide_documents

    # 변환된 데이터가 하나도 없는 경우 처리
    if not documents:
        print("변환할 데이터가 없습니다.")
        sys.exit(1)

    print(f"QA 문서: {len(qa_documents)}개")
    print(f"원문 매뉴얼 문서: {len(manual_documents)}개")
    print(f"FAQ 문서: {len(faq_documents)}개")
    print(f"도메인 구분 가이드 문서: {len(domain_guide_documents)}개")
    print(f"총 {len(documents)}개의 지식을 준비했습니다.")

    # 2. 임베딩 및 DB 저장
    print("Embedding 모델을 준비 중입니다...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
        del vectordb
        gc.collect()
        time.sleep(1)
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
