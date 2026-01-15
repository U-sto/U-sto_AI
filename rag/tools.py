import json
from langchain_core.tools import tool
from typing import Optional

# [메타 데이터] 동의어 및 계층 구조 정의
# 사용자의 구어체(Key)를 데이터베이스 표준 용어(Value 리스트)로 매핑
KEYWORD_SYNONYMS = {
    "pc": ["개인용컴퓨터", "데스크탑", "본체", "컴퓨터"],
    "노트북": ["포터블컴퓨터", "랩탑", "맥북", "그램"],
    "서버": ["네트워크서버", "GPU서버", "메인프레임"],
    "모니터": ["디스플레이", "LCD", "LED"],
    "소융대": ["소프트웨어융합대학"],
    "공대": ["공과대학"]
}

# [가상 데이터베이스] (백엔드 API 대용)
MOCK_ASSET_DB = [
    {
        "asset_id": "2023-0001",
        "asset_name": "고성능 GPU 네트워크서버", # 사용자는 '서버'라고 검색함
        "department": "소프트웨어융합대학",
        "acquisition_date": "2023-03-15",
        "current_status": "사용중",
        "summary": "AI 연구용 GPU 서버 (H100). 305호 서버실 가동 중."
    },
    {
        "asset_id": "2022-5042",
        "asset_name": "행정용 포터블컴퓨터 (LG)", # 사용자는 '노트북'이라고 검색함
        "department": "교무처",
        "acquisition_date": "2022-01-20",
        "current_status": "수리중",
        "summary": "교무처 행정 업무용 노트북. 액정 파손으로 입고."
    },
    {
        "asset_id": "A-1001", 
        "asset_name": "개인용컴퓨터 (i9급)", # 사용자는 'PC'라고 검색함
        "department": "IT 개발팀", 
        "acquisition_date": "2023-01-15", 
        "current_status": "사용 중", 
        "summary": "신규 개발자 지급용 고사양 데스크탑."
    }
]

# 물품 기본 정보 조회 (로직 강화됨)
@tool
def get_asset_basic_info(asset_name: Optional[str] = None, asset_id: Optional[str] = None) -> str:
    """
    물품의 이름(동의어 포함)이나 고유번호를 통해 기본 정보를 조회합니다.
    """
    
    if not asset_name and not asset_id:
        return json.dumps({"error": "검색할 물품명이나 번호를 입력해주세요."})

    found_assets = []
    
    # [검색어 확장 로직] 사용자의 단어를 DB 용어로 확장
    search_keywords = []
    if asset_name:
        # 1. 원래 검색어 추가
        search_keywords.append(asset_name)
        # 2. 동의어 사전에 있으면 확장 검색어 추가
        # 예: "PC" -> ["개인용컴퓨터", "데스크탑", ...]
        clean_name = asset_name.lower().strip()
        if clean_name in KEYWORD_SYNONYMS:
            search_keywords.extend(KEYWORD_SYNONYMS[clean_name])
    
    # 가상 DB 검색
    for asset in MOCK_ASSET_DB:
        # ID 검색 (정확 일치)
        if asset_id and asset_id == asset["asset_id"]:
            found_assets.append(asset)
            continue
        
        # 이름 검색 (확장된 키워드 중 하나라도 포함되면 매칭)
        # 예: asset_name="고성능 GPU 네트워크서버" vs keywords=["서버", "네트워크서버"...]
        if search_keywords:
            for kw in search_keywords:
                if kw in asset["asset_name"]: # 부분 일치 확인
                    found_assets.append(asset)
                    break # 중복 추가 방지

    if not found_assets:
        # 못 찾았을 때 힌트 제공
        return json.dumps({
            "message": "검색 결과가 없습니다.",
            "debug_note": f"검색 시도 키워드: {search_keywords}" 
        }, ensure_ascii=False)

    return json.dumps({
        "count": len(found_assets),
        "results": found_assets
    }, ensure_ascii=False)


# 메뉴/화면 이동
@tool
def navigate_to_page(page_type: str, target_id: Optional[str] = None) -> str:
    """
    사용자가 특정 기능 수행을 원할 때 해당 시스템 화면으로 안내합니다.
    (상세정보, 수명예측, 불용관리 등)
    """
    base_url = "https://univ-asset-system.kr"
    
    url_map = {
        "detail": f"/assets/view/{target_id}" if target_id else "/assets/list",
        "prediction": "/analysis/prediction",
        "disuse": "/assets/disuse/register",
        "return": "/assets/return/register",
        "disposal": "/assets/disposal/register",
        "list": "/assets/list"
    }
    
    key = page_type.lower()
    if "예측" in key or "prediction" in key: key = "prediction"
    elif "상세" in key or "detail" in key: key = "detail"
    elif "불용" in key or "disuse" in key: key = "disuse"
    
    final_url = base_url + url_map.get(key, "/assets/list")
    
    return json.dumps({
        "action": "navigate",
        "url": final_url,
        "message": f"요청하신 {page_type} 화면으로 이동합니다."
    }, ensure_ascii=False)