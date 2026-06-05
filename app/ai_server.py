import json
import os
import joblib
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import openai
import uuid

# ==========================================
# [1] 설정 영역
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


def load_local_env(path: Path = ENV_PATH):
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일 또는 환경변수를 확인하세요.")

# ==========================================
# [1.5] 임시 데이터베이스 (인메모리 DB)
# ==========================================
sessions_db = {}
predictions_db = {}

# ==========================================
# [2] 서버 초기화 및 데이터 모델 정의
# ==========================================

app = FastAPI(
    title="AI Team Server",
    description="백엔드 연동용 챗봇 및 조달 권고 API (FastAPI + OpenAI + CatBoost + LightGBM)",
    version="4.0.0"
)

LIFE_MODEL_PATH = BASE_DIR / "asset_life_model.pkl"
MONTHLY_DEMAND_MODEL_PATH = BASE_DIR / "monthly_demand_model.pkl"
CSV_PATH = BASE_DIR / "phase4_training_data.csv"

LIFE_FEATURES = [
    "내용연수",
    "부서가혹도",
    "월평균사용시간",
    "사용강도지수",
    "누적점검수리횟수",
    "누적수리횟수",
    "최근2년수리횟수",
    "마지막수리후경과개월",
    "취득금액대비수리비율",
    "최대장애심각도",
    "부서예산등급_Code",
    "부서교체성향",
    "G2B목록명_Code",
    "물품분류명_Code",
    "운용부서코드_Code",
]

MONTHLY_DEMAND_FEATURES = [
    "trend",
    "month",
    "month_sin",
    "month_cos",
    "lag_12",
    "rolling_mean_6",
    "rolling_std_6",
]

life_model = None
monthly_demand_model = None
df = None


def unwrap_model(model):
    return getattr(model, "best_estimator_", model)


def load_model(path: Path, label: str):
    if not path.exists():
        print(f"❌ {label} 모델 파일 없음: {path}")
        return None

    try:
        model = unwrap_model(joblib.load(path))
        print(f"✅ {label} 모델 로딩 성공: {path.name}")
        return model
    except Exception as e:
        print(f"❌ {label} 모델 로딩 실패: {e}")
        return None


life_model = load_model(LIFE_MODEL_PATH, "자산 수명")
monthly_demand_model = load_model(MONTHLY_DEMAND_MODEL_PATH, "월별 수요")

# 데이터 파일 로딩
if CSV_PATH.exists():
    try:
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding="cp949")

        code_columns = [
            ("G2B목록명", "G2B목록명_Code"),
            ("물품분류명", "물품분류명_Code"),
            ("운용부서코드", "운용부서코드_Code"),
            ("캠퍼스", "캠퍼스_Code"),
            ("부서예산등급", "부서예산등급_Code"),
        ]
        for src_col, code_col in code_columns:
            if code_col not in df.columns and src_col in df.columns:
                df[code_col] = df[src_col].astype("category").cat.codes

        print("✅ 학습용 데이터 로딩 완료!")
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")

# --- CORS 설정 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --- Request 스키마 정의 ---
class ChatRequest(BaseModel):
    threadId: str
    query: str

class SessionRenameRequest(BaseModel):
    new_title: str

class ForecastRenameRequest(BaseModel):
    new_title: str

# 수정 포인트: 프론트에서 값이 비어있을 때 422 에러 방지를 위해 Optional 처리
class PredictionConditions(BaseModel):
    year: Optional[int] = None
    semester: Optional[str] = None 
    campus: str = "한양대학교 ERICA캠퍼스" 
    dept_name: Optional[str] = None
    category: Optional[str] = None
    risk_level: Optional[str] = None 

class PredictionRequest(BaseModel):
    prompt: str
    conditions: PredictionConditions
 

# ==========================================
# [3] 유틸리티 및 LLM 함수
# ==========================================

def get_lead_time_info(price: float):
    if price <= 20000000:
        return 0, 20.0, 0.48, 7
    elif price < 50000000:
        return 1, 60.0, 0.81, 20
    else:
        return 2, 100.0, 1.12, 38

def calculate_sigma_d(counts_list: list):
    n = len(counts_list)
    if n <= 1: 
        return 0.0
    mean = sum(counts_list) / n
    variance = sum((x - mean) ** 2 for x in counts_list) / (n - 1)
    return math.sqrt(variance)


def get_model_features(model, fallback_features: list):
    if model is None:
        return fallback_features
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_") and getattr(model, "feature_names_", None):
        return list(model.feature_names_)
    if hasattr(model, "feature_name_") and getattr(model, "feature_name_", None):
        return list(model.feature_name_)
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    return fallback_features


def require_columns(dataframe: pd.DataFrame, columns: list, label: str):
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"{label} 입력 컬럼이 데이터에 없습니다: {missing}"
        )


def get_semester_period(year: int, semester: str):
    if semester in ["1", "1학기"]:
        return datetime(year, 3, 2), datetime(year, 6, 20)
    if semester in ["여름", "여름방학", "summer"]:
        return datetime(year, 6, 21), datetime(year, 8, 31)
    if semester in ["2", "2학기"]:
        return datetime(year, 9, 1), datetime(year, 12, 20)
    if semester in ["겨울", "겨울방학", "winter"]:
        return datetime(year, 12, 21), datetime(year + 1, 2, 28)
    return datetime(year, 1, 1), datetime(year, 12, 31)


def filter_current_assets(dataframe: pd.DataFrame):
    if "데이터세트구분" in dataframe.columns:
        prediction_df = dataframe[
            dataframe["데이터세트구분"].astype(str).str.lower() == "prediction"
        ].copy()
        if not prediction_df.empty:
            return prediction_df

    if "학습데이터여부" in dataframe.columns:
        prediction_df = dataframe[
            dataframe["학습데이터여부"].astype(str).str.upper() == "N"
        ].copy()
        if not prediction_df.empty:
            return prediction_df

    return dataframe.copy()


def add_lead_time_columns(dataframe: pd.DataFrame):
    if "취득금액" not in dataframe.columns:
        dataframe["리드타임등급"] = 0
        dataframe["등급점수"] = 20.0
        dataframe["sqrt_L"] = 0.48
        dataframe["리드타임_일"] = 7
        return dataframe

    lead_values = dataframe["취득금액"].fillna(0).apply(get_lead_time_info)
    dataframe["리드타임등급"], dataframe["등급점수"], dataframe["sqrt_L"], dataframe["리드타임_일"] = zip(*lead_values)
    return dataframe


def add_asset_life_predictions(dataframe: pd.DataFrame):
    life_features = get_model_features(life_model, LIFE_FEATURES)
    require_columns(dataframe, life_features, "자산 수명 모델")

    input_data = dataframe[life_features].copy()
    for col in input_data.columns:
        input_data[col] = pd.to_numeric(input_data[col], errors="coerce").fillna(0)

    predicted_total_life_months = pd.Series(
        life_model.predict(input_data),
        index=dataframe.index
    ).clip(lower=1)

    if "운용연차" in dataframe.columns:
        operation_years = pd.to_numeric(dataframe["운용연차"], errors="coerce").fillna(0)
    elif "취득일자" in dataframe.columns:
        acquired_at = pd.to_datetime(dataframe["취득일자"], errors="coerce")
        operation_years = ((pd.Timestamp.now() - acquired_at).dt.days / 365.25).fillna(0)
    else:
        operation_years = pd.Series(0, index=dataframe.index)

    operation_months = operation_years * 12
    predicted_remaining_months = (predicted_total_life_months - operation_months).clip(lower=0)

    dataframe["예측총수명_개월"] = predicted_total_life_months
    dataframe["예측잔여수명"] = predicted_remaining_months
    dataframe["AI예측고장일"] = pd.Timestamp.now() + pd.to_timedelta(
        predicted_remaining_months * 30.4,
        unit="D"
    )
    dataframe["고장예상월기간"] = dataframe["AI예측고장일"].dt.to_period("M")
    return dataframe


def build_monthly_history(history_source_df: pd.DataFrame):
    if "불용일자" not in history_source_df.columns:
        return pd.Series(dtype=float)

    actual_df = history_source_df.copy()
    if "학습데이터여부" in actual_df.columns:
        actual_df = actual_df[actual_df["학습데이터여부"].astype(str).str.upper() == "Y"]

    dates = pd.to_datetime(actual_df["불용일자"], errors="coerce").dropna()
    if dates.empty:
        return pd.Series(dtype=float)

    return dates.dt.to_period("M").value_counts().sort_index().astype(float)


def predict_monthly_demand(history: pd.Series, end_period: pd.Period):
    monthly_features = get_model_features(monthly_demand_model, MONTHLY_DEMAND_FEATURES)

    if history.empty:
        base_period = pd.Timestamp.now().to_period("M") - 24
        history = pd.Series(
            0.0,
            index=pd.period_range(base_period, pd.Timestamp.now().to_period("M"), freq="M")
        )
    else:
        history = history.sort_index()

    counts = history.to_dict()
    base_period = history.index.min()
    current_period = history.index.max() + 1

    while current_period <= end_period:
        last6 = [float(counts.get(current_period - i, 0.0)) for i in range(1, 7)]
        row = {
            "trend": (current_period.year - base_period.year) * 12 + current_period.month - base_period.month + 1,
            "month": current_period.month,
            "month_sin": math.sin(2 * math.pi * current_period.month / 12),
            "month_cos": math.cos(2 * math.pi * current_period.month / 12),
            "lag_12": float(counts.get(current_period - 12, 0.0)),
            "rolling_mean_6": float(np.mean(last6)),
            "rolling_std_6": float(np.std(last6, ddof=1)) if len(last6) > 1 else 0.0,
        }
        input_data = pd.DataFrame([row], columns=monthly_features)
        predicted_qty = float(monthly_demand_model.predict(input_data)[0])
        counts[current_period] = max(0, int(round(predicted_qty)))
        current_period += 1

    return counts


def period_month_count(period_counts: dict, month: int):
    return sum(int(value) for period, value in period_counts.items() if period.month == month)


def average_datetime(series: pd.Series, fallback: datetime):
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return fallback
    return parsed.mean().to_pydatetime()


def is_navigation_request(query: str) -> bool:
    q = query.replace(" ", "")

    strong_markers = [
        "바로가기",
        "이동",
        "열어줘",
        "열어",
        "접속",
        "들어가",
        "링크",
        "가줘",
        "보여줘",
    ]
    if any(marker in q for marker in strong_markers):
        return True

    location_markers = [
        "어디",
        "어디서",
        "위치",
        "어느메뉴",
        "어떤메뉴",
        "무슨메뉴",
    ]
    if any(marker in q for marker in location_markers):
        return True

    action_markers = [
        "등록하고싶",
        "등록하러",
        "등록해줘",
        "조회하고싶",
        "조회하러",
        "조회해줘",
        "관리하고싶",
        "관리하러",
        "처리하고싶",
        "처리하러",
    ]
    return any(marker in q for marker in action_markers)


def get_action_button_for_query(query: str):
    if not is_navigation_request(query):
        return None

    q = query.replace(" ", "")
    button_rules = [
        (["반납"], {"label": "물품 반납 관리 바로가기", "url": "/return-management"}),
        (["불용"], {"label": "불용 관리 바로가기", "url": "/disposal-management"}),
        (["처분"], {"label": "처분 관리 바로가기", "url": "/disposal-management"}),
        (["취득", "자산등록"], {"label": "자산 취득 등록 바로가기", "url": "/acquisition-management"}),
        (["보유현황", "보유현황조회", "상태"], {"label": "보유현황조회 바로가기", "url": "/status-inquiry"}),
        (["사용주기", "AI예측"], {"label": "사용주기 AI 예측 바로가기", "url": "/ai-forecast"}),
    ]

    for keywords, button in button_rules:
        if any(keyword.replace(" ", "") in q for keyword in keywords):
            return button
    return None


def suppress_frontend_auto_buttons(reply: str) -> str:
    replacements = [
        ("[불용/처분 관리]", "해당 업무 화면"),
        ("불용/처분 관리", "해당 업무 화면"),
        ("[물품 불용 관리]", "해당 불용 업무 화면"),
        ("물품 불용 관리", "해당 불용 업무 화면"),
        ("[물품 처분 관리]", "해당 처분 업무 화면"),
        ("물품 처분 관리", "해당 처분 업무 화면"),
        ("[물품 반납 관리]", "해당 반납 업무 화면"),
        ("물품 반납 관리", "해당 반납 업무 화면"),
        ("불용 관리", "불용 업무"),
        ("처분 관리", "처분 업무"),
        ("반납 관리", "반납 업무"),
    ]

    for old, new in replacements:
        reply = reply.replace(old, new)
    return reply


def build_navigation_reply(action_button: dict) -> str:
    if action_button["url"] == "/return-management":
        return (
            "해당 업무 화면은 [물품 관리] 메뉴의 [물품 운용 관리] 하위에서 이용할 수 있습니다. "
            "이 페이지에서 등록 목록과 승인 상태를 조회하고, 신규 등록을 시작할 수 있습니다."
        )

    if action_button["url"] == "/disposal-management":
        return (
            "해당 업무 화면은 [물품 관리] 메뉴 하위에서 이용할 수 있습니다. "
            "이 페이지에서 등록 목록과 승인 상태를 조회하고, 신규 등록을 시작할 수 있습니다."
        )

    return (
        "해당 업무 화면에서 관련 목록을 조회하고 필요한 작업을 진행할 수 있습니다. "
        "아래 바로가기 버튼을 눌러 이동해 주세요."
    )

REPORT_SYSTEM_PROMPT = """
당신은 대학 자산 관리 실무자를 돕는 'SCM AI 분석 파트너'입니다.
제공된 분석 데이터와 '사용자 요청(Prompt)'을 바탕으로 대시보드 패널에 들어갈 [AI 최적화 요약 코멘트]를 작성해주세요.

[지시사항]
1. 단순히 수치만 기계적으로 나열하지 마세요.
2. '사용자 요청(Prompt)'의 말투나 질문 의도를 파악하여, 그에 대한 직접적인 대답이 되도록 자연스럽게 1~2문장으로 요약하세요.
3. 분석 데이터의 고장 집중월(peak_month)과 권장 발주마감일(rec_date_str)을 근거로 실무적인 조언을 포함하세요.

반드시 아래 JSON 형식으로 응답하세요:
{
  "ai_summary_comment": "요약 코멘트 내용"
}
"""

def get_llm_ai_guide(prompt: str, target_item: str, total_qty: int, rec_date_str: str, peak_month: int):
    try:
        context = f"품목:{target_item}, 총 필요수량:{total_qty}개, 최적 발주마감일:{rec_date_str}, 고장집중월:{peak_month}월. 사용자요청:{prompt}"
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "system", "content": REPORT_SYSTEM_PROMPT}, {"role": "user", "content": context}],
            response_format={"type": "json_object"},
            temperature=0.7 
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "ai_summary_comment": f"수요 분석 결과, {peak_month}월 전후로 노후 장비 처분이 예측됩니다. 원활한 실습실 운영을 위해 가이드된 일정에 맞춰 발주를 진행해 주세요."
        }

# ==========================================
# [4] 챗봇 세션(쓰레드) 관리 API 
# ==========================================

@app.post("/api/ai/chat/threads")
async def create_thread():
    threadId = str(uuid.uuid4())
    sessions_db[threadId] = {
        "title": "새 채팅",
        "messages": []
    }
    return {"status": "success", "data": {"threadId": threadId, "title": "새 채팅"}}

@app.get("/api/ai/chat/threads")
async def get_threads():
    thread_list = [{"threadId": tid, "title": data["title"]} for tid, data in sessions_db.items()]
    return {"status": "success", "data": thread_list}

@app.put("/api/ai/chat/threads/{threadId}")
async def rename_thread(threadId: str, req: SessionRenameRequest):
    if threadId not in sessions_db:
        raise HTTPException(status_code=404, detail="쓰레드를 찾을 수 없습니다.")
    sessions_db[threadId]["title"] = req.new_title
    return {"status": "success", "message": "이름이 변경되었습니다.", "data": {"threadId": threadId, "new_title": req.new_title}}

@app.delete("/api/ai/chat/threads")
async def delete_thread(threadId: str):
    if threadId in sessions_db:
        del sessions_db[threadId]
        return {"status": "success", "message": "삭제 완료"}
    raise HTTPException(status_code=404, detail="쓰레드를 찾을 수 없습니다.")

@app.get("/api/ai/chat/messages/{threadId}/search")
async def get_thread_messages(threadId: str):
    # 수정 포인트: 프론트엔드가 [+] 버튼 등으로 새 쓰레드ID만 가지고 조회 요청을 할 때 터지지 않도록 방어 로직 추가
    if threadId not in sessions_db:
        sessions_db[threadId] = {"title": "새 채팅", "messages": []}
    return {"status": "success", "data": sessions_db[threadId]["messages"]}

@app.get("/api/ai/chat/messages/search")
async def search_all_messages(keyword: Optional[str] = None):
    result = []
    for tid, data in sessions_db.items():
        for msg in data["messages"]:
            if keyword is None or keyword.lower() in msg["content"].lower():
                result.append({
                    "threadId": tid, 
                    "role": msg["role"], 
                    "content": msg["content"],
                    "created_at": msg.get("created_at", "")
                })
    return {"status": "success", "data": result}


# ==========================================
# [5] API 엔드포인트 (AI 응답)
# ==========================================

@app.post("/api/ai/chat")
async def chat_completions(req: ChatRequest):
    current_time = datetime.now().isoformat()
    
    if req.threadId not in sessions_db:
        sessions_db[req.threadId] = {"title": req.query[:10], "messages": []}
    
    history = sessions_db[req.threadId]["messages"]
    history.append({"role": "user", "content": req.query, "created_at": current_time})

    q = req.query.replace(" ", "") 
    selected_file = None
    
    if any(w in q for w in ["취득", "취득정리구분", "취득일자", "정리일자", "자산등록"]): selected_file = "manual_chapter1.json"
    elif any(w in q for w in ["운용", "라벨", "물품고유번호", "운용대장"]): selected_file = "manual_chapter2.json"
    elif any(w in q for w in ["반납", "반납사유", "반납일자", "반납확정일자"]): selected_file = "manual_chapter3.json"
    elif any(w in q for w in ["불용", "불용일자", "불용확정일자"]): selected_file = "manual_chapter4.json"
    elif any(w in q for w in ["처분", "처분정리구분", "처분일자", "처분확정일자"]): selected_file = "manual_chapter5.json"
    elif any(w in q for w in ["보유현황", "보유", "현황", "조회기준", "목록"]): selected_file = "manual_chapter6.json"
    elif any(w in q for w in ["사용주기", "AI예측", "수명", "교체시기", "분석"]): selected_file = "manual_chapter7.json"
    elif any(w in q for w in ["챗봇", "도움말", "사용법", "가이드"]): selected_file = "manual_chapter8.json"

    manual_content = ""
    refs = []
    if selected_file:
        refs = [selected_file] 
        try:
            if os.path.exists(selected_file):
                with open(selected_file, "r", encoding="utf-8") as f:
                    manual_content = json.dumps(json.load(f), ensure_ascii=False)
        except Exception as e: pass

    # 메뉴 이동 질문은 RAG 답변을 생성하지 않고 바로가기 응답만 한 번 반환한다.
    # 프론트의 키워드 기반 자동 버튼과 서버 action_buttons가 동시에 뜨는 것을 막기 위함이다.
    action_button = get_action_button_for_query(req.query)
    if action_button:
        ai_reply = build_navigation_reply(action_button)
        history.append({"role": "assistant", "content": ai_reply, "created_at": current_time})
        return {
            "status": "success",
            "data": {
                "reply": ai_reply,
                "action_buttons": [action_button],
                "references": [],
                "source_references": refs,
                "created_at": current_time
            }
        }

    # 수정 포인트: 보유현황조회 안내 및 메뉴 라우팅 강화
    sys_inst = f"""당신은 대학 물품관리시스템을 돕는 똑똑하고 친절한 AI 챗봇입니다.
    아래 제공된 [매뉴얼 데이터]를 최우선으로 참고하여 답변하세요. 

    [핵심 지시사항]
    1. 매뉴얼 관련 질문: 매뉴얼의 절차, 용어, 기준을 바탕으로 정확하고 쉽게 안내하세요.
    2. 개별 물품고유번호(예: 12345번 등) 또는 특정 물품의 상태 조회 요청: "AI 챗봇은 개별 물품의 상태를 직접 조회할 수 없습니다. 개별 물품 상세 정보는 [보유현황조회] 메뉴를 이용해 확인해 주세요."라고 구체적이고 명확하게 안내하세요.
    3. 매뉴얼에 없는 내용이나 일상 질문: 대화를 거절하지 말고, 일반적인 지식을 활용해 자연스럽고 유용하게 답변하세요.

    [매뉴얼 데이터]
    {manual_content}
    """

    try:
        messages_for_llm = [{"role": "system", "content": sys_inst}] + history

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages_for_llm,
            temperature=0.6 
        )
        ai_reply = response.choices[0].message.content

        history.append({"role": "assistant", "content": ai_reply, "created_at": current_time})

        return {
            "status": "success", 
            "data": {
                "reply": ai_reply, 
                "action_buttons": [], 
                "references": [],
                "source_references": refs,
                "created_at": current_time
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/ai/forecast")
async def predict_analysis(req: PredictionRequest):
    if life_model is None or monthly_demand_model is None or df is None:
        return {"status": "error", "message": "모델이나 데이터가 없습니다."}
        
    # 수정 포인트: 분석 조건 필수 입력 방어 코드 추가
    cond = req.conditions
    if not cond.year or not cond.semester or not cond.dept_name:
        raise HTTPException(status_code=400, detail="분석조건(운용부서, 년도, 학기)을 필수로 입력해주세요.")

    try:
        # 1. 대상 데이터 필터링: 이력 데이터는 월별 수요 모델용, 현재 자산은 수명 모델용으로 분리
        source_df = df[df['운용부서명'] == cond.dept_name].copy()
        if cond.category and cond.category != "전체":
            source_df = source_df[source_df['물품분류명'] == cond.category]

        if source_df.empty:
            return {
                "status": "success", 
                "data": { 
                    "section_1_time_series": [], 
                    "section_2_strategic_guide": {}, 
                    "section_3_recommendations": [],
                    "section_4_algorithm_guide": {}
                }
            }

        # 2. 현재 보유/예측 대상 자산에 대해 CatBoost 자산 수명 모델 적용
        target_df = filter_current_assets(source_df)
        target_df = add_lead_time_columns(target_df)
        target_df = add_asset_life_predictions(target_df)

        # 3. 요청 기간과 월별 수요 예측값 산출
        req_year = int(cond.year)
        start_date, end_date = get_semester_period(req_year, cond.semester)
        target_periods = pd.period_range(
            pd.Timestamp(start_date).to_period("M"),
            pd.Timestamp(end_date).to_period("M"),
            freq="M"
        )

        filtered_df = target_df[
            (target_df['AI예측고장일'] >= start_date)
            & (target_df['AI예측고장일'] <= end_date)
        ].copy()

        asset_monthly_counts = (
            filtered_df.groupby("고장예상월기간").size().to_dict()
            if not filtered_df.empty
            else {}
        )
        monthly_history = build_monthly_history(source_df)
        monthly_demand_counts = predict_monthly_demand(monthly_history, target_periods.max())

        # 4. 월별 고장예상수량: 자산 수명 모델의 개별 고장월과 LightGBM 월별 수요를 결합
        final_monthly_counts = {}
        for period in target_periods:
            asset_qty = int(asset_monthly_counts.get(period, 0))
            demand_qty = int(monthly_demand_counts.get(period, 0))
            final_monthly_counts[period] = max(asset_qty, demand_qty)

        target_months = sorted({period.month for period in target_periods})

        # -------------------------------------------------------------------
        # [구역 3] 조달권고안 (개별 품목별 데이터)
        # -------------------------------------------------------------------
        recommendations = []
        rop_trigger_months = [] 

        total_base_qty_all = 0
        total_safety_stock_all = 0
        
        # 수정 포인트: High 선택 시 버퍼를 크게 잡아 안전하고 "일찍" 발주하도록 매핑값 반전 변경
        z_val_map = {
            "Low": 0.0, "LOW": 0.0,         # 리스크 수용(재고 안 둠) -> 늦은 발주
            "Medium": 1.28, "MEDIUM": 1.28, # 표준 타협
            "High": 1.65, "HIGH": 1.65      # 결품 리스크 회피 최우선 -> 안전 재고 증가 -> 앞당겨진 이른 발주
        }
        
        z_val = z_val_map.get(cond.risk_level, 1.28)
        buffer_days_map = {
            "Low": 0, "LOW": 0,
            "Medium": 14, "MEDIUM": 14,
            "High": 30, "HIGH": 30,
        }
        buffer_days = buffer_days_map.get(cond.risk_level, 14)

        asset_period_total = sum(int(asset_monthly_counts.get(period, 0)) for period in target_periods)
        demand_period_total = sum(int(final_monthly_counts.get(period, 0)) for period in target_periods)
        demand_scale = max(1.0, demand_period_total / asset_period_total) if asset_period_total > 0 else 1.0

        if not filtered_df.empty:
            grouped = filtered_df.groupby('G2B목록명')
            item_id = 1
            
            for item_name, group_df in grouped:
                monthly_counts_item = group_df.groupby('고장예상월기간').size().to_dict()
                counts_list = [
                    math.ceil(monthly_counts_item.get(period, 0) * demand_scale)
                    for period in target_periods
                ]
                
                monthly_avg_demand = sum(counts_list) / len(target_months) if target_months else 0
                avg_lead_days = float(group_df['리드타임_일'].mean()) if '리드타임_일' in group_df.columns else 20.0
                lead_time_months = avg_lead_days / 30.4
                avg_sqrt_L = float(group_df['sqrt_L'].mean()) if 'sqrt_L' in group_df.columns else math.sqrt(lead_time_months)
                
                sigma_d = calculate_sigma_d(counts_list)
                safety_stock = math.ceil(z_val * sigma_d * avg_sqrt_L)
                rop_qty = math.ceil((monthly_avg_demand * lead_time_months) + safety_stock)
                
                cumulative_demand = 0
                trigger_period = target_periods[0] if len(target_periods) > 0 else pd.Timestamp.now().to_period("M")
                rop_triggered = False
                
                for period, period_qty in zip(target_periods, counts_list):
                    cumulative_demand += period_qty
                    if cumulative_demand >= rop_qty and not rop_triggered:
                        trigger_period = period
                        rop_triggered = True
                
                if not rop_triggered:
                    trigger_period = max(
                        target_periods,
                        key=lambda period: monthly_counts_item.get(period, 0)
                    ) if len(target_periods) > 0 else pd.Timestamp.now().to_period("M")
                    
                rop_trigger_months.append(trigger_period.month)
                
                base_qty = max(1, math.ceil(len(group_df) * demand_scale))
                total_req_qty = base_qty + safety_stock
                
                unit_price = int(group_df['취득금액'].mean()) if len(group_df) > 0 else 0
                urgent_budget = total_req_qty * unit_price 
                
                total_base_qty_all += base_qty
                total_safety_stock_all += safety_stock
                
                pred_failure_date = average_datetime(group_df['AI예측고장일'], start_date)
                
                # buffer_days가 클수록(High risk level) 더 일찍 발주하게 됨.
                rec_order_date = pred_failure_date - timedelta(days=(avg_lead_days + buffer_days))
                rec_order_date = max(rec_order_date, datetime.now())
                
                recommendations.append({
                    "id": item_id,
                    "item_name": item_name,
                    "quantity": total_req_qty, 
                    "unit_price": unit_price,
                    "estimated_budget": urgent_budget,
                    "recommend_order_date": rec_order_date.strftime("%Y-%m-%d"),
                    "base_qty": base_qty,
                    "safety_stock": safety_stock,
                    "rop": rop_qty,
                    "lead_time_days": round(avg_lead_days, 1),
                    "monthly_avg_demand": round(monthly_avg_demand, 2),
                })
                item_id += 1 
                
        else:
            target_item = cond.category if cond.category and cond.category != "전체" else "전체 품목"
            demand_counts_list = [int(final_monthly_counts.get(period, 0)) for period in target_periods]
            base_qty = sum(demand_counts_list)

            if base_qty > 0:
                avg_lead_days = float(target_df['리드타임_일'].mean()) if '리드타임_일' in target_df.columns else 20.0
                lead_time_months = avg_lead_days / 30.4
                avg_sqrt_L = float(target_df['sqrt_L'].mean()) if 'sqrt_L' in target_df.columns else math.sqrt(lead_time_months)
                sigma_d = calculate_sigma_d(demand_counts_list)
                safety_stock = math.ceil(z_val * sigma_d * avg_sqrt_L)
                total_req_qty = base_qty + safety_stock
                monthly_avg_demand = base_qty / len(target_periods) if len(target_periods) > 0 else 0
                rop_qty = math.ceil((monthly_avg_demand * lead_time_months) + safety_stock)
                peak_period = max(target_periods, key=lambda period: final_monthly_counts.get(period, 0))
                rop_trigger_months.append(peak_period.month)

                unit_price = int(target_df['취득금액'].mean()) if '취득금액' in target_df.columns else 0
                urgent_budget = total_req_qty * unit_price
                rec_order_date = peak_period.to_timestamp(how="start").to_pydatetime() - timedelta(days=(avg_lead_days + buffer_days))
                rec_order_date = max(rec_order_date, datetime.now())

                total_base_qty_all = base_qty
                total_safety_stock_all = safety_stock
                recommendations.append({
                    "id": 1,
                    "item_name": target_item,
                    "quantity": total_req_qty,
                    "unit_price": unit_price,
                    "estimated_budget": urgent_budget,
                    "recommend_order_date": rec_order_date.strftime("%Y-%m-%d"),
                    "base_qty": base_qty,
                    "safety_stock": safety_stock,
                    "rop": rop_qty,
                    "lead_time_days": round(avg_lead_days, 1),
                    "monthly_avg_demand": round(monthly_avg_demand, 2),
                })
            else:
                recommendations.append({
                    "id": 1,
                    "item_name": target_item,
                    "quantity": 0,
                    "unit_price": 0,
                    "estimated_budget": 0,
                    "recommend_order_date": "-",
                    "base_qty": 0,
                    "safety_stock": 0,
                    "rop": 0,
                    "lead_time_days": 0,
                    "monthly_avg_demand": 0,
                })
        
        valid_dates = [datetime.strptime(r['recommend_order_date'], "%Y-%m-%d") for r in recommendations if r['recommend_order_date'] != "-"]
        earliest_order_date = min(valid_dates).strftime("%Y-%m-%d") if valid_dates else "-"
        final_rop_month = min(valid_dates).month if valid_dates else 0

        # -------------------------------------------------------------------
        # [구역 1] 수요 예측 시계열
        # -------------------------------------------------------------------
        time_series = []
        for m in range(1, 13):
            qty = period_month_count(final_monthly_counts, m) if m in target_months else 0
            is_rop_flag = (m == final_rop_month)
            
            ts_item = {
                "month": m,
                "quantity": qty,
                "is_rop": is_rop_flag
            }
            if is_rop_flag:
                ts_item["rop_date"] = earliest_order_date 
                ts_item["base_qty"] = total_base_qty_all 
                ts_item["safety_stock"] = total_safety_stock_all
                ts_item["total_order_qty"] = total_base_qty_all + total_safety_stock_all 
                
            time_series.append(ts_item)

        # -------------------------------------------------------------------
        # [구역 2] AI 전략적 조달 가이드 (좌측 패널 - 전체 요약)
        # -------------------------------------------------------------------
        total_qty_all = sum(r['quantity'] for r in recommendations)
        total_budget_all = sum(r['estimated_budget'] for r in recommendations)

        if total_qty_all > 0:
            
            target_item_name = cond.category if cond.category and cond.category != "전체" else "전체 품목"
            peak_period = max(target_periods, key=lambda period: final_monthly_counts.get(period, 0))
            peak_month = max(rop_trigger_months, key=rop_trigger_months.count) if rop_trigger_months else peak_period.month
            
            ai_guide_data = get_llm_ai_guide(req.prompt, target_item_name, total_qty_all, earliest_order_date, peak_month)
            
            # Risk Level 표기 맵핑 조정 반영
            service_level_map = {"Low": "50% 수준", "Medium": "90% 수준", "High": "95% 이상 안정"}
            sl_text = service_level_map.get(cond.risk_level, "90% 수준")

            budget_in_thousands = total_budget_all // 1000
            
            ai_strategic_guide = {
                "ai_summary_comment": ai_guide_data.get("ai_summary_comment", ""),
                "smart_forecasting": f"CatBoost 자산 수명 모델로 예측잔여수명을 계산해 AI예측고장일을 만들고, LightGBM 월별 수요 모델로 월별 고장예상수량을 보정했습니다. 분석 기간의 기본 고장 예상 수량({total_base_qty_all}개)에 안전재고({total_safety_stock_all}개)를 더해 {sl_text} 서비스 수준 기준 총 {total_qty_all}대의 필요 수량을 산출했습니다.",
                "time_to_procure": f"ROP와 리드타임, 리스크 버퍼를 역산한 결과입니다. 수업 운영에 차질이 없도록 늦어도 {earliest_order_date} 이전까지 발주 절차를 진행하는 것이 적합합니다.",
                "budget_guide": f"해당 수량 조달 및 설치를 위해 약 {budget_in_thousands:,}천 원의 예산 확보를 권고합니다."
            }
        else:
            ai_strategic_guide = {
                "ai_summary_comment": "선택하신 기간 내 교체가 필요한 노후 장비가 발견되지 않았습니다.",
                "smart_forecasting": "고장 예상 수량 및 필요 안전 재고가 0대로 도출되었습니다.",
                "time_to_procure": "현재 양호한 상태를 유지 중이므로 당장의 발주 절차는 필요하지 않습니다.",
                "budget_guide": "해당 기간 내 추가 조달로 요구되는 예산은 없습니다."
            }
        
        # -------------------------------------------------------------------
        # [구역 4] AI 분석 알고리즘 가이드
        # -------------------------------------------------------------------
        algorithm_guide = {
            "formula_1": "예측잔여수명 = CatBoost 예측총수명(개월) - 현재 운용개월",
            "formula_2": "AI예측고장일 = 현재일 + 예측잔여수명 X 30.4일",
            "formula_3": "월별 고장예상수량 = max(자산별 AI예측고장월 집계, LightGBM 월별 수요 예측)",
            "formula_4": "안전재고 = Z값(리스크 수준) X 월별 수요 표준편차 X sqrt(리드타임)",
            "formula_5": "ROP = 월평균 수요 X 리드타임 + 안전재고",
            "formula_6": "권장발주기한 = AI예측고장일 - 리드타임 - 리스크 버퍼"
        }

        forecastId = f"pred-{str(uuid.uuid4())[:8]}"
        created_at = datetime.now().isoformat()
        
        # =========================================================
        # 프론트엔드가 화면 상단에 그릴 수 있도록 
        # prompt, target, risk, period 데이터를 final_result의 1Depth에 추가
        # =========================================================
        final_result = {
            "forecastId": forecastId,
            "created_at": created_at,
            "prompt": req.prompt,                                # <-- 프론트엔드 '이전 예측' 영역에 표시될 질문 내용
            "target": cond.dept_name,                            # <-- Target 표시용
            "risk": cond.risk_level,                             # <-- Risk 표시용
            "period": f"{cond.year} - {cond.semester}",          # <-- Period 표시용 (예: "2030 - 2학기")
            "conditions": {                                      # <-- 원본 조건도 백업용으로 전달 (필요시 프론트 사용)
                "year": cond.year,
                "semester": cond.semester,
                "dept_name": cond.dept_name,
                "category": cond.category,
                "risk_level": cond.risk_level
            },
            "section_1_time_series": time_series,
            "section_2_strategic_guide": ai_strategic_guide,
            "section_3_recommendations": recommendations,
            "section_4_algorithm_guide": algorithm_guide
        }
        
        predictions_db[forecastId] = {
            "title": req.prompt[:15] + "..." if len(req.prompt) > 15 else req.prompt,
            "prompt": req.prompt,
            "created_at": created_at,
            "data": final_result
        }

        return final_result

    except Exception as e:
        print(f"서버 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")


# ==========================================
# [5.2] 예측 기록 관리 API (GET, DELETE)
# ==========================================

@app.get("/api/ai/forecast")
async def get_forecast_history():
    history_list = []
    for hid, info in reversed(predictions_db.items()):
        history_list.append({
            "forecastId": hid,
            "title": info.get("title", info["prompt"]), 
            "prompt": info["prompt"],
            "created_at": info["created_at"]
        })
    return {"status": "success", "data": history_list}

@app.get("/api/ai/forecast/contents/{forecastId}")
async def get_forecast_contents(forecastId: str):
    if forecastId not in predictions_db:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    return predictions_db[forecastId]["data"]

@app.delete("/api/ai/forecast")
async def delete_forecast_history(forecastId: str):
    if forecastId in predictions_db:
        del predictions_db[forecastId]
        return {"status": "success", "message": "기록이 삭제되었습니다."}
    raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")

@app.put("/api/ai/forecast/{forecastId}")
async def rename_forecast_history(forecastId: str, req: ForecastRenameRequest):
    if forecastId not in predictions_db:
        raise HTTPException(status_code=404, detail="기록을 찾을 수 없습니다.")
    
    predictions_db[forecastId]["title"] = req.new_title
    return {
        "status": "success", 
        "message": "예측 기록 이름이 변경되었습니다.", 
        "data": {"forecastId": forecastId, "new_title": req.new_title}
    }


