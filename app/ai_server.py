import json
import os
import joblib
import pandas as pd
import math
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import openai
import uuid

# ==========================================
# [1] 설정 영역
# ==========================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AI_MODEL = "gpt-4o" 

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
    description="백엔드 연동용 챗봇 및 통계 예측 API (FastAPI + OpenAI + Random Forest)",
    version="3.4.1"
)

MODEL_PATH = "rf_final_model.pkl"
CSV_PATH = "phase4_training_data.csv"

rf_model = None
df = None

# 모델 로딩
if os.path.exists(MODEL_PATH):
    try:
        rf_model = joblib.load(MODEL_PATH)
        print("✅ AI 모델 로딩 성공!")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")

# 데이터 파일 로딩
if os.path.exists(CSV_PATH):
    try:
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding="cp949")
            
        df['G2B목록명_Code'] = df['G2B목록명'].astype('category').cat.codes
        df['물품분류명_Code'] = df['물품분류명'].astype('category').cat.codes
        df['운용부서코드_Code'] = df['운용부서코드'].astype('category').cat.codes
        df['캠퍼스_Code'] = df['캠퍼스'].astype('category').cat.codes
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

client = openai.OpenAI(api_key=OPENAI_API_KEY)

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

        # 수정 포인트: 질문뿐만 아니라 AI 답변 내용까지 파악하여 관련된 바로가기 버튼을 모두 띄움
        action_buttons = []
        q_and_a = req.query + " " + ai_reply
        
        if any(w in q_and_a for w in ["반납"]):
            action_buttons.append({"label": "물품 반납 관리 바로가기", "url": "/return-management"})
        if any(w in q_and_a for w in ["처분", "불용"]):
            action_buttons.append({"label": "불용/처분 관리 바로가기", "url": "/disposal-management"})
        if any(w in q_and_a for w in ["취득", "자산등록"]):
            action_buttons.append({"label": "자산 취득 등록 바로가기", "url": "/acquisition-management"})
        if any(w in q_and_a for w in ["보유현황", "보유현황조회", "상태"]):
            action_buttons.append({"label": "보유현황조회 바로가기", "url": "/status-inquiry"})
        if any(w in q_and_a for w in ["사용주기", "AI예측", "AI 예측"]):
            action_buttons.append({"label": "사용주기 AI 예측 바로가기", "url": "/ai-forecast"})

        # 중복된 버튼 제거 (같은 목적지의 버튼이 여러 개 달리는 것 방지)
        unique_buttons = {btn["label"]: btn for btn in action_buttons}.values()
        action_buttons = list(unique_buttons)

        return {
            "status": "success", 
            "data": {
                "reply": ai_reply, 
                "action_buttons": action_buttons, 
                "references": refs, 
                "created_at": current_time
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/ai/forecast")
async def predict_analysis(req: PredictionRequest):
    if rf_model is None or df is None:
        return {"status": "error", "message": "모델이나 데이터가 없습니다."}
        
    # 수정 포인트: 분석 조건 필수 입력 방어 코드 추가
    cond = req.conditions
    if not cond.year or not cond.semester or not cond.dept_name:
        raise HTTPException(status_code=400, detail="분석조건(운용부서, 년도, 학기)을 필수로 입력해주세요.")

    try:
        # 1. 대상 데이터 필터링
        target_df = df[df['운용부서명'] == cond.dept_name].copy()
        if cond.category and cond.category != "전체":
            target_df = target_df[target_df['물품분류명'] == cond.category]

        if target_df.empty:
            return {
                "status": "success", 
                "data": { 
                    "section_1_time_series": [], 
                    "section_2_strategic_guide": {}, 
                    "section_3_recommendations": [],
                    "section_4_algorithm_guide": {}
                }
            }

        # 2. 파생변수 동적 계산
        if '취득금액' in target_df.columns:
            target_df['리드타임등급'], target_df['등급점수'], target_df['sqrt_L'], target_df['리드타임_일'] = zip(*target_df['취득금액'].apply(get_lead_time_info))
            if '가격민감도' in target_df.columns:
                target_df['장비중요도'] = (target_df['가격민감도'] * 100 * 0.5) + (target_df['등급점수'] * 0.5)

        # 3. AI 모델 예측 수행
        features = ['내용연수', '취득금액', '부서가혹도', '가격민감도', '장비중요도', 'G2B목록명_Code', '물품분류명_Code', '운용부서코드_Code', '캠퍼스_Code']
        input_data = target_df[features]
        target_df['예측수명_월'] = rf_model.predict(input_data)
        
        # 4. 고장 예상일 계산
        target_df['고장예상일'] = target_df['예측수명_월'].apply(
            lambda x: datetime.now() + timedelta(days=float(x) * 30.4)
        )

        req_year = int(cond.year)
        req_semester = cond.semester
        
        # 학기별 날짜 필터링
        if req_semester in ["1", "1학기"]:      
            start_date = datetime(req_year, 3, 2)
            end_date = datetime(req_year, 6, 20)
        elif req_semester in ["여름", "여름방학", "summer"]: 
            start_date = datetime(req_year, 6, 21)
            end_date = datetime(req_year, 8, 31)
        elif req_semester in ["2", "2학기"]:    
            start_date = datetime(req_year, 9, 1)
            end_date = datetime(req_year, 12, 20)
        elif req_semester in ["겨울", "겨울방학", "winter"]: 
            start_date = datetime(req_year, 12, 21)
            end_date = datetime(req_year + 1, 2, 28)
        else: 
            start_date = datetime(req_year, 1, 1)
            end_date = datetime(req_year, 12, 31)
            
        filtered_df = target_df[(target_df['고장예상일'] >= start_date) & (target_df['고장예상일'] <= end_date)].copy()

        if start_date.year == end_date.year:
            target_months = list(range(start_date.month, end_date.month + 1))
        else:
            target_months = list(range(start_date.month, 13)) + list(range(1, end_date.month + 1))

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
        model_rmse = 5.0 
        buffer_days = math.ceil(z_val * model_rmse) 

        if not filtered_df.empty:
            filtered_df['고장예상월'] = filtered_df['고장예상일'].dt.month
            grouped = filtered_df.groupby('G2B목록명')
            item_id = 1
            
            for item_name, group_df in grouped:
                monthly_counts_item = group_df.groupby('고장예상월').size().to_dict()
                counts_list = [monthly_counts_item.get(m, 0) for m in target_months]
                
                monthly_avg_demand = sum(counts_list) / len(target_months) if target_months else 0
                avg_lead_days = group_df['리드타임_일'].mean()
                lead_time_months = avg_lead_days / 30.4
                avg_sqrt_L = group_df['sqrt_L'].mean()
                
                sigma_d = calculate_sigma_d(counts_list)
                safety_stock = math.ceil(z_val * sigma_d * avg_sqrt_L)
                rop_qty = math.ceil((monthly_avg_demand * lead_time_months) + safety_stock)
                
                cumulative_demand = 0
                trigger_month = target_months[0] if target_months else 1
                rop_triggered = False
                
                for m in target_months:
                    cumulative_demand += monthly_counts_item.get(m, 0)
                    if cumulative_demand >= rop_qty and not rop_triggered:
                        trigger_month = m
                        rop_triggered = True
                
                if not rop_triggered:
                    trigger_month = max(target_months, key=lambda x: monthly_counts_item.get(x, 0)) if target_months else 1
                    
                rop_trigger_months.append(trigger_month)
                
                base_qty = len(group_df)
                total_req_qty = base_qty + safety_stock
                
                unit_price = int(group_df['취득금액'].mean()) if len(group_df) > 0 else 0
                urgent_budget = total_req_qty * unit_price 
                
                total_base_qty_all += base_qty
                total_safety_stock_all += safety_stock
                
                avg_timestamp = group_df['고장예상일'].astype('int64').mean()
                pred_failure_date = pd.to_datetime(avg_timestamp)
                
                # buffer_days가 클수록(High risk level) rec_order_date는 날짜가 뺴지므로 더 일찍 발주하게 됨.
                rec_order_date = pred_failure_date - timedelta(days=(avg_lead_days + buffer_days))
                
                recommendations.append({
                    "id": item_id,
                    "item_name": item_name,
                    "quantity": total_req_qty, 
                    "unit_price": unit_price,
                    "estimated_budget": urgent_budget,
                    "recommend_order_date": rec_order_date.strftime("%Y-%m-%d")
                })
                item_id += 1 
                
        else:
            target_item = cond.category if cond.category and cond.category != "전체" else "전체 품목"
            recommendations.append({
                "id": 1,
                "item_name": target_item,
                "quantity": 0,
                "unit_price": 0,
                "estimated_budget": 0,
                "recommend_order_date": "-"
            })
        
        valid_dates = [datetime.strptime(r['recommend_order_date'], "%Y-%m-%d") for r in recommendations if r['recommend_order_date'] != "-"]
        earliest_order_date = min(valid_dates).strftime("%Y-%m-%d") if valid_dates else "-"
        final_rop_month = min(valid_dates).month if valid_dates else 0

        # -------------------------------------------------------------------
        # [구역 1] 수요 예측 시계열
        # -------------------------------------------------------------------
        if not filtered_df.empty:
            monthly_counts_total = filtered_df.groupby('고장예상월').size().to_dict()
        else:
            monthly_counts_total = {}
            
        time_series = []
        for m in range(1, 13):
            qty = int(monthly_counts_total.get(m, 0)) if m in target_months else 0
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
        if not filtered_df.empty:
            total_qty_all = sum(r['quantity'] for r in recommendations)
            total_budget_all = sum(r['estimated_budget'] for r in recommendations)
            
            target_item_name = cond.category if cond.category and cond.category != "전체" else "전체 품목"
            peak_month = max(rop_trigger_months, key=rop_trigger_months.count) if rop_trigger_months else final_rop_month
            
            ai_guide_data = get_llm_ai_guide(req.prompt, target_item_name, total_qty_all, earliest_order_date, peak_month)
            
            # Risk Level 표기 맵핑 조정 반영
            service_level_map = {"Low": "50% 수준", "Medium": "90% 수준", "High": "95% 이상 안정"}
            sl_text = service_level_map.get(cond.risk_level, "90% 수준")

            budget_in_thousands = total_budget_all // 1000
            
            ai_strategic_guide = {
                "ai_summary_comment": ai_guide_data.get("ai_summary_comment", ""),
                "smart_forecasting": f"분석 기간 내 발생할 것으로 예상되는 순수 고장 예상 수량({total_base_qty_all}개)에, 예상치 못한 장비 부족으로 인한 수업 결손을 방지하기 위한 안전 재고({total_safety_stock_all}개)를 더하여 최종 권장 발주 수량을 산출했습니다. (설정된 {sl_text} 서비스 수준 기준, 총 {total_qty_all}대의 필요 수량 도출)",
                "time_to_procure": f"물품 발주부터 실제 실습실 설치까지 소요되는 리드 타임(Lead Time)을 역산하여 산출한 결과입니다. 수업 운영에 차질이 없도록 늦어도 {earliest_order_date} 이전까지 발주 절차를 진행하시는 것이 가장 적합합니다.",
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
            "formula_1": "적정 권장 수량 = 고장 예상 수량 + 안전 재고 (갑작스러운 고장이나 수급 불안정 시에도 수업 중단 없이 운영 가능한 최소 물량)",
            "formula_2": "발주 시점(ROP) = (월 별 평균 수요량 X 리드 타임) + 안전 재고",
            "formula_3": "잔여 수명(RUL): 장비의 상태 기록과 부품별 내구연한을 딥러닝 모델로 분석하여 예측한 남은 가동 가능 시간"
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