import os
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 0. 설정 및 초기화
# ---------------------------------------------------------
fake = Faker('ko_KR')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data_lifecycle")
os.makedirs(SAVE_DIR, exist_ok=True)

# 시뮬레이션 설정
SIMULATION_START_YEAR = 2005
TODAY = datetime.now()

# 승인 상태 비율 설정 (확정 97%, 대기 2%, 반려 1%)
APPROVAL_RATIOS = [0.97, 0.02, 0.01]
APPROVAL_STATUSES = ['확정', '대기', '반려']

# ---------------------------------------------------------
# 1. 마스터 데이터 정의 (Master Data)
# ---------------------------------------------------------

# 1-0. 비고 템플릿 (품목별 용도)
REMARK_TEMPLATES_BY_CLASS = {
    # IT / 전산 장비
    "노트북컴퓨터": ["AI 실습 수업용 노트북", "전산 실습실 공용 장비", "교수·연구원 업무용", "학과 공용 전산 자산"],
    "데스크톱컴퓨터": ["전산 실습실 고정형 PC", "연구실 분석 업무용", "행정 업무용 데스크톱"],
    "액정모니터": ["전산 실습실 보조 모니터", "사무환경 개선용", "연구실 다중 화면 구성용"],
    "허브": ["전산망 확충용", "실습실 네트워크 구성용"],
    "라우터": ["실습실 네트워크 증설", "학과 전산망 고도화"],
    "하드디스크드라이브": ["연구 데이터 저장용", "서버 증설용 스토리지"],
    "플래시메모리저장장치": ["교육 자료 배포용", "백업 매체"],
    "스캐너": ["행정 문서 전산화", "자료 디지털 아카이빙"],
    "레이저프린터": ["행정 문서 출력용", "학과 공용 프린터"],
    
    # 가구 / 집기
    "책상": ["강의실 환경 개선", "연구실 집기 교체", "신규 연구실 구축"],
    "작업용의자": ["사무환경 개선", "노후 집기 교체"],
    "책걸상": ["강의실 집기 교체", "노후 책걸상 교체"],
    "서랍형수납장": ["연구실 문서 보관용", "행정 자료 수납용"],
    
    # 교육 기자재
    "칠판보조장": ["강의실 기자재 보강", "노후 기자재 교체"],
    "인터랙티브화이트보드": ["스마트 강의실 구축", "디지털 강의 환경 개선"],
    
    # 신규 품목
    "다기능복사기": ["보안 문서 파기용", "사무실 비치용"],
    "디지털카메라": ["홍보팀 촬영 지원", "현장 기록용", "행사 기록용"],
    "공기청정기": ["사무실 환경 개선", "강의실 미세먼지 관리"],
}

# 1-1. G2B 품목 마스터 (18개 품목)
# 구조: [물품분류코드(8), 물품식별코드(8), 품목명, 분류명, 내용연수, 평균단가(원)]
G2B_MASTER_DATA = [
    # =========================
    # 전자·정보·통신·영상 (10)
    # =========================
    ("43211503", "24343967", "노트북컴퓨터", "노트북컴퓨터, Dell, (CN)Latitude 3520-5110H, Intel Core i5 1135G7(2.4GHz), 액세서리별도", 6, 1133000),
    ("43211503", "24510198", "노트북컴퓨터", "노트북컴퓨터, Lenovo, (CN)82JBS00300, Intel Celeron N5100(1.1GHz), 액세서리별도", 6, 555000),
    ("43211507", "24355228", "데스크톱컴퓨터", "데스크톱컴퓨터, Dell, (CN)OptiPlex 5090, Intel Core i5 10505(3.1GHz)", 5, 2627000),
    ("43211507", "24158946", "데스크톱컴퓨터", "데스크톱컴퓨터, 서버앤컴퓨터, DECA-N3802, Intel Core i3 10100(3.6GHz)", 5, 546000),
    ("43211902", "24407366", "액정모니터", "액정모니터, 엘지전자, 27MP500W, 68.6cm", 5, 513000),
    ("43212105", "23858386", "레이저프린터", "레이저프린터, HP, (JP)HP Color LaserJet Enterprise M856dn, A3/컬러56/흑백56ppm", 6, 3465000),
    ("43211711", "24204348", "스캐너", "스캐너, Kodak alaris, (CN)S3100F, 600dpi", 6, 5500000),
    ("43222609", "23908131", "네트워크라우터", "43222609 네트워크라우터", 9, 542000),
    ("43201803", "23809899", "하드디스크드라이브", "하드디스크드라이브, Hitachi vantara, (US)R2H-H10RSS, 10TB", 7, 5340000),
    ("43223308", "22060848", "네트워크시스템장비용랙", "네트워크시스템장비용랙, 600×2000×750mm", 10, 891700),
    
    # =========================
    # 사무·교육·가구 (5)
    # =========================
    ("56101703", "25114372", "책상", "책상, 우드림, WD-WIZDE100, 2700×2150×750mm, 1인용", 9, 6000000),
    ("56112102", "24128496", "작업용의자", "작업용의자, 오피스안건사, AC-051, 513×520×783mm", 8, 93000),
    ("56112108", "24917370", "책상용콤비의자", "책걸상, 애니체, AMD-WT100A, 617×790×850mm", 10, 465000),
    ("56121798", "25616834", "칠판보조장", "칠판보조장, 우드림, WR-BSC7040, 7000×300×3000mm", 7, 10500000),
    ("44111911", "25460962", "인터랙티브화이트보드및액세서리", "인터랙티브화이트보드, 미래디스플레이, MDI86110, 279.4cm, IR센서/손/도구/LED", 7, 24200000),

    # =========================
    # [NEW] 소형/기타 전자기기 (데이터 희석용)
    # =========================
    ("45121504", "25468676", "디지털카메라", "디지털카메라, Nikon, (TH)Z6 III, 2450만화소", 8, 2980000),
    ("44101503", "25652906", "다기능복사기", "다기능복사기, Brother, (PH)DCP-T830DW, A4/흑백17/컬러16.5ipm", 8, 450000),
    ("40161602", "25676461", "공기청정기", "공기청정기, 엘지전자, AS235DWSP, 74.7㎡, 51W", 9, 840000),
]

# [NEW] 특수 목적 물품 (별도 로직 적용)
SPECIAL_ITEM_SERVER = ("43232902", "25461942", "통신서버소프트웨어", "통신소프트웨어, 세인트로그, SMART-CM V1.5, 통합방송솔루션, 1~4Core(Server)", 6, 60000000)

# 1-2. 부서 마스터 (부서별 규모 Scale 추가)
# (부서코드, 부서명, 규모_가중치) -> 가중치가 높으면 물품을 많이 가짐
DEPT_MASTER_DATA = [
    ("C354", "소프트웨어융합대학RC행정팀(ERICA)", 1.8), # 큼 (SW 중심)
    ("C352", "공학대학RC행정팀(ERICA)", 1.6),         # 큼 (공대)
    ("C364", "경상대학RC행정팀(ERICA)", 1.2),         # 보통
    ("C360", "글로벌문화통상대학RC행정팀(ERICA)", 1.2), # 보통
    ("A351", "시설팀(ERICA)", 1.0),                    # 작음 (시설 관리 위주)
    ("A320", "학생지원팀(ERICA)", 1.2),                 # 보통 (학생 복지)
]

# ---------------------------------------------------------
# 2. 로직: "수명 주기 기반(Lifecycle-based)" 데이터 생성
# ---------------------------------------------------------

def generate_acquisition_data_lifecycle():
    print(f"🚀 [Phase 1] 수명 주기 기반(Lifecycle) 데이터 생성을 시작합니다...")
    acquisition_list = []
    
    # 1. 부서별로 루프 (각 부서의 정원을 채우는 방식)
    for dept_code, dept_name, dept_scale in DEPT_MASTER_DATA:
        
        # 2. 각 물품(G2B Item)별 보유 정원(Quota) 결정
        for item_data in G2B_MASTER_DATA:
            class_code, id_code, item_name, model_name, life_years, base_price = item_data
            
            # (1) 품목별 총 보유 목표 수량(Total Quota) 결정 [수정됨]
            target_total_qty = 0
            
            # --- A. 핵심 IT 장비 (PC, 모니터) ---
            if item_name in ["노트북컴퓨터", "데스크톱컴퓨터", "액정모니터"]:
                # SW/공대는 실습실 수요로 인해 일반 행정팀보다 훨씬 많음
                multiplier = 1.5 if "소프트웨어" in dept_name or "공학" in dept_name else 0.6
                # (예: SW대학=45대, 학생팀=18대)
                target_total_qty = int(random.randint(20, 40) * dept_scale * multiplier)
                
            # --- B. 사무 주변기기 (프린터, 스캐너, 공기청정기 등) ---
            elif item_name in ["레이저프린터", "스캐너", "다기능복사기", "공기청정기", "세단기"]:
                # 부서 규모에 따라 2~5대 수준 보유
                target_total_qty = int(random.randint(2, 4) * dept_scale)
                
            # --- C. 네트워크/인프라 장비 (특수 부서용) ---
            elif item_name in ["네트워크라우터", "네트워크시스템장비용랙", "하드디스크드라이브", "허브", "플래시메모리저장장치"]:
                # 시설팀, SW, 공대 위주 보유 (나머지 부서는 0~1개)
                if "시설" in dept_name or "소프트웨어" in dept_name or "공학" in dept_name:
                    target_total_qty = int(random.randint(3, 8) * dept_scale)
                else:
                    target_total_qty = int(random.randint(0, 1))

            # --- D. 가구/강의실 비품 (대량) ---
            elif item_name in ["책상", "작업용의자", "책걸상"]:
                # 인원수 + 강의실/회의실 수요 (30~60개)
                target_total_qty = int(random.randint(30, 60) * dept_scale)

            # --- E. 고가/특수 교육 기자재 ---
            elif item_name in ["인터랙티브화이트보드", "칠판보조장"]:
                target_total_qty = int(random.randint(1, 3) * dept_scale)

            # --- F. 기타 (카메라 등) ---
            else: 
                target_total_qty = int(random.randint(0, 2) * dept_scale)

            # (2) 목표 수량을 채울 때까지 '구매 건(Batch)' 생성
            remaining_qty = target_total_qty
            
            while remaining_qty > 0:
                # A. 이번 구매 건의 수량(Batch Size) 결정 [수정됨]
                is_bulk_purchase = False 
                
                # 1) 대량 구매 가능 품목 (PC, 가구)
                if item_name in ["노트북컴퓨터", "데스크톱컴퓨터", "책상", "작업용의자", "책걸상"]:
                    # 30% 확률로 강의실/실습실 구축용 대량 구매 (10~20개)
                    if remaining_qty >= 10 and random.random() < 0.3:
                        batch_size = random.randint(10, 20)
                        is_bulk_purchase = True
                    else:
                        # 나머지는 개인 지급/소량 교체 (1~3개)
                        batch_size = random.randint(1, 3)
                        
                # 2) 네트워크/주변기기 (소량 묶음)
                elif item_name in ["네트워크라우터", "네트워크시스템장비용랙", "하드디스크드라이브"]:
                     # 인프라 구축 시 2~4개씩 살 수 있음
                     batch_size = random.randint(1, 4)
                     
                # 3) 그 외 단일 품목 (프린터, 카메라 등)
                else:
                    batch_size = 1

                # 남은 목표 수량보다 많이 살 순 없음
                if batch_size > remaining_qty:
                    batch_size = remaining_qty

                # B. 최초 도입 시점 결정 (2015 ~ 2019 분산)
                # 대량 구매끼리는 날짜가 겹치지 않게 연도 분산
                start_year = random.randint(SIMULATION_START_YEAR, 2019)
                start_month = random.randint(1, 12)
                start_day = random.randint(1, 28)
                current_date = datetime(start_year, start_month, start_day)

                # C. 생애주기 루프 (최초 구매 -> 수명 종료 후 교체 구매)
                while current_date < TODAY:
                    
                    # 1) 승인 상태
                    approval_status = np.random.choice(APPROVAL_STATUSES, p=APPROVAL_RATIOS)
                    
                    # 과거 데이터 대기 방지
                    if approval_status == '대기' and current_date < datetime(2024, 10, 1):
                        approval_status = '확정'
                    
                    # 반려 시뮬레이션
                    if approval_status == '반려':
                        _create_acquisition_row(acquisition_list, current_date, item_data, dept_code, dept_name, approval_status, batch_size, is_bulk_purchase)
                        current_date = current_date + timedelta(days=random.randint(14, 60))
                        approval_status = '확정'

                    # 2) 정리일자
                    clear_date_str = ""
                    if approval_status == '확정':
                        # 대량 구매는 검수 기간이 깁니다 (7~20일)
                        days_add = random.randint(7, 20) if is_bulk_purchase else random.randint(3, 7)
                        c_date = current_date + timedelta(days=days_add)
                        if c_date > TODAY: c_date = TODAY
                        clear_date_str = c_date.strftime('%Y-%m-%d')
                    
                    # 3) 데이터 생성 (batch_size 그대로 전달)
                    _create_acquisition_row(acquisition_list, current_date, item_data, dept_code, dept_name, approval_status, batch_size, is_bulk_purchase, clear_date_str)
                    
                    # 4) 다음 교체 시기 계산
                    # 내용연수 + 지연(0~2년)
                    usage_years = life_years + random.uniform(0, 2)
                    next_purchase_date = current_date + timedelta(days=int(usage_years * 365) + random.randint(-30, 30))
                    
                    current_date = next_purchase_date
                
                # 남은 목표 수량 차감
                remaining_qty -= batch_size
    
    # [NEW] 특수 물품(서버) 데이터 주입
    _inject_special_server_data(acquisition_list)

    return pd.DataFrame(acquisition_list)

# ---------------------------------------------------------
# 2. 로직: "수명 주기 기반(Lifecycle-based)" 데이터 생성 (배치 구매 적용)
# ---------------------------------------------------------

def generate_acquisition_data_lifecycle():
    print(f"🚀 [Phase 1] 수명 주기 기반(Lifecycle) 데이터 생성을 시작합니다...")
    acquisition_list = []
    
    # 1. 부서별로 루프
    for dept_code, dept_name, dept_scale in DEPT_MASTER_DATA:
        
        # 2. 각 물품(G2B Item)별 보유 정원(Quota) 결정
        for item_data in G2B_MASTER_DATA:
            class_code, id_code, item_name, model_name, life_years, base_price = item_data
            
            # (1) 품목별 총 보유 목표 수량(Total Quota) 결정
            target_total_qty = 0
            
            # --- IT 장비 ---
            if item_name in ["노트북컴퓨터", "데스크톱컴퓨터", "액정모니터"]:
                # SW/공대는 실습실이 있어 많음, 일반 행정팀은 직원 수만큼(10~15명)
                multiplier = 1.5 if "소프트웨어" in dept_name or "공학" in dept_name else 0.5
                target_total_qty = int(random.randint(20, 40) * dept_scale * multiplier)
                
            # --- 가구/비품 (대량) ---
            elif item_name in ["책상", "작업용의자", "책걸상"]:
                # 강의실 포함
                target_total_qty = int(random.randint(30, 60) * dept_scale)

            # --- 고가 장비 (소량) ---
            elif item_name in ["인터랙티브화이트보드", "서버", "네트워크시스템장비용랙"]:
                target_total_qty = int(random.randint(1, 3) * dept_scale)

            # --- 일반 비품 (적당량) ---
            else: 
                target_total_qty = int(random.randint(2, 5) * dept_scale)

            # (2) 목표 수량을 채울 때까지 '구매 건(Batch)' 생성
            # 핵심 변경: 1개씩 루프 돌지 않고, 덩어리(Batch)로 차감함
            remaining_qty = target_total_qty
            
            while remaining_qty > 0:
                # A. 이번 구매 건의 수량(Batch Size) 결정
                is_bulk_purchase = False # 대량 구매 여부 플래그
                
                # PC/책상류는 확률적으로 대량 구매 (강의실/실습실 구축)
                if item_name in ["노트북컴퓨터", "데스크톱컴퓨터", "책상", "작업용의자", "책걸상"]:
                    if random.random() < 0.4: # 40% 확률로 대량 구매 프로젝트
                        batch_size = random.randint(10, 20)
                        is_bulk_purchase = True
                    else:
                        batch_size = random.randint(1, 3) # 소량 구매 (개인 지급용)
                else:
                    batch_size = random.randint(1, 2) # 기타 장비는 소량

                # 남은 수량보다 많이 살 순 없음
                if batch_size > remaining_qty:
                    batch_size = remaining_qty
                
                # B. 최초 도입 시점 결정 (2015 ~ 2019 분산)
                # 대량 구매끼리는 날짜가 겹치지 않게 연도 분산
                start_year = random.randint(SIMULATION_START_YEAR, 2019)
                start_month = random.randint(1, 12)
                start_day = random.randint(1, 28)
                current_date = datetime(start_year, start_month, start_day)

                # C. 생애주기 루프 (최초 구매 -> 수명 종료 후 교체 구매)
                while current_date < TODAY:
                    
                    # 1) 승인 상태
                    approval_status = np.random.choice(APPROVAL_STATUSES, p=APPROVAL_RATIOS)
                    
                    # 과거 데이터 대기 방지
                    if approval_status == '대기' and current_date < datetime(2024, 10, 1):
                        approval_status = '확정'
                    
                    # 반려 시뮬레이션
                    if approval_status == '반려':
                        _create_acquisition_row(acquisition_list, current_date, item_data, dept_code, dept_name, approval_status, batch_size, is_bulk_purchase)
                        current_date = current_date + timedelta(days=random.randint(14, 60))
                        approval_status = '확정'

                    # 2) 정리일자
                    clear_date_str = ""
                    if approval_status == '확정':
                        # 대량 구매는 검수 기간이 깁니다 (7~20일)
                        days_add = random.randint(7, 20) if is_bulk_purchase else random.randint(3, 7)
                        c_date = current_date + timedelta(days=days_add)
                        if c_date > TODAY: c_date = TODAY
                        clear_date_str = c_date.strftime('%Y-%m-%d')
                    
                    # 3) 데이터 생성 (batch_size 그대로 전달)
                    _create_acquisition_row(acquisition_list, current_date, item_data, dept_code, dept_name, approval_status, batch_size, is_bulk_purchase, clear_date_str)
                    
                    # 4) 다음 교체 시기 계산
                    # 내용연수 + 지연(0~2년)
                    usage_years = life_years + random.uniform(0, 2)
                    next_purchase_date = current_date + timedelta(days=int(usage_years * 365) + random.randint(-30, 30))
                    
                    current_date = next_purchase_date
                
                # 남은 목표 수량 차감
                remaining_qty -= batch_size
    
    # [NEW] 특수 물품(서버) 데이터 주입
    _inject_special_server_data(acquisition_list)

    return pd.DataFrame(acquisition_list)

def _create_acquisition_row(data_list, date_obj, item_data, dept_code, dept_name, approval_status, quantity, is_bulk, clear_date_str=""):
    """단일 취득 데이터 행을 생성 (수량은 외부에서 결정된 값을 사용)"""
    class_code, id_code, item_name, model_name, life_years, base_price = item_data
    
    # 1) 금액 계산 (수량 * 단가)
    # 2015년 대비 물가 상승 반영
    years_passed = date_obj.year - 2015
    inflation_rate = 1.0 + (0.015 * years_passed)
    
    # 대량 구매(10개 이상) 시 단가 할인 (5%)
    bulk_discount = 0.95 if quantity >= 10 else 1.0

    final_unit_price = int(base_price * inflation_rate * bulk_discount * random.uniform(0.95, 1.05))
    final_unit_price = (final_unit_price // 1000) * 1000 
    
    total_amount = final_unit_price * quantity

    # 2) 비고 생성
    remark = ""
    if approval_status == '반려':
        remark = random.choice(["예산 초과", "규격 불일치", "재고 활용 권고", "사업 타당성 재검토"])
    else:
        # 대량 구매인 경우 비고를 그럴싸하게 작성
        if is_bulk:
            if "컴퓨터" in item_name or "모니터" in item_name:
                places = ["제1실습실", "제2실습실", "AI센터", "SW교육실", "종합설계실"]
                remark = f"{random.choice(places)} 환경개선 기자재 확충"
            elif "책상" in item_name or "의자" in item_name:
                remark = "노후 강의실 집기 일괄 교체"
            else:
                remark = "학과 공용 기자재 확충"
        else:
            # 소량 구매는 랜덤 템플릿
            if random.random() < 0.3:
                key = item_name
                if key not in REMARK_TEMPLATES_BY_CLASS:
                    if "컴퓨터" in key: key = "데스크톱컴퓨터"
                    elif "의자" in key: key = "작업용의자"
                
                candidates = REMARK_TEMPLATES_BY_CLASS.get(key, [])
                if candidates:
                    remark = random.choice(candidates)

    # 3) 취득 구분
    acq_method = np.random.choice(['자체구입', '자체제작', '기증'], p=[0.95, 0.02, 0.03])

    row = {
        'G2B_목록번호': class_code + id_code,
        'G2B_목록명': item_name,
        '물품분류코드': class_code,
        '물품분류명': item_name, 
        '물품식별코드': id_code,
        '물품품목명': model_name,
        '캠퍼스': 'ERICA',
        '취득일자': date_obj.strftime('%Y-%m-%d'),
        '취득금액': total_amount,
        '정리일자': clear_date_str,
        '운용부서': dept_name,
        '운용부서코드': dept_code,
        '운용상태': '취득',
        '내용연수': life_years,
        '수량': quantity,
        '승인상태': approval_status,
        '취득정리구분': acq_method,
        '비고': remark
    }
    data_list.append(row)

def _inject_special_server_data(data_list):
    """특수 물품(서버) 데이터를 별도로 주입"""
    print("⚡ [Phase 1] 특수 물품(서버) 데이터를 주입합니다...")
    
    sv_class_code, sv_id_code, sv_item_name, sv_model_name, sv_life, sv_price = SPECIAL_ITEM_SERVER
    sv_full_code = sv_class_code + sv_id_code

    # 서버 할당 시나리오 (시설팀 2대, 학생팀 1대)
    allocations = [
        ("A351", "시설팀(ERICA)", 2),
        ("A320", "학생지원팀(ERICA)", 1)
    ]

    for dept_code, dept_name, qty in allocations:
        for i in range(qty):
            # 서버는 드물게 도입 (2016~2018년 사이 1번 도입 가정)
            start_d = datetime(2016, 1, 1)
            end_d = datetime(2018, 12, 31)
            temp_date = fake.date_between(start_date=start_d, end_date=end_d)
            acq_date = datetime(temp_date.year, temp_date.month, temp_date.day)
            
            # 정리일자 (도입 기간 김)
            clear_date = acq_date + timedelta(days=random.randint(14, 45))
            
            row = {
                'G2B_목록번호': sv_full_code,
                'G2B_목록명': sv_item_name,
                '물품분류코드': sv_class_code,
                '물품분류명': sv_item_name,
                '물품식별코드': sv_id_code,
                '물품품목명': sv_model_name,
                '캠퍼스': 'ERICA',
                '취득일자': acq_date.strftime('%Y-%m-%d'),
                '취득금액': sv_price,
                '정리일자': clear_date.strftime('%Y-%m-%d'),
                '운용부서': dept_name,
                '운용부서코드': dept_code,
                '운용상태': '취득',
                '내용연수': sv_life,
                '수량': 1,
                '승인상태': '확정',
                '취득정리구분': '자체구입',
                '비고': f"{dept_name} 메인 서버 구축"
            }
            data_list.append(row)

# ---------------------------------------------------------
# 3. 실행 및 저장
# ---------------------------------------------------------

# 함수 호출하여 데이터프레임 생성
df_acquisition = generate_acquisition_data_lifecycle()

# [03-01] 물품 취득 대장 목록 (Main Output)
cols_acquisition = [
    'G2B_목록번호', 'G2B_목록명', '캠퍼스', '취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '수량', '승인상태', 
    '취득정리구분', '운용부서코드', '비고'
]
df_acquisition[cols_acquisition].to_csv(os.path.join(SAVE_DIR, '03_01_acquisition_master.csv'), index=False, encoding='utf-8-sig')

# [03-02] G2B 목록 조회용 (Popup Output)
df_class = df_acquisition[['물품분류코드', '물품분류명']].drop_duplicates()
df_class.to_csv(os.path.join(SAVE_DIR, '03_02_g2b_class_list.csv'), index=False, encoding='utf-8-sig')

df_item = df_acquisition[['물품식별코드', '물품품목명', '물품분류코드']].drop_duplicates()
df_item.to_csv(os.path.join(SAVE_DIR, '03_02_g2b_item_list.csv'), index=False, encoding='utf-8-sig')

print("✅ [Phase 1] 수명 주기 기반 데이터 생성 완료!")
print(f"   - 총 {len(df_acquisition)}건 생성됨 (시뮬레이션 기반)")
print(f"   - 데이터 기간: {df_acquisition['취득일자'].min()} ~ {df_acquisition['취득일자'].max()}")
print(f"   - 상위 품목 분포:\n{df_acquisition['G2B_목록명'].value_counts().head(7)}")