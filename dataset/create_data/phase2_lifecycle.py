import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

# [Professor Fix 1] 시드 및 날짜 고정
SEED_VAL = 42
random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
Faker.seed(SEED_VAL)
fake = Faker('ko_KR') 

# [Professor Fix 1] 기준일자 고정
FIXED_TODAY_STR = "2026-02-10"
TODAY = datetime.strptime(FIXED_TODAY_STR, "%Y-%m-%d")
now = TODAY # 코드 내 now 변수 호환용

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_lifecycle") # create_data/data_lifecycle
os.makedirs(DATA_DIR, exist_ok=True)
# ---------------------------------------------------------
# [NEW] 현실 기반 물품별 기대 수명 통계 (평균 μ, 표준편차 σ) - 단위: 년
# 출처: SquareTrade, ScienceDirect, Google Research, MS/OEM Guide 등
# ---------------------------------------------------------
REAL_LIFETIME_STATS = {
    # [IT 기기]
    "노트북": (4.3, 0.9),       # 보증사/연구 통계 기반
    "데스크톱": (5.0, 1.2),     # 기업 교체 주기 반영
    "모니터": (7.0, 1.5),       # LCD 패널 수명 고려
    "프린터": (6.0, 1.5),       # 레이저프린터 기준
    "스캐너": (6.5, 1.5),
    "라우터": (5.5, 1.5),       # 엔터프라이즈 장비 기준
    "하드디스크": (4.5, 1.2),   # HDD AFR 고려
    "서버": (6.0, 1.5),
    
    # [가구/시설]
    "랙": (15.0, 4.0),         # 철제 구조물
    "책상": (15.0, 3.5),       # 고품질 오피스 가구
    "실습대": (15.0, 3.5),
    "실험대": (15.0, 3.5),
    "보조장": (15.0, 3.5),
    "의자": (9.5, 2.0),        # 작업용/라운지 의자 평균
    "소파": (11.0, 3.0),
    "화이트보드": (7.0, 2.0),  # 인터랙티브(전자) 화이트보드
    
    # [기본값]
    "default": (8.0, 2.0)      # 매칭 안되는 품목용
}
# ---------------------------------------------------------
# 반납/불용 사유 그룹 정의
# ---------------------------------------------------------
# 1. 반납 사유 (행정적/업무적 요인)
REASONS_RETURN = ['사업종료', '잉여물품', '공용전환']
PROBS_RETURN_REASON = [0.6, 0.15, 0.25]

# 2. 불용 사유 (물리적/규정적 요인)
# - 수명(Normal Dist)이 다했을 때 선택될 사유들
REASONS_PHYSICAL_END = ['고장/파손', '노후화(성능저하)', '수리비용과다']
# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
try:
    file_path = os.path.join(DATA_DIR, '03_01_acquisition_master.csv')
    df_acq = pd.read_csv(file_path)
    print(f"📂 [Phase 2] 취득 데이터 로드 완료: {len(df_acq)}건")
except FileNotFoundError:
    print("❌ 오류: '03_01_acquisition_master.csv' 파일이 없습니다. Phase 1을 먼저 실행해주세요.")
    exit()

# 사용자/부서 마스터 
ADMIN_USER = ("hyl0610", "황팀장")
STAFF_USER = ("badbergjr", "박대리")

# Phase 1의 부서 마스터 데이터 정의 (재사용 시 부서 재배정용)
DEPT_MASTER_DATA = [
    ("C354", "소프트웨어융합대학RC행정팀(ERICA)"),
    ("C352", "공학대학RC행정팀(ERICA)"),
    ("C364", "경상대학RC행정팀(ERICA)"),
    ("C360", "글로벌문화통상대학RC행정팀(ERICA)"),
    ("A351", "시설팀(ERICA)"),
    ("A320", "학생지원팀(ERICA)"),
]

# ---------------------------------------------------------
# 시뮬레이션 확률 상수 정의 (Magic Numbers 제거)
# ---------------------------------------------------------
# 출력 상태 확률 (출력, 미출력)
PROBS_PRINT_STATUS = [0.8, 0.2]

# 반납 발생 확률
PROB_EARLY_RETURN = 0.01     # 초기 반납(신품, 잉여) 확률: 1%
PROB_RETURN_OVER_3Y = 0.05   # 3년 초과 반납 확률: 5%
PROB_RETURN_OVER_5Y = 0.15   # 5년 초과 반납 확률: 15%

# 승인 상태 (확정, 대기, 반려)
STATUS_CHOICES = ['확정', '대기', '반려']
# 최근 대기 상태 몰림 기준일
RECENT_WAIT_START = datetime(2024, 10, 1)  # 2024-10 이후

# 각 단계별 승인 상태 확률
PROBS_STATUS_RETURN = [0.85, 0.1, 0.05] 
PROBS_STATUS_DISUSE = [0.70, 0.25, 0.05] 
PROBS_STATUS_DISPOSAL = [0.93, 0.06, 0.01]

PROB_SURPLUS_STORE = 0.9  # 잉여물품 보관 확률 (불용 스킵)

# [NEW] 재사용 및 운용전환 관련 상수
PROB_REUSE_FROM_RETURN = 0.05   # 반납품 중 재사용(운용전환)될 확률 (5% - 아주 소수)
PROB_DIRECT_TRANSFER = 0.02     # 운용 중인 물품이 다른 부서로 바로 전환될 확률 (2% - 아주 소수)
RECENT_USE_LIMIT_DAYS = 365 * 2 # "사용한 지 얼마 안 된" 기준 (2년 이내)

# 처분 방식 확률 (신품/중고품일 때 vs 아닐 때)
METHODS_DISPOSAL = ['매각', '폐기', '멸실', '도난']
PROBS_DISPOSAL_GOOD = [0.85, 0.13, 0.01, 0.01] # 상태 좋음
PROBS_DISPOSAL_BAD = [0.03, 0.95, 0.01, 0.01]  # 상태 나쁨 

MAX_REUSE_CYCLES = 3     # 최대 재사용 횟수 제한

# ---------------------------------------------------------
# 1. 헬퍼 함수 & 데이터 구조 초기화
# ---------------------------------------------------------
# 결과 저장을 위한 전역 리스트
results = {
    'req': [],      # 운용신청
    'return': [],   # 반납
    'disuse': [],   # 불용
    'disposal': [], # 처분
    'history': []   # 이력
}
# [Professor Fix 3 + User Requirement] 자산 ID 생성 방식 개선 (하이브리드)
# 기존 포맷(M+연도+시퀀스)을 유지하되, 정렬 기준을 고정하여 재현성 확보
def create_asset_ids(df: pd.DataFrame) -> pd.Series:
    """
    형식: M{연도(4)}{시퀀스(5)} -> 예: M202400001
    개선점: 모든 컬럼 정보(Tie-Breaker)를 포함한 안정 정렬을 사용하여
            입력 순서가 바뀌어도 ID 부여 결과가 항상 동일하도록 보장
    """
    # 원본 인덱스 보존
    df_temp = df.copy()
    
    # 1. 연도 추출
    df_temp['temp_year'] = pd.to_datetime(df_temp['취득일자']).dt.year
    
    # 2. [Tie-Breaker] 동률 처리를 위한 '행 내용 기반' 정렬 키 생성
    #    모든 컬럼 값을 문자열로 이어붙여서, 내용이 조금이라도 다르면 순서가 고정되게 함
    df_temp['row_content_hash'] = df_temp.astype(str).sum(axis=1)
    
    # 3. [핵심] 완전 결정적 정렬 (Deterministic Sort)
    # 정렬 키: 연도 -> 부서 -> 품목 -> 금액 -> 날짜 -> 비고 -> (Tie-Breaker)내용해시
    sort_cols = ['temp_year', '운용부서코드', 'G2B_목록번호', '취득금액', '취득일자', '비고', 'row_content_hash']
    
    # 존재하는 컬럼만 필터링 (안전장치)
    valid_sort_cols = [col for col in sort_cols if col in df_temp.columns]
    
    df_temp = df_temp.sort_values(
        by=valid_sort_cols,
        ascending=[True] * len(valid_sort_cols),
        kind='mergesort' # [Review Fix] Stable Sort 사용 (동률 시 순서 유지 보장)
    )
    
    # 4. 연도별 그룹핑 후 시퀀스 생성 (1, 2, 3...)
    df_temp['temp_seq'] = df_temp.groupby('temp_year').cumcount() + 1
    
    # 5. ID 조합 (M + 2024 + 00001)
    df_temp['asset_id'] = (
        'M' + 
        df_temp['temp_year'].astype(str) + 
        df_temp['temp_seq'].astype(str).str.zfill(5)
    )
    
    # 6. 원래 순서대로 정렬하여 ID 시리즈 반환
    return df_temp['asset_id'].sort_index()

def add_history(asset_id, date_str, prev_stat, curr_stat, reason, user_tuple=STAFF_USER):
    """이력 추가 헬퍼 함수"""
    results['history'].append({
        '물품고유번호': asset_id,
        '변경일자': date_str,
        '(이전)운용상태': prev_stat,
        '(변경)운용상태': curr_stat,
        '사유': reason,
        '관리자명': user_tuple[1], '관리자ID': user_tuple[0],
        '등록자명': user_tuple[1], '등록자ID': user_tuple[0]
    })

def get_approval_status_and_date(base_date, prob_dist=None, event_type=None, is_op_req=False):
    """
    승인 상태 및 확정일자 결정
    :param base_date: 기준일자
    :param prob_dist: 승인 상태 선택에 사용할 확률 분포 (STATUS_CHOICES 순서의 리스트 또는 배열)
    :param event_type: 'op_req', 'return', 'disuse', 'disposal' 등 이벤트 종류
    :param is_op_req: 운용 신청 여부(True인 경우 운용 신청 전용 승인 로직 사용)
    :return: (status, confirm_date, req_date) 튜플. status는 승인 상태 문자열,
            confirm_date는 실제 승인/처리일자, req_date는 신청/요청일자(대기 상태일 경우 확인일자)
    """
    # 상태 결정
    if is_op_req:
        # 운용 신청의 경우 날짜에 따라 확률 다름
        days_diff = (TODAY - base_date).days
        if days_diff <= 14:
            status = np.random.choice(['확정', '대기', '반려'], p=[0.5, 0.4, 0.1])
        else:
            status = np.random.choice(['확정', '반려'], p=[0.99, 0.01])
    else:
        status = np.random.choice(STATUS_CHOICES, p=prob_dist)

    # 날짜 결정
    confirm_date = base_date
    req_date_final = base_date

    if status == '대기':
        min_allowed = max(base_date, RECENT_WAIT_START)

        # 시작일이 오늘보다 미래라면 오늘로 강제 조정
        if min_allowed > TODAY: min_allowed = TODAY
        
        # start_date와 end_date가 같은 경우(또는 역전) 방지
        if min_allowed >= TODAY:
            req_date_final = TODAY
        else:
            temp_date = fake.date_between(start_date=min_allowed, end_date=TODAY)
            req_date_final = datetime(temp_date.year, temp_date.month, temp_date.day)
            
        confirm_date = req_date_final 
        
    elif status == '확정':
        # [Fix] 이벤트 타입에 따라 처리 기간 차등 적용
        days_add = random.randint(3, 14)
        if event_type == 'disuse': days_add = random.randint(14, 30)
        if event_type == 'disposal': days_add = random.randint(30, 90)
        
        confirm_date = base_date + timedelta(days=days_add)
        if confirm_date > TODAY: confirm_date = TODAY
        
    return status, confirm_date, req_date_final

# ---------------------------------------------------------
# 2. 단계별 상세 처리 함수 (Refactoring)
# ---------------------------------------------------------

def step_operation_transfer(ctx, is_direct=False):
    """
    A. 운용 전환 신청 단계 (재사용 또는 직접전환)
    - 반납된 물품을 다른 부서가 사용하겠다고 신청하는 과정
    param is_direct: True면 운용 중 직접 전환, False면 반납 후 재사용
    """
    # sim_cursor_date는 기본적으로 '반납확정일자' 시점이나,
    # is_direct=True 인 경우에는 직접전환 이벤트 발생일(운용 중 전환일)을 의미함
    sim_date = ctx['sim_cursor_date']
    asset_id = ctx['asset_id']
    row = ctx['row']
    
    # 신청일 결정
    if is_direct:
        # 직접 전환은 이벤트 발생일이 곧 신청일
        op_req_date = sim_date
    else:
        # 반납 후 재사용은 반납확정일 + 1~7일 후
        op_req_date = sim_date + timedelta(days=random.randint(1, 7))
    
    if op_req_date > TODAY: return False # 미래 시점이면 종료

    # 승인 상태 및 날짜 계산 (운용전환은 대부분 확정됨)
    status, confirm_date, req_date_fixed = get_approval_status_and_date(op_req_date, event_type='op_req', is_op_req=True)
    
    # 재사용 차수 증가 (직접전환이든 재사용이든 횟수 차감하여 제한)
    reuse_cnt = ctx.get('reuse_count', 0) + 1
    ctx['reuse_count'] = reuse_cnt
    
    # 신청 구분 및 비고 멘트, 반려 시 상태 설정
    new_dept = ctx['curr_dept_name']
    if is_direct:
        req_type = '운용전환(직접)'
        transfer_remark = f"{new_dept}로 운용전환(직접인계) 신청"
        prev_stat_log = '운용'
        fail_fallback_status = '운용' # 반려되면 그냥 운용 상태 유지
    else:
        req_type = '운용전환(재사용)'
        transfer_remark = f"{new_dept}에서 운용전환(재사용) 신청(재사용 {reuse_cnt}회차)"
        prev_stat_log = '반납'
        fail_fallback_status = '반납' # 반려되면 반납 상태 유지

    # 승인 상태에 따른 표시 상태 (확정되면 '운용', 아니면 이전 상태인 '반납' 유지)
    if status == '확정':
        display_status = '운용'
    else:
        display_status = fail_fallback_status

    results['req'].append({
        '운용신청일자': req_date_fixed.strftime('%Y-%m-%d'),
        '등록일자': req_date_fixed.strftime('%Y-%m-%d'),
        '운용확정일자': confirm_date.strftime('%Y-%m-%d') if status == '확정' else '',
        '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
        '승인상태': status,
        'G2B_목록번호': row.G2B_목록번호, 'G2B_목록명': row.G2B_목록명,
        '물품고유번호': asset_id, 
        '취득일자': row.취득일자, '취득금액': row.취득금액,
        '운용부서': ctx['curr_dept_name'], 
        '사용자': transfer_remark, # 전환 신청 내용을 '사용자' 필드에 기입
        '신청구분': req_type,
        '운용상태': display_status
    })
    
    if status != '확정': return False # 확정 안되면 시뮬레이션 중단

    # 상태 업데이트
    use_start_date = confirm_date
    ctx['sim_cursor_date'] = use_start_date
    ctx['prev_status'] = prev_stat_log
    ctx['curr_status'] = '운용'
    
    # [Fix] 코드 리뷰 반영: 새로운 운용 시작일을 컨텍스트에 기록 (반납 시 사용 기간 계산용)
    ctx['last_operation_start_date'] = confirm_date

    # 운용대장 업데이트 (메모리 상)
    ctx['df_operation'].at[ctx['idx'], '운용상태'] = '운용'
    ctx['df_operation'].at[ctx['idx'], '운용부서'] = ctx['curr_dept_name']
    ctx['df_operation'].at[ctx['idx'], '운용부서코드'] = ctx['curr_dept_code']
    ctx['df_operation'].at[ctx['idx'], '운용확정일자'] = confirm_date.strftime('%Y-%m-%d')
    
    # 이력 추가
    add_history(asset_id, confirm_date.strftime('%Y-%m-%d'), prev_stat_log, '운용', f'{req_type} 승인 ({new_dept})')
    
    return True

def step_determine_event(ctx):
    """B. 운용 중 사건 발생 결정"""
    sim_date = ctx['sim_cursor_date']
    df_operation = ctx['df_operation']
    acq_date = pd.to_datetime(ctx['row'].취득일자)
    use_start_date = pd.to_datetime(df_operation.at[ctx['idx'], '운용확정일자']) if '운용확정일자' in df_operation.columns and pd.notna(df_operation.at[ctx['idx'], '운용확정일자']) else sim_date
    
    age_days = (TODAY - acq_date).days
    days_since_use = (TODAY - use_start_date).days

    event_date = TODAY + timedelta(days=1)

    # -----------------------------------------------------------
    # 1. [불용신청] 현실 수명 도달 -> 물리적 한계로 인한 사용자 불용 신청
    # -----------------------------------------------------------
    limit_real = ctx.get('assigned_limit_days', 365*5)
    
    if age_days >= limit_real:
        # 수명 도달 시점을 이벤트 발생일로 정확히 계산
        eol_date = acq_date + timedelta(days=limit_real)
        
        # 이벤트 날짜는 오늘을 넘을 수 없음
        calc_date = min(eol_date, TODAY)
        
        # 단, 시뮬레이션 커서(이전 단계 날짜)보다 과거일 수는 없음 (시간 역행 방지)
        if calc_date < sim_date:
            calc_date = sim_date
            
        return '불용신청', calc_date
    # -----------------------------------------------------------
    # 2. [직접전환] 운용 중 다른 부서로 이동 (반납 없이 바로 운용전환) - NEW
    # -----------------------------------------------------------
    # 조건: 사용 시작 후 최소 3개월은 지났고, 아직 수명은 넉넉할 때
    if days_since_use > 90 and age_days < (limit_real * 0.8):
        if random.random() < PROB_DIRECT_TRANSFER:
            transfer_date = sim_date + timedelta(days=random.randint(10, 180))
            if transfer_date <= TODAY:
                return '직접전환', transfer_date
            
    # -----------------------------------------------------------
    # 3. [반납] 업무적 사유(사업종료, 잉여 등)에 의한 랜덤 발생
    # -----------------------------------------------------------
    # 확률 체크 (기존 로직 활용)
    is_return_triggered = False
    
    # (1) 조기 반납 (1%)
    if random.random() < PROB_EARLY_RETURN:
        early_date = sim_date + timedelta(days=random.randint(1, 30))
        if early_date <= TODAY:
            event_date = early_date
            is_return_triggered = True

    # (2) 사용 기간에 따른 일반 반납 확률
    if not is_return_triggered and age_days > (365 * 3):
        prob = PROB_RETURN_OVER_5Y if age_days > (365 * 5) else PROB_RETURN_OVER_3Y
        if random.random() < prob:
            if days_since_use >= 30:
                calc_date = sim_date + timedelta(days=random.randint(30, 365))
                event_date = calc_date
                is_return_triggered = True

    if is_return_triggered:
        if event_date > TODAY:
            return '유지', event_date
        else:
            return '반납', event_date # 반납은 반납대로

    # 아무 일도 없으면 유지
    return '유지', event_date

def step_process_return(ctx, event_date):
    """
    C-1. 반납 처리 및 재사용 여부 결정
    """
    # 1. 반납 사유 결정
    reason = np.random.choice(REASONS_RETURN, p=PROBS_RETURN_REASON)
    
    # 2. 물품 상태 결정
    if reason == '잉여물품':
        condition = '신품'
    elif reason == '사업종료':
        condition = np.random.choice(['신품', '중고품','정비필요품'], p=[0.4, 0.5, 0.1])
    elif reason == '공용전환':
        condition = np.random.choice(['신품', '중고품'], p=[0.3, 0.7])
    
    ctx['curr_condition'] = condition

    # 3. 승인 처리
    status, confirm_date, req_date = get_approval_status_and_date(
        event_date,
        PROBS_STATUS_RETURN,
        event_type='return'
    )
    confirm_str = confirm_date.strftime('%Y-%m-%d') if status == '확정' else ''

    # 반납 리스트 저장 시, 확정 상태여야만 '반납'으로 표기, 아니면 기존 '운용' 유지
    display_status = '반납' if status == '확정' else '운용'

    # 반납 리스트 저장
    results['return'].append({
        '반납일자': req_date.strftime('%Y-%m-%d'),
        '반납확정일자': confirm_str,
        '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
        '승인상태': status,
        'G2B_목록번호': ctx['row'].G2B_목록번호, 'G2B_목록명': ctx['row'].G2B_목록명,
        '물품고유번호': ctx['asset_id'], 
        '취득일자': ctx['row'].취득일자,'취득금액': ctx['row'].취득금액,
        '정리일자': ctx['clear_date_str'], 
        '운용부서': ctx['curr_dept_name'], '운용상태': display_status,
        '물품상태': condition, '사유': reason
    })

    if status == '확정':
        # 대장 및 이력 업데이트
        ctx['df_operation'].at[ctx['idx'], '운용상태'] = '반납'
        ctx['df_operation'].at[ctx['idx'], '운용부서'] = ''
        ctx['prev_status'] = '운용'
        ctx['curr_status'] = '반납'
        add_history(ctx['asset_id'], confirm_str, '운용', '반납', reason)
        
        ctx['sim_cursor_date'] = confirm_date
        
        # 반납 후 처리 경로
        # 1. 재사용 (부서 재배정)
        # A. 신품이거나
        # B. 중고품인데 사용한지 얼마 안 된 것 (RECENT_USE_LIMIT_DAYS 이내)
        # 2. 불용 진행 (재활용 불가 판단 등)
        acq_dt = pd.to_datetime(ctx['row'].취득일자)
        # 최근 운용 시작일을 기준으로 사용 기간 계산 (없으면 취득일자 기준)
        operation_start_dt = ctx.get('last_operation_start_date')
        if isinstance(operation_start_dt, str):
            operation_start_dt = pd.to_datetime(operation_start_dt)
        base_dt = operation_start_dt if operation_start_dt is not None and not pd.isna(operation_start_dt) else acq_dt
        days_used = (confirm_date - base_dt).days
        is_recent_used = (condition == '중고품' and days_used <= RECENT_USE_LIMIT_DAYS)
        
        can_reuse = (condition == '신품') or is_recent_used
        
        # 후보군 중에서 '아주 소수'만 실제로 재사용 신청
        if can_reuse and random.random() < PROB_REUSE_FROM_RETURN:
            # 부서 변경
            new_dept = random.choice(DEPT_MASTER_DATA)
            ctx['curr_dept_code'] = new_dept[0]
            ctx['curr_dept_name'] = new_dept[1]
            return '재사용', reason
        else:
            # 재사용 안되면 불용 처리
            return '불용진행', reason
            
    return '종료', reason

def step_process_disuse(ctx, trigger_event, inherited_reason=None):
    """C-2. 불용 및 처분 처리"""
    # 1. 불용 사유 및 상태 결정    
    if trigger_event == '불용신청':
        # [NEW] 현실 수명이 다해서 오는 경우 -> 물리적 사유 선택
        reason = random.choice(REASONS_PHYSICAL_END)
        condition = '폐품' if reason in ['고장/파손'] else '불용품'
        prev_stat = '운용' # 반납 거치지 않고 바로 옴
        
    elif trigger_event == '불용진행':
        # 반납 후 불용으로 넘어오는 경우 (사유 상속 또는 매핑)
        # 사유: 활용부서 부재, 구형화 등
        if inherited_reason in ['잉여물품', '사업종료']:
            reason = np.random.choice(['활용부서부재', '구형화'], p =[0.7, 0.3])
        else:
            reason = inherited_reason # 공용전환 등
            
        condition = ctx['curr_condition']
        prev_stat = '반납'
        
        # 잉여물품 보관 스킵 로직 (확률적으로 불용 안하고 창고 보관 -> 시뮬 종료)
        if inherited_reason == '잉여물품' and condition == '신품':
             if random.random() < PROB_SURPLUS_STORE: return # 불용 기록 안하고 종료

    else:
        reason = '기타'
        condition = '불용품'
        prev_stat = '운용'
    
    # 2. 불용 승인 처리
    du_date = ctx['sim_cursor_date'] + timedelta(days=random.randint(1, 14))
    if du_date > TODAY: du_date = TODAY

    status, confirm_date, req_date = get_approval_status_and_date(
        du_date,
        PROBS_STATUS_DISUSE,
        event_type='disuse'
    )
    confirm_str = confirm_date.strftime('%Y-%m-%d') if status == '확정' else ''

    # 불용 리스트 저장 시, 확정 상태여야만 '불용'으로 표기, 아니면 기존 상태 유지
    if status == '확정':
        display_status = '불용'
    else:
        display_status = prev_stat

    # 불용 데이터 저장
    results['disuse'].append({
        '불용일자': req_date.strftime('%Y-%m-%d'),
        '불용확정일자': confirm_str,
        '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
        '승인상태': status,
        'G2B_목록번호': ctx['row'].G2B_목록번호, 'G2B_목록명': ctx['row'].G2B_목록명,
        '물품고유번호': ctx['asset_id'], 
        '취득일자': ctx['row'].취득일자, '취득금액': ctx['row'].취득금액,
        '정리일자': ctx['clear_date_str'],
        '운용부서': ctx['curr_dept_name'], 
        '운용상태' : display_status,
        '내용연수': ctx['row'].내용연수,
        '물품상태': condition, '사유': reason
    })

    # 대장 업데이트
    if status == '확정':
        ctx['df_operation'].at[ctx['idx'], '운용상태'] = '불용'
        add_history(ctx['asset_id'], confirm_str, prev_stat, '불용', reason, ADMIN_USER)
        ctx['sim_cursor_date'] = confirm_date


    # 처분 진행 (불용 확정시에만)
    if status == '확정':
        step_process_disposal(ctx, condition, reason)

def step_process_disposal(ctx, condition, disuse_reason):
    """C-3. 처분 처리"""
    dp_date = ctx['sim_cursor_date'] + timedelta(days=random.randint(1, 14))
    if dp_date > TODAY: dp_date = TODAY

    # 처분 방식
    probs = PROBS_DISPOSAL_GOOD if condition in ['신품', '중고품'] else PROBS_DISPOSAL_BAD
    method = np.random.choice(METHODS_DISPOSAL, p=probs)

    status, confirm_date, req_date = get_approval_status_and_date(
        dp_date,
        PROBS_STATUS_DISPOSAL,
        event_type='disposal'
    )
    confirm_str = confirm_date.strftime('%Y-%m-%d') if status == '확정' else ''

    if status == '확정':
        ctx['df_operation'].at[ctx['idx'], '운용상태'] = '처분'
        add_history(ctx['asset_id'], confirm_str, '불용', '처분', f"{method} 완료", ADMIN_USER)

    results['disposal'].append({
        '처분일자': req_date.strftime('%Y-%m-%d'),
        '처분확정일자': confirm_str,
        '처분정리구분': method,
        '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
        '승인상태': status,
        'G2B_목록번호': ctx['row'].G2B_목록번호, 'G2B_목록명': ctx['row'].G2B_목록명,
        '물품고유번호': ctx['asset_id'], 
        '취득일자': ctx['row'].취득일자, '취득금액': ctx['row'].취득금액,
        '처분방식': method, '물품상태': condition, '사유': disuse_reason,
        '불용일자': ctx['sim_cursor_date'].strftime('%Y-%m-%d'),
        '내용연수': ctx['row'].내용연수, '정리일자': ctx['clear_date_str'],
    })

# ---------------------------------------------------------
# 3. 메인 시뮬레이션 루프
# ---------------------------------------------------------

# 데이터 전처리 (Explosion & ID Generation)
print("⚙️ [Phase 2] 개별 자산 분화 및 고유번호 생성 중...")
df_confirmed = df_acq[df_acq['승인상태'] == '확정'].copy()
df_operation = df_confirmed.loc[df_confirmed.index.repeat(df_confirmed['수량'])].reset_index(drop=True)

df_operation['취득금액'] = (df_operation['취득금액'] / df_operation['수량']).fillna(0).astype('int64')
df_operation['수량'] = 1
df_operation['물품고유번호'] = create_asset_ids(df_operation)
# [수정] 초기 상태를 '운용'으로 설정 (취득 즉시 운용대장 등재)
df_operation['운용상태'] = '운용' 
# [수정] 최초 운용 등재 시 PROBS_PRINT_STATUS 확률로 출력상태 설정
df_operation['출력상태'] = np.random.choice(
    ['출력', '미출력'],
    size=len(df_operation),
    p=PROBS_PRINT_STATUS
)

# [수정] 초기 운용확정일자는 취득 정리일자와 동일하게 설정
df_operation['운용확정일자'] = df_operation['정리일자'].fillna(df_operation['취득일자'])

print("⏳ [Phase 2] 자산 생애주기 시뮬레이션 시작 (운용 Loop)...")

for row in df_operation.itertuples():
    # Context 객체: 함수 간 상태 공유용
    clear_date = pd.to_datetime(row.정리일자) if pd.notna(row.정리일자) else pd.to_datetime(row.취득일자)
    
    # ---------------------------------------------------------
    # [NEW] 1. 물품별 현실적 기대 수명(Natural Life Limit) 계산
    # 우선 기본값 설정
    mu, sigma = REAL_LIFETIME_STATS["default"]
    
    # 목록명이나 분류명에서 키워드 검색하여 통계 적용
    # 대소문자 무시 및 긴 키워드 우선 매칭 적용
    target_name = str(row.G2B_목록명)
    target_name_norm = target_name.strip().casefold()
    
    # 키워드 길이 역순 정렬 (구체적인 단어가 먼저 매칭되도록)
    sorted_keys = sorted(REAL_LIFETIME_STATS.keys(), key=len, reverse=True)
    
    for key in sorted_keys:
        if key == "default": continue # default는 루프 밖에서 처리하거나 마지막에
        
        # casefold()로 대소문자 무시 비교
        if key.casefold() in target_name_norm:
            mu, sigma = REAL_LIFETIME_STATS[key]
            break
            
    # [NEW] 2. 정규분포(Normal Distribution)에서 샘플링
    # - mu(평균)와 sigma(표준편차)를 이용해 랜덤 수명 생성
    # - 최소 1년(365일)은 사용한다고 가정 (음수 방지)
    assigned_life_years = max(1.0, np.random.normal(mu, sigma))
    
    # 일(Day) 단위로 변환
    assigned_limit_days = int(assigned_life_years * 365)

    ctx = {
        'idx': getattr(row, 'Index', 0), # 인덱스 안전하게 가져오기
        'row': row,
        'asset_id': row.물품고유번호,
        'sim_cursor_date': clear_date,
        'clear_date_str': clear_date.strftime('%Y-%m-%d'),
        'curr_dept_name': row.운용부서,
        'curr_dept_code': row.운용부서코드,

        # [수정] 초기 상태 '운용'으로 시작
        'curr_status': '운용', 
        'prev_status': '취득',

        'curr_condition': '신품',
        'reuse_count': 0,
        'df_operation': df_operation,
        'assigned_limit_days': assigned_limit_days,  # <--- 현실 수명 할당
    }
    # 1. 취득 이력 생성 (동일 일자 내에서 운용 이력보다 먼저 발생하도록 미세 시간차 부여)
    acq_dt = datetime.combine(clear_date.date(), datetime.min.time())
    op_dt = acq_dt + timedelta(seconds=1)
    acq_dt_str = acq_dt.strftime('%Y-%m-%d %H:%M:%S')
    op_dt_str = op_dt.strftime('%Y-%m-%d %H:%M:%S')
    add_history(ctx['asset_id'], acq_dt_str, '-', '취득', '신규 취득')

    # 2. 곧바로 운용 등재 (전산상 자동 전환)
    add_history(ctx['asset_id'], op_dt_str, '취득', '운용', '신규 운용 등재')
    # ==========================================================================
    # [NEW] 특수 물품(서버) 전용 로직 (시뮬레이션 루프 패스)
    # ==========================================================================
    if "통신서버" in row.G2B_목록명:
        # 1) 날짜 및 기본 정보 세팅
        acq_dt = pd.to_datetime(row.취득일자)
        op_start_date = ctx['sim_cursor_date'] + timedelta(days=random.randint(1, 7))
        if op_start_date > TODAY: op_start_date = TODAY

        # [Fix] 코드 리뷰 반영: 서버의 최초 운용 시작일 기록
        ctx['last_operation_start_date'] = op_start_date

        # 서버는 관리태그 부착 필수 (초기 랜덤값 무시하고 강제 설정)
        df_operation.at[ctx['idx'], '출력상태'] = '출력'

        # 2) 구형 서버 (2020년 이전) -> 운용하다가 불용/처분됨
        if acq_dt.year < 2020:
            # 내용연수 6년 + 알파 시점에 불용
            life_years = 6
            disuse_date = acq_dt + timedelta(days=365*life_years + random.randint(0, 60))
            
            # 불용 리스트 추가
            disuse_reason = '내구연한 경과(노후화)'
            results['disuse'].append({
                '불용일자': disuse_date.strftime('%Y-%m-%d'),
                '불용확정일자': disuse_date.strftime('%Y-%m-%d'),
                '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
                '승인상태': '확정',
                'G2B_목록번호': row.G2B_목록번호, 'G2B_목록명': row.G2B_목록명,
                '물품고유번호': ctx['asset_id'], 
                '취득일자': row.취득일자, '취득금액': row.취득금액,
                '정리일자': row.정리일자, '운용부서': row.운용부서, 
                '운용상태' : '운용', '내용연수': row.내용연수,
                '물품상태': '폐품', '사유': disuse_reason
            })
            
            # 대장 상태 변경
            df_operation.at[ctx['idx'], '운용상태'] = '불용'
            add_history(ctx['asset_id'], disuse_date.strftime('%Y-%m-%d'), '운용', '불용', disuse_reason, ADMIN_USER)

            # 처분 (매각)
            disposal_date = disuse_date + timedelta(days=random.randint(30, 90))
            if disposal_date > TODAY: disposal_date = TODAY # 미래 방지

            results['disposal'].append({
                '처분일자': disposal_date.strftime('%Y-%m-%d'),
                '처분확정일자': disposal_date.strftime('%Y-%m-%d'),
                '처분정리구분': '매각',
                '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
                '승인상태': '확정',
                'G2B_목록번호': row.G2B_목록번호, 'G2B_목록명': row.G2B_목록명,
                '물품고유번호': ctx['asset_id'], 
                '취득일자': row.취득일자, '취득금액': row.취득금액,
                '처분방식': '매각', '물품상태': '폐품', '사유': disuse_reason,
                '불용일자': disuse_date.strftime('%Y-%m-%d'),
                '내용연수': row.내용연수, '정리일자': row.정리일자
            })

            # 최종 상태 변경
            df_operation.at[ctx['idx'], '운용상태'] = '처분'
            add_history(ctx['asset_id'], disposal_date.strftime('%Y-%m-%d'), '불용', '처분', '매각 완료', ADMIN_USER)

        # 4) 신형 서버 (2020년 이후) -> 그냥 '운용' 상태 유지 (별도 코드 필요 없음)
        
        continue # [중요] 아래 while 루프(랜덤 시뮬레이션)를 건너뜀
    # ==========================================================================

    # 2. Lifecycle Loop (운용 -> 반납 -> 운용전환(재사용)/불용 -> 처분)
    while ctx['reuse_count'] <  MAX_REUSE_CYCLES:

        # A. 이벤트 결정 (유지, 반납, 불용신청, 직접전환)
        event_type, event_date = step_determine_event(ctx)

        if event_type == '유지':
            break
        
        # [NEW] B-0. 운용 중 직접 전환 (소수 케이스)
        elif event_type == '직접전환':
            ctx['sim_cursor_date'] = event_date
            # 부서 변경 (랜덤)
            new_dept = random.choice(DEPT_MASTER_DATA)
            ctx['curr_dept_code'] = new_dept[0]
            ctx['curr_dept_name'] = new_dept[1]
            
            # 직접 전환 신청 수행 (is_direct=True)
            if step_operation_transfer(ctx, is_direct=True):
                continue # 성공 시 루프 유지 (새 부서에서 운용 시작)
            else:
                break # 실패 시 종료

        # B-1. 반납 처리
        elif event_type == '반납':
            result_action, reason = step_process_return(ctx, event_date)
            
            if result_action == '재사용':
                # 재사용이 결정되면 -> 운용 전환 신청(Operation Transfer) 수행
                # 반납 후 재사용 신청 (is_direct=False)
                if step_operation_transfer(ctx, is_direct=False):
                    # 운용 전환 성공 시, 다시 루프 처음(운용 상태)으로 돌아가서 다음 이벤트 대기
                    continue 
                else:
                    break # 신청 반려 시 종료
            
            elif result_action == '불용진행':
                step_process_disuse(ctx, '불용진행', inherited_reason=reason)
                break # 불용으로 가면 운용 루프는 끝
            else:
                break # 종료

        # B-2. 물리적 수명 만료 (불용신청)
        elif event_type == '불용신청':
            ctx['sim_cursor_date'] = event_date
            step_process_disuse(ctx, '불용신청')
            break

# ---------------------------------------------------------
# 4. 파일 저장
# ---------------------------------------------------------
print("💾 [Phase 2] 결과 저장 중...")

# 각 CSV별 컬럼 정의 (빈 결과가 나와도 헤더를 유지하기 위함)
COLS_REQ = [
    '운용신청일자', '등록일자', '운용확정일자', '등록자ID', '등록자명', '승인상태',
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '운용부서', 
    '사용자', '신청구분', '운용상태'
]
COLS_RETURN = [
    '반납일자', '반납확정일자', '등록자ID', '등록자명', '승인상태', 'G2B_목록번호', 
    'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '정리일자', '운용부서', 
    '운용상태', '물품상태', '사유'
]
COLS_DISUSE = [
    '불용일자', '불용확정일자', '등록자ID', '등록자명', '승인상태', 'G2B_목록번호',
    'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '정리일자', '운용부서',
    '운용상태', '내용연수', '물품상태', '사유'
]
COLS_DISPOSAL = [
    '처분일자', '처분확정일자', '처분정리구분', '등록자ID', '등록자명', '승인상태',
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '처분방식',
    '물품상태', '사유', '불용일자', '내용연수', '정리일자'
]
COLS_HISTORY = [
    '물품고유번호', '변경일자', '(이전)운용상태', '(변경)운용상태', '사유', 
    '관리자명', '관리자ID', '등록자명', '등록자ID'
]

# 데이터프레임 생성 시 columns 명시
df_op_req = pd.DataFrame(results['req'], columns=COLS_REQ)
df_return = pd.DataFrame(results['return'], columns=COLS_RETURN)
df_disuse = pd.DataFrame(results['disuse'], columns=COLS_DISUSE)
df_disposal = pd.DataFrame(results['disposal'], columns=COLS_DISPOSAL)
df_history = pd.DataFrame(results['history'], columns=COLS_HISTORY)

cols_operation = [
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '캠퍼스','취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '출력상태', '승인상태', '취득정리구분', '운용부서코드', '비고', '운용확정일자'
]

# 1. 비고 등 원본 데이터 병합
if '비고' not in df_operation.columns:
    add_info = df_acq[['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '비고', '승인상태']].drop_duplicates()
    df_operation = df_operation.merge(
        add_info,
        on=['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '승인상태'],
        how='left'
    )

# 2. '운용확정일자' 컬럼이 없는 경우 생성 (KeyError 방지)
if '운용확정일자' not in df_operation.columns:
    # 시뮬레이션 루프에서 업데이트되지 않은 경우(예: 로직 타기 전)를 대비해 빈 값으로 생성
    # 하지만 보통 루프 내에서 업데이트 되므로, 여기서는 안전장치로 추가
    df_operation['운용확정일자'] = ''

df_operation[cols_operation].to_csv(os.path.join(DATA_DIR, '04_01_operation_master.csv'), index=False, encoding='utf-8-sig')

df_op_req.to_csv(os.path.join(DATA_DIR, '04_02_operation_req_list.csv'), index=False, encoding='utf-8-sig')
df_return.to_csv(os.path.join(DATA_DIR, '04_03_return_list.csv'), index=False, encoding='utf-8-sig')
df_disuse.to_csv(os.path.join(DATA_DIR, '05_01_disuse_list.csv'), index=False, encoding='utf-8-sig')
df_disposal.to_csv(os.path.join(DATA_DIR, '06_01_disposal_list.csv'), index=False, encoding='utf-8-sig')
df_history.to_csv(os.path.join(DATA_DIR, '99_asset_status_history.csv'), index=False, encoding='utf-8-sig')

print("🎉 [Phase 2] 생애주기 시뮬레이션 및 파일 생성 완료!")
print(f"   - 총 자산 규모: {len(df_operation)}건")

# [수정] 이력(History) 횟수가 아닌, 현재 대장(Operation) 상의 '최종 상태' 분포를 출력
print("\n📊 [현재 자산 상태 현황]")
current_status_counts = df_operation['운용상태'].value_counts()
for status, count in current_status_counts.items():
    print(f"      └ {status}: {count}건")

# 참고용으로 총 이력 건수만 한 줄로 표시
print(f"\n   (참고: 생성된 전체 상태 변경 이력 로그는 총 {len(df_history)}건 입니다.)")

# [NEW] 물품별 수량 통계 출력 (Phase 2 결과 기준)
print("\n📦 물품별 보유 수량 (상위 22개):")
print(df_operation['G2B_목록명'].value_counts().head(22))