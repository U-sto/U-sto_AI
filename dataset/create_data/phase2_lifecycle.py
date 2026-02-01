import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker('ko_KR') 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_lifecycle") # create_data/data_lifecycle
os.makedirs(DATA_DIR, exist_ok=True)

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
PROB_RETURN_OVER_5Y = 0.15   # 5년 초과(내구연한) 반납 확률: 15%

# 반납 사유 확률 (사용연한, 고장, 불용, 사업, 잉여)
REASONS_RETURN = ['사용연한경과', '고장/파손', '불용결정', '사업종료', '잉여물품']
PROBS_RETURN_REASON = [0.6, 0.15, 0.1, 0.1, 0.05]

# 승인 상태 (확정, 대기, 반려)
STATUS_CHOICES = ['확정', '대기', '반려']
# 최근 대기 상태 몰림 기준일
RECENT_WAIT_START = datetime(2024, 10, 1)  # 2024-10 이후

# 각 단계별 승인 상태 확률
PROBS_STATUS_RETURN = [0.85, 0.1, 0.05] 
PROBS_STATUS_DISUSE = [0.70, 0.25, 0.05] 
PROBS_STATUS_DISPOSAL = [0.90, 0.08, 0.02]

PROB_SURPLUS_STORE = 0.9  # 잉여물품 보관 확률 (불용 스킵)

# 처분 방식 확률 (신품/중고품일 때 vs 아닐 때)
METHODS_DISPOSAL = ['매각', '폐기', '멸실', '도난']
PROBS_DISPOSAL_GOOD = [0.85, 0.13, 0.01, 0.01] # 상태 좋음
PROBS_DISPOSAL_BAD = [0.03, 0.95, 0.01, 0.01]  # 상태 나쁨 

MAX_REUSE_CYCLES = 3     # 최대 재사용 횟수 제한

PROB_SURPLUS_REUSE = 0.1
PROB_MANDATORY_DISUSE = 0.05  # 8년 이상 직권 불용 확률: 5%

# 기준일자 (오늘)
now = datetime.now()
TODAY = datetime(now.year, now.month, now.day)

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

def create_asset_ids(df: pd.DataFrame) -> pd.Series:
    """자산 고유번호 생성 로직 (M + 연도 + 시퀀스)"""
    acq_dates = pd.to_datetime(df['취득일자'])
    year_strs = acq_dates.dt.strftime('%Y')
    seq_strs = (
        pd.Series(np.arange(len(df)) + 1, index=df.index)
        .astype(str)
        .str.zfill(5)
    )
    return "M" + year_strs + seq_strs

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
        if min_allowed > TODAY: min_allowed = TODAY
        
        # [Fix] start_date와 end_date가 같을 때 Faker 오류 방지
        if min_allowed < TODAY:
            temp_date = fake.date_between(start_date=min_allowed, end_date=TODAY)
            req_date_final = datetime(temp_date.year, temp_date.month, temp_date.day)
        else:
            req_date_final = TODAY
            
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

def step_operation_req(ctx):
    """A. 운용 신청/재신청 단계"""
    # 컨텍스트에서 필요한 변수 추출
    sim_date = ctx['sim_cursor_date']
    asset_id = ctx['asset_id']
    row = ctx['row']
    
    # 운용신청일 생성
    op_req_date = sim_date + timedelta(days=random.randint(1, 14))
    if op_req_date > TODAY: return False # 미래 시점이면 종료

    # 승인 상태 및 날짜 계산
    # event_type 명시
    status, confirm_date, req_date_fixed = get_approval_status_and_date(op_req_date, event_type='op_req',is_op_req=True)
    
    # 재사용 차수 명시
    if ctx['need_initial_req']:
        req_type = '신규운용'
    else:
        # 재사용 시 재사용 차수를 증가시키고 컨텍스트에 저장
        reuse_cnt = ctx.get('reuse_count', 0) + 1
        ctx['reuse_count'] = reuse_cnt
        req_type = f'재사용({reuse_cnt}회차)' if reuse_cnt > 0 else '재사용'
    
    results['req'].append({
        '운용신청일자': req_date_fixed.strftime('%Y-%m-%d'),
        '등록일자': req_date_fixed.strftime('%Y-%m-%d'),
        '운용확정일자': confirm_date.strftime('%Y-%m-%d') if status == '확정' else '',
        '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
        '승인상태': status,
        'G2B_목록번호': row.G2B_목록번호, 'G2B_목록명': row.G2B_목록명,
        '물품고유번호': asset_id, 
        '취득일자': row.취득일자, '취득금액': row.취득금액,
        '운용부서': ctx['curr_dept_name'], '사용자': row.비고, '신청구분': req_type
    })
    
    ctx['need_initial_req'] = False # 다음부터는 재사용

    if status != '확정': return False # 확정 안되면 시뮬레이션 중단

    # 상태 업데이트
    use_start_date = confirm_date
    ctx['sim_cursor_date'] = use_start_date
    ctx['prev_status'] = ctx['curr_status']
    ctx['curr_status'] = '운용'
    
    # 운용대장 업데이트 (메모리 상)
    df_operation.at[ctx['idx'], '운용상태'] = '운용'
    df_operation.at[ctx['idx'], '운용부서'] = ctx['curr_dept_name']
    df_operation.at[ctx['idx'], '운용부서코드'] = ctx['curr_dept_code']
    
    if ctx['loop_count'] == 1:
        df_operation.at[ctx['idx'], '출력상태'] = np.random.choice(['출력', '미출력'], p=PROBS_PRINT_STATUS)

    # 이력 추가
    add_history(asset_id, confirm_date.strftime('%Y-%m-%d'), ctx['prev_status'], '운용', f'{req_type} 승인 및 사용 시작')
    
    return True

def step_determine_event(ctx):
    """B. 운용 중 사건 발생 결정"""
    sim_date = ctx['sim_cursor_date']
    df_operation = ctx['df_operation']
    acq_date = pd.to_datetime(ctx['row'].취득일자)
    use_start_date = pd.to_datetime(df_operation.at[ctx['idx'], '운용확정일자']) if '운용확정일자' in df_operation.columns and pd.notna(df_operation.at[ctx['idx'], '운용확정일자']) else sim_date
    
    age_days = (TODAY - acq_date).days
    days_since_use = (TODAY - use_start_date).days
    
    next_event = '유지'
    event_date = TODAY + timedelta(days=1)
    is_early = False

    # 1. 조기 반납 (1%)
    if random.random() < PROB_EARLY_RETURN:
        early_date = sim_date + timedelta(days=random.randint(1, 30))
        event_date = early_date
        next_event = '반납'
        is_early = True

    # 2. 일반/노후 반납
    if next_event == '유지' and age_days > (365 * 3):
        prob = PROB_RETURN_OVER_5Y if age_days > (365 * 5) else PROB_RETURN_OVER_3Y
        if random.random() < prob:
            # 30일 이상 사용 조건
            if days_since_use >= 30:
                calc_date = sim_date + timedelta(days=random.randint(30, 365))
                if calc_date > sim_date:
                    event_date = calc_date
                    next_event = '반납'
                    is_early = False
            
    # 3. 직권 불용 (8년 이상, 5%)
    if next_event == '유지' and age_days > (365 * 8):
        if random.random() < PROB_MANDATORY_DISUSE:
            event_date = sim_date + timedelta(days=random.randint(30, 90))
            next_event = '직권불용'

    if event_date > TODAY:
        # 미래 사건은 유지로 처리 (커서는 이동하지 않음)
        return '유지', event_date, False
    # 실제 사건이 발생하지 않은 경우, 시뮬레이션 커서를 이동하지 않는다.
    if next_event == '유지':
        return next_event, event_date, is_early

    ctx['sim_cursor_date'] = event_date
    return next_event, event_date, is_early

def step_process_return(ctx, event_date, is_early):
    """C-1. 반납 처리 및 재사용 여부 결정"""
    # 사유 및 물품상태 결정
    if is_early:
        reason = '잉여물품'
        condition = '신품'
    else:
        # 일반 반납 (잉여물품 제외한 나머지 사유 중 선택)
        late_reasons = ['사용연한경과', '고장/파손', '불용결정', '사업종료']
        late_probs = [0.5, 0.3, 0.1, 0.1]
        reason = np.random.choice(late_reasons, p=late_probs)
        
        if reason == '고장/파손': condition = '정비필요품'
        elif reason == '사용연한경과': condition = '폐품'
        else: condition = '중고품'
    
    ctx['curr_condition'] = condition

    # 승인 처리
    status, confirm_date, req_date = get_approval_status_and_date(
        event_date,
        PROBS_STATUS_RETURN,
        event_type='return'
    )
    confirm_str = confirm_date.strftime('%Y-%m-%d') if status == '확정' else ''

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
        '운용부서': ctx['curr_dept_name'], '운용상태': '운용', 
        '물품상태': condition, '사유': reason
    })

    if status == '확정':
        # 대장 및 이력 업데이트
        df_operation.at[ctx['idx'], '운용상태'] = '반납'
        df_operation.at[ctx['idx'], '운용부서'] = ''
        add_history(ctx['asset_id'], confirm_str, '운용', '반납', reason)
        
        ctx['sim_cursor_date'] = confirm_date
        
        # 재사용 여부 결정 (신품 & 10% 확률)
        if condition == '신품' and random.random() < PROB_SURPLUS_REUSE:
            # 부서 재배정
            new_dept = random.choice(DEPT_MASTER_DATA)
            ctx['curr_dept_code'] = new_dept[0]
            ctx['curr_dept_name'] = new_dept[1]
            return '재사용', reason
        else:
            return '불용진행', reason
            
    return '종료', reason

def step_process_disuse(ctx, trigger_event, inherited_reason):
    """C-2. 불용 및 처분 처리"""
    # 반납 사유 → 불용 사유 매핑
    DISUSE_REASON_MAP = {'잉여물품': '활용부서부재'}

    if trigger_event == '직권불용':
        reason = '직권 불용(파손/노후)'; condition = '폐품'; prev_stat = '운용'
    else:
        # 반납 사유 체크는 매핑 전에 수행
        condition = ctx['curr_condition']
        prev_stat = '반납'
        
        # 잉여물품 보관 스킵 로직을 '매핑 전'에 수행 
        # inherited_reason(반납사유)이 '잉여물품'인지 확인해야 함
        if inherited_reason == '잉여물품' and condition == '신품':
            if random.random() < PROB_SURPLUS_STORE: return # 스킵

        # 매핑 적용
        reason = DISUSE_REASON_MAP.get(inherited_reason, inherited_reason)

    # 불용 신청
    du_date = ctx['sim_cursor_date'] + timedelta(days=random.randint(1, 14))
    if du_date > TODAY: du_date = TODAY

    status, confirm_date, req_date = get_approval_status_and_date(
        du_date,
        PROBS_STATUS_DISUSE,
        event_type='disuse'
    )
    confirm_str = confirm_date.strftime('%Y-%m-%d') if status == '확정' else ''

    # 대장 업데이트
    if status == '확정':
        df_operation.at[ctx['idx'], '운용상태'] = '불용'
        add_history(ctx['asset_id'], confirm_str, prev_stat, '불용', reason, ADMIN_USER)
        ctx['sim_cursor_date'] = confirm_date

    # 불용 데이터 저장
    results['disuse'].append({
        '불용일자': req_date.strftime('%Y-%m-%d'),
        '불용확정일자': confirm_str,
        '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1],
        '승인상태': status,
        'G2B_목록번호': ctx['row'].G2B_목록번호, 'G2B_목록명': ctx['row'].G2B_목록명,
        '물품고유번호': ctx['asset_id'], 
        '취득일자': ctx['row'].취득일자, '취득금액': ctx['row'].취득금액,
        '정리일자': ctx['clear_date_str'],
        '운용부서': ctx['curr_dept_name'] if trigger_event == '직권불용' else '', 
        '운용상태' : prev_stat, 
        '내용연수': ctx['row'].내용연수,
        '물품상태': condition, '사유': reason
    })

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
        df_operation.at[ctx['idx'], '운용상태'] = '처분'
        add_history(ctx['asset_id'], confirm_str, '불용', '처분', f"{method} 완료", ADMIN_USER)

    results['disposal'].append({
        '처분일자': req_date.strftime('%Y-%m-%d'),
        '처분확정일자': confirm_str,
        '처분정리구분': method,
        '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1],
        '승인상태': status,
        'G2B_목록번호': ctx['row'].G2B_목록번호, 'G2B_목록명': ctx['row'].G2B_목록명,
        '물품고유번호': ctx['asset_id'], 
        '취득일자': ctx['row'].취득일자, '취득금액': ctx['row'].취득금액,
        '처분방식': method, '물품상태': condition, '사유': disuse_reason,
        '불용일자': ctx['sim_cursor_date'].strftime('%Y-%m-%d'),
        '내용연수': ctx['row'].내용연수, '정리일자': ctx['clear_date_str']
    })

# ---------------------------------------------------------
# 3. 메인 시뮬레이션 루프
# ---------------------------------------------------------

# 데이터 전처리 (Explosion & ID Generation)
print("⚙️ [Phase 2] 개별 자산 분화 및 고유번호 생성 중...")
df_confirmed = df_acq[df_acq['승인상태'] == '확정'].copy()
df_operation = df_confirmed.loc[df_confirmed.index.repeat(df_confirmed['수량'])].reset_index(drop=True)
df_operation['수량'] = 1
df_operation['물품고유번호'] = create_asset_ids(df_operation)
df_operation['운용상태'] = '취득'

print("⏳ [Phase 2] 자산 생애주기 시뮬레이션 시작 (운용 Loop)...")

# [Fix] 출력상태 컬럼 미리 초기화 (NaN 방지 - Review 반영)
# 기본값은 '미출력'으로 하거나, 아예 랜덤으로 미리 깔아두고 운용 확정 시 재설정하지 않도록 할 수도 있음.
# 여기서는 '미출력'을 기본으로 둠.
df_operation['출력상태'] = '미출력'

for row in df_operation.itertuples():
    # Context 객체: 함수 간 상태 공유용
    clear_date = pd.to_datetime(row.정리일자) if pd.notna(row.정리일자) else pd.to_datetime(row.취득일자)
    
    ctx = {
        'idx': row.Index,
        'row': row,
        'asset_id': row.물품고유번호,
        'sim_cursor_date': clear_date,
        'clear_date_str': clear_date.strftime('%Y-%m-%d'),
        'curr_dept_name': row.운용부서,
        'curr_dept_code': row.운용부서코드,
        'curr_status': '취득',
        'prev_status': '-',
        'curr_condition': '신품',
        'need_initial_req': True,
        'loop_count': 0
    }

    # 1. 취득 이력 생성
    add_history(ctx['asset_id'], ctx['clear_date_str'], '-', '취득', '신규 취득')

    # 2. Lifecycle Loop (운용 -> 반납 -> 재사용/불용 -> 처분)
    while ctx['loop_count'] <  MAX_REUSE_CYCLES:
        ctx['loop_count'] += 1

        # A. 운용 신청
        if not step_operation_req(ctx):
            break # 신청 안되거나 승인 안되면 종료
        
        # 운용 신청이 정상적으로 이루어진 경우에만 루프 카운트 증가
        ctx['loop_count'] += 1

        # B. 이벤트 결정 (유지, 반납, 직권불용)
        event_type, event_date, is_early = step_determine_event(ctx)

        if event_type == '유지':
            break

        # C-1. 반납 처리
        elif event_type == '반납':
            result_action, reason = step_process_return(ctx, event_date, is_early)
            
            if result_action == '재사용':
                # 재사용 시, 다음 루프의 이력 생성을 위해 현재 상태를 '반납'으로 명시
                ctx['curr_status'] = '반납'
                ctx['prev_status'] = '반납'
                continue # 루프 처음으로 (운용신청 다시 함)
            elif result_action == '불용진행':
                ctx['curr_status'] = '반납'
                step_process_disuse(ctx, '불용진행', reason)
                break # 불용으로 가면 운용 루프는 끝
            else:
                break # 종료

        # C-2. 직권 불용 처리
        elif event_type == '직권불용':
            ctx['sim_cursor_date'] = event_date
            step_process_disuse(ctx, '직권불용', '')
            break

# ---------------------------------------------------------
# 4. 파일 저장
# ---------------------------------------------------------
print("💾 [Phase 2] 결과 저장 중...")

df_op_req = pd.DataFrame(results['req'])
df_return = pd.DataFrame(results['return'])
df_disuse = pd.DataFrame(results['disuse'])
df_disposal = pd.DataFrame(results['disposal'])
df_history = pd.DataFrame(results['history'])

cols_operation = [
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '캠퍼스','취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '출력상태', '승인상태', '취득정리구분', '운용부서코드', '비고'
]

# 누락 컬럼 보정
if '비고' not in df_operation.columns:
    add_info = df_acq[['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '비고', '승인상태']].drop_duplicates()
    df_operation = df_operation.merge(
        add_info,
        on=['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '승인상태'],
        how='left'
    )

df_operation[cols_operation].to_csv(os.path.join(DATA_DIR, '04_01_operation_master.csv'), index=False, encoding='utf-8-sig')

if not df_op_req.empty: df_op_req.to_csv(os.path.join(DATA_DIR, '04_02_operation_req_list.csv'), index=False, encoding='utf-8-sig')
if not df_return.empty: df_return.to_csv(os.path.join(DATA_DIR, '04_03_return_list.csv'), index=False, encoding='utf-8-sig')
if not df_disuse.empty: df_disuse.to_csv(os.path.join(DATA_DIR, '05_01_disuse_list.csv'), index=False, encoding='utf-8-sig')
if not df_disposal.empty: df_disposal.to_csv(os.path.join(DATA_DIR, '06_01_disposal_list.csv'), index=False, encoding='utf-8-sig')
df_history.to_csv(os.path.join(DATA_DIR, '99_asset_status_history.csv'), index=False, encoding='utf-8-sig')

print("🎉 [Phase 2] 생애주기 시뮬레이션 및 파일 생성 완료!")
print(f"   - 운용 자산: {len(df_operation)}건")
print(f"   - 상태 이력: {len(df_history)}건")
if not df_history.empty:
    for status in ['취득', '운용', '반납', '불용', '처분']:
        print(f"      └ {status}: {len(df_history[df_history['(변경)운용상태'] == status])}건")