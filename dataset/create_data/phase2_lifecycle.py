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
# Phase 1 결과물 로드
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
PROBS_PRINT_STATUS = [0.2, 0.8]

# 반납 발생 확률 (현실적으로 낮춤)
PROB_EARLY_RETURN = 0.01     # 초기 반납(신품, 잉여) 확률: 1%
PROB_RETURN_OVER_3Y = 0.05   # 3년 초과 반납 확률: 5%
PROB_RETURN_OVER_5Y = 0.15   # 5년 초과(내구연한) 반납 확률: 15%

# 반납 사유 확률 (사용연한, 고장, 불용, 사업, 잉여)
REASONS_RETURN = ['사용연한경과', '고장/파손', '불용결정', '사업종료', '잉여물품']
PROBS_RETURN_REASON = [0.6, 0.1, 0.05, 0.1, 0.05]

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

# ---------------------------------------------------------
# 1. 데이터 분화 (Explosion) & 고유번호 부여
# ---------------------------------------------------------
# 승인상태가 '확정'인 건만 운용 대장으로 넘어감
df_confirmed = df_acq[df_acq['승인상태'] == '확정'].copy()

# 행 복제 (수량 N -> N개 행)
df_operation = df_confirmed.loc[df_confirmed.index.repeat(df_confirmed['수량'])].reset_index(drop=True)
df_operation['수량'] = 1

def create_asset_ids(df: pd.DataFrame) -> pd.Series:
    acq_dates = pd.to_datetime(df['취득일자'])
    year_strs = acq_dates.dt.strftime('%Y')
    seq_strs = (
        pd.Series(np.arange(len(df)) + 1, index=df.index)
        .astype(str)
        .str.zfill(5)
    )
    return "M" + year_strs + seq_strs

print("⚙️ [Phase 2] 개별 자산 분화 및 고유번호 생성 중...")
df_operation['물품고유번호'] = create_asset_ids(df_operation)

# 초기 상태 설정
df_operation['운용상태'] = '취득'

# ---------------------------------------------------------
# 2. 생애주기 시뮬레이션 (Lifecycle Loop)
# ---------------------------------------------------------
operation_history_list = [] 
operation_req_list = []     
return_list = [] 
disuse_list = [] 
disposal_list = [] 

# 기준일자 (오늘)
now = datetime.now()
today = datetime(now.year, now.month, now.day)

print("⏳ [Phase 2] 자산 생애주기 시뮬레이션 시작 (운용 Loop)...")

for row in df_operation.itertuples():
    idx = row.Index 
    
    # -------------------------------------------------------
    # [초기 변수 세팅]
    # -------------------------------------------------------
    g2b_full_code = row.G2B_목록번호
    g2b_name = row.G2B_목록명
    asset_id = row.물품고유번호
    acq_date = pd.to_datetime(row.취득일자)
    total_amount = row.취득금액
    life_years = row.내용연수
    
    # 정리일자 처리
    if pd.isna(row.정리일자) or row.정리일자 == '':
        clear_date = acq_date 
        clear_date_str = ''
    else:
        clear_date = pd.to_datetime(row.정리일자)
        clear_date_str = clear_date.strftime('%Y-%m-%d')

    # 상태 추적 변수 (Loop 내에서 변화)
    current_dept_name = row.운용부서
    current_dept_code = row.운용부서코드
    current_status = '취득'
    current_condition = '신품'
    
    # 시뮬레이션 커서 날짜 (이 날짜를 기준으로 사건 발생)
    sim_cursor_date = clear_date 

    # -------------------------------------------------------
    # 2-0. 취득 기록
    # -------------------------------------------------------
    operation_history_list.append({
        '물품고유번호': asset_id,
        '변경일자': clear_date.strftime('%Y-%m-%d'),
        '(이전)운용상태': '-',
        '(변경)운용상태': '취득',
        '사유': '신규 취득',
        '관리자명': STAFF_USER[1], '관리자ID': STAFF_USER[0],
        '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
    })

    # -------------------------------------------------------
    # [Main Lifecycle Loop] 
    # 운용 -> (유지/반납/직권불용) -> (재사용/불용) -> 처분
    # -------------------------------------------------------
    active_flag = True
    loop_count = 0  
    need_initial_req = True # 첫 루프는 신규 운용 신청

    while active_flag and loop_count < 3:
        loop_count += 1
        
        # ===================================================
        # A. [운용 신청/재신청 단계]
        # ===================================================
        # 운용신청일 = 정리일자(또는 이전 사건일) + 1일~2주 사이
        op_req_date = sim_cursor_date + timedelta(days=random.randint(1, 14))
        
        if op_req_date > today: break
            
        op_req_date_str = op_req_date.strftime('%Y-%m-%d')
        
        # 승인 상태 결정 로직 (최신 대기 몰림 반영)
        days_diff = (today - op_req_date).days
        op_status = '확정'
        op_confirm_date_str = ''
        
        if days_diff <= 14: # 최근 신청건
            op_status = np.random.choice(['확정', '대기', '반려'], p=[0.5, 0.4, 0.1])
        else: # 과거 신청건
            op_status = np.random.choice(['확정', '반려'], p=[0.99, 0.01])

        # 확정일자 (신청 후 3일 ~ 14일)
        if op_status == '확정':
            confirm_days = random.randint(3, 14)
            op_confirm_date = op_req_date + timedelta(days=confirm_days)
            if op_confirm_date > today: op_confirm_date = today
            op_confirm_date_str = op_confirm_date.strftime('%Y-%m-%d')
        else:
            op_confirm_date = op_req_date # 미확정이면 커서 이동용으로만 사용

        # 운용 신청 데이터 적재
        req_type = '신규운용' if need_initial_req else '재사용'
        op_req_row = {
            '운용신청일자': op_req_date_str,
            '등록일자': op_req_date_str,
            '운용확정일자': op_confirm_date_str,
            '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
            '승인상태': op_status,
            'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
            '물품고유번호': asset_id, 
            '취득일자': row.취득일자, '취득금액': total_amount,
            '운용부서': current_dept_name, '사용자': row.비고, '신청구분': req_type
        }
        operation_req_list.append(op_req_row)
        need_initial_req = False 

        # 확정되지 않으면 루프 종료 (운용 시작 안됨)
        if op_status != '확정':
            active_flag = False
            break

        # [운용 시작] 상태 업데이트
        use_start_date = op_confirm_date
        sim_cursor_date = use_start_date # 커서 이동
        
        prev_status = current_status
        current_status = '운용'

        # 대장 반영
        df_operation.at[idx, '운용상태'] = '운용'
        df_operation.at[idx, '운용부서'] = current_dept_name
        df_operation.at[idx, '운용부서코드'] = current_dept_code
        
        # 출력상태 (랜덤)
        if loop_count == 1: 
             df_operation.at[idx, '출력상태'] = np.random.choice(['출력', '미출력'], p=PROBS_PRINT_STATUS)

        # 이력
        operation_history_list.append({
            '물품고유번호': asset_id,
            '변경일자': op_confirm_date_str,
            '(이전)운용상태': prev_status, '(변경)운용상태': '운용',
            '사유': f'{req_type} 승인 및 사용 시작',
            '관리자명': STAFF_USER[1], '관리자ID': STAFF_USER[0],
            '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
        })

        # ===================================================
        # B. [운용 중 사건 발생 결정] - 반납/직권불용/유지 결정
        # ===================================================
        next_event = '유지' # 기본값
        event_date = today + timedelta(days=1) # 기본값(미래)

        # 현재 물품의 나이 계산
        age_days = (today - acq_date).days
        days_since_use_start = (today - use_start_date).days
        
        # ---------------------------------------------------
        # 시나리오 1: 조기 반납 (신품/잉여)
        # 조건: 운용 시작한 지 얼마 안 됨 + 확률 1%
        # ---------------------------------------------------
        is_early_return = False
        if random.random() < PROB_EARLY_RETURN:
            # 운용확정일로부터 30일 이내 반납 가능한지 체크
            # (단, 현재 시뮬레이션 시점이 해당 기간을 지나지 않았거나, 과거 데이터 생성용일 때)
            
            # 반납일 = 운용확정일 + 1~30일
            early_return_date = use_start_date + timedelta(days=random.randint(1, 30))
            
            # 시간 역행 방지: 계산된 반납일이 sim_cursor_date보다 뒤여야 함
            if early_return_date > sim_cursor_date:
                event_date = early_return_date
                next_event = '반납'
                is_early_return = True

        # ---------------------------------------------------
        # 시나리오 2: 일반/노후 반납 (고장/내구연한)
        # 조건: 취득 후 3년 이상 경과 (하자/노후는 3년 이후부터)
        # ---------------------------------------------------
        if next_event == '유지' and age_days > (365 * 3):
            # 확률 결정
            prob_return = PROB_RETURN_OVER_3Y # 5%
            if age_days > (365 * 5): 
                prob_return = PROB_RETURN_OVER_5Y # 15%
            
            if random.random() < prob_return:
                # 반납일 = 현재 시점 + 30~365일 사이 (미래 사건)
                late_return_date = sim_cursor_date + timedelta(days=random.randint(30, 365))
                
                event_date = late_return_date
                next_event = '반납'
                is_early_return = False

        # ---------------------------------------------------
        # 시나리오 3: 직권 불용 (아주 오래된 물품)
        # ---------------------------------------------------
        if next_event == '유지' and age_days > (365 * 8): # 8년 넘은 것들
            if random.random() < 0.05: # 5% 확률로 직권 폐기
                event_date = sim_cursor_date + timedelta(days=random.randint(30, 90))
                next_event = '직권불용'

        # 미래 날짜 체크
        if event_date > today:
            active_flag = False
            break
        
        sim_cursor_date = event_date # 커서 이동

        # ===================================================
        # C. [사건 처리 로직]
        # ===================================================
        
        # CASE 1: 유지 (Loop 종료)
        if next_event == '유지':
            active_flag = False
            break

        # CASE 2: 반납
        elif next_event == '반납':
            # [사유 및 상태 결정 로직 분기]
            if is_early_return:
                # 조기 반납 -> 무조건 '잉여물품', 상태 '신품'
                return_reason = '잉여물품'
                current_condition = '신품'
            else:
                # 일반 반납 -> 고장, 내구연한 등 (잉여물품 제외)
                # 3년 넘은 물건을 잉여라고 하진 않으므로 제외
                late_reasons = ['사용연한경과', '고장/파손', '불용결정', '사업종료']
                late_probs = [0.5, 0.3, 0.1, 0.1]
                return_reason = np.random.choice(late_reasons, p=late_probs)
                
                if return_reason == '고장/파손': current_condition = '정비필요품'
                elif return_reason == '사용연한경과': current_condition = '폐품'
                else: current_condition = '중고품'

            # 반납 승인 상태
            return_status = np.random.choice(STATUS_CHOICES, p=PROBS_STATUS_RETURN)
            
            # [대기 상태 처리] 날짜를 최근으로 재설정
            if return_status == '대기':
                min_allowed = max(event_date, RECENT_WAIT_START)
                if min_allowed > today: min_allowed = today
                temp_date = fake.date_between(start_date=min_allowed, end_date=today)
                return_date = datetime(temp_date.year, temp_date.month, temp_date.day)
            else:
                return_date = event_date

            # 반납 확정일자 (신청 + 3일 ~ 2주)
            rt_confirm_date_str = ''
            rt_confirm_date = return_date # 초기화

            if return_status == '확정':
                rt_confirm_date = return_date + timedelta(days=random.randint(3, 14))
                if rt_confirm_date > today: rt_confirm_date = today
                rt_confirm_date_str = rt_confirm_date.strftime('%Y-%m-%d')

                # 대장 업데이트
                df_operation.at[idx, '운용상태'] = '반납'
                df_operation.at[idx, '운용부서'] = ''
                
                # 이력
                operation_history_list.append({
                    '물품고유번호': asset_id,
                    '변경일자': rt_confirm_date_str,
                    '(이전)운용상태': '운용', '(변경)운용상태': '반납',
                    '사유': return_reason,
                    '관리자명': STAFF_USER[1], '관리자ID': STAFF_USER[0],
                    '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
                })
                sim_cursor_date = rt_confirm_date # 커서 이동

            # 반납 데이터 저장
            return_list.append({
                '반납일자': return_date.strftime('%Y-%m-%d'),
                '반납확정일자': rt_confirm_date_str,
                '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
                '승인상태': return_status,
                'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                '물품고유번호': asset_id, '취득일자': row.취득일자,'취득금액': total_amount,
                '정리일자': clear_date_str, 
                '운용부서': current_dept_name, '운용상태': '운용', 
                '물품상태': current_condition, '사유': return_reason
            })

            # [반납 후 분기] 확정 건에 한해 재사용 vs 불용 결정
            if return_status == '확정':
                # 재사용 조건: 신품이고 10% 확률
                if current_condition == '신품' and random.random() < 0.1:
                    # -> 재사용 결정! (부서 변경 후 Loop 처음으로)
                    new_dept = random.choice(DEPT_MASTER_DATA)
                    current_dept_code = new_dept[0]
                    current_dept_name = new_dept[1]
                    continue 
                else:
                    # -> 불용 결정! (아래 불용 로직으로 진입)
                    next_event = '불용진행'
            else:
                # 대기/반려면 루프 종료
                active_flag = False
                break
        
        # CASE 3: 직권 불용 OR 반납 후 불용 (Loop 종료 예정)
        if next_event == '직권불용' or next_event == '불용진행':
            
            disuse_reason_mapped = ''
            prev_status_for_disuse = ''

            if next_event == '직권불용':
                disuse_reason_mapped = '직권 불용(파손/노후)'
                current_condition = '폐품' # 직권불용은 주로 폐품
                prev_status_for_disuse = '운용'
            else:
                disuse_reason_mapped = return_reason # 반납 사유 승계
                prev_status_for_disuse = '반납'

            # 잉여물품 + 신품 -> 보관(불용 스킵) 확률 체크
            skip_disuse = False
            if disuse_reason_mapped == '잉여물품' and current_condition == '신품':
                if random.random() < PROB_SURPLUS_STORE:
                    skip_disuse = True
            
            if skip_disuse:
                active_flag = False
                break

            # 불용 신청 (사건일로부터 1~14일 뒤)
            du_date = sim_cursor_date + timedelta(days=random.randint(1, 14))
            
            # 승인 상태 결정
            disuse_status = np.random.choice(STATUS_CHOICES, p=PROBS_STATUS_DISUSE)
            
            # [대기 상태 처리]
            if disuse_status == '대기':
                min_allowed = max(du_date, RECENT_WAIT_START)
                if min_allowed > today: min_allowed = today
                temp_date = fake.date_between(start_date=min_allowed, end_date=today)
                du_date = datetime(temp_date.year, temp_date.month, temp_date.day)
            
            # 날짜 체크
            if du_date > today: 
                active_flag = False
                break

            # 불용 확정 (신청 + 14일 ~ 30일)
            du_confirm_str = ''
            du_confirm_date = du_date

            if disuse_status == '확정':
                du_confirm_date = du_date + timedelta(days=random.randint(14, 30))
                if du_confirm_date > today: du_confirm_date = today
                du_confirm_str = du_confirm_date.strftime('%Y-%m-%d')

                # 대장 업데이트
                df_operation.at[idx, '운용상태'] = '불용'
                
                # 이력
                operation_history_list.append({
                    '물품고유번호': asset_id,
                    '변경일자': du_confirm_str,
                    '(이전)운용상태': prev_status_for_disuse, '(변경)운용상태': '불용',
                    '사유': disuse_reason_mapped,
                    '관리자명': ADMIN_USER[1], '관리자ID': ADMIN_USER[0],
                    '등록자명': ADMIN_USER[1], '등록자ID': ADMIN_USER[0]
                })
                sim_cursor_date = du_confirm_date

            # 불용 데이터 저장
            disuse_list.append({
                '불용일자': du_date.strftime('%Y-%m-%d'),
                '불용확정일자': du_confirm_str,
                '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1],
                '승인상태': disuse_status,
                'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                '물품고유번호': asset_id, '취득일자': row.취득일자, '취득금액': total_amount,
                '정리일자': clear_date_str,
                '운용부서': current_dept_name if next_event == '직권불용' else '', 
                '운용상태' : prev_status_for_disuse, 
                '내용연수': life_years,
                '물품상태': current_condition, '사유': disuse_reason_mapped
            })

            # [처분 단계] (불용 확정 시에만)
            if disuse_status == '확정':
                # 처분 신청 (불용확정 + 1~14일)
                dp_date = sim_cursor_date + timedelta(days=random.randint(1, 14))
                
                # 처분 방식 결정
                if current_condition in ['신품', '중고품']:
                    method = np.random.choice(METHODS_DISPOSAL, p=PROBS_DISPOSAL_GOOD)
                else:
                    method = np.random.choice(METHODS_DISPOSAL, p=PROBS_DISPOSAL_BAD)
                
                # 승인 상태
                dp_status = np.random.choice(STATUS_CHOICES, p=PROBS_STATUS_DISPOSAL)
                
                # [대기 상태 처리]
                if dp_status == '대기':
                    min_allowed = max(dp_date, RECENT_WAIT_START)
                    if min_allowed > today: min_allowed = today
                    temp_date = fake.date_between(start_date=min_allowed, end_date=today)
                    dp_date = datetime(temp_date.year, temp_date.month, temp_date.day)

                if dp_date <= today:
                    dp_confirm_str = ''
                    
                    if dp_status == '확정':
                        # 처분 확정 (신청 + 30일 ~ 90일)
                        dp_confirm_date = dp_date + timedelta(days=random.randint(30, 90))
                        if dp_confirm_date > today: dp_confirm_date = today
                        dp_confirm_str = dp_confirm_date.strftime('%Y-%m-%d')

                        # 대장 업데이트 (최종)
                        df_operation.at[idx, '운용상태'] = '처분'
                        
                        # 이력
                        operation_history_list.append({
                            '물품고유번호': asset_id,
                            '변경일자': dp_confirm_str,
                            '(이전)운용상태': '불용', '(변경)운용상태': '처분',
                            '사유': f"{method} 완료",
                            '관리자명': ADMIN_USER[1], '관리자ID': ADMIN_USER[0],
                            '등록자명': ADMIN_USER[1], '등록자ID': ADMIN_USER[0]
                        })

                    # 처분 데이터 저장
                    disposal_list.append({
                        '처분일자': dp_date.strftime('%Y-%m-%d'),
                        '처분확정일자': dp_confirm_str,
                        '처분정리구분': method,
                        '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1],
                        '승인상태': dp_status,
                        'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                        '물품고유번호': asset_id, '취득일자': row.취득일자, '취득금액': total_amount,
                        '처분방식': method, '물품상태': current_condition, '사유': disuse_reason_mapped,
                        '불용일자': du_confirm_str, '내용연수': life_years, '정리일자': clear_date_str
                    })

            # 불용/처분 단계까지 오면 루프 종료
            active_flag = False
            break

# ---------------------------------------------------------
# 3. 데이터프레임 변환 및 저장
# ---------------------------------------------------------
df_op_req = pd.DataFrame(operation_req_list)
df_return = pd.DataFrame(return_list)
df_disuse = pd.DataFrame(disuse_list)
df_disposal = pd.DataFrame(disposal_list)
df_history = pd.DataFrame(operation_history_list)

# 저장
# [04-01] 물품 운용 대장 목록
cols_operation = [
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '캠퍼스','취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수', '출력상태', '승인상태', '취득정리구분', '운용부서코드', '비고'
]

# 누락 컬럼 보정 (안전장치)
if '비고' not in df_operation.columns:
    add_info = df_acq[['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '비고', '승인상태']].drop_duplicates()
    df_operation = df_operation.merge(
        add_info,
        on=['취득일자', 'G2B_목록번호', '취득정리구분', '운용부서코드', '승인상태'],
        how='left'
    )

df_operation[cols_operation].to_csv(os.path.join(DATA_DIR, '04_01_operation_master.csv'), index=False, encoding='utf-8-sig')

# [04-02] 운용 신청 목록
if not df_op_req.empty:
    df_op_req.to_csv(os.path.join(DATA_DIR, '04_02_operation_req_list.csv'), index=False, encoding='utf-8-sig')

# [04-03] 반납 관련
if not df_return.empty:
    df_return.to_csv(os.path.join(DATA_DIR, '04_03_return_list.csv'), index=False, encoding='utf-8-sig')

# [05-01] 불용 관련
if not df_disuse.empty:
    df_disuse.to_csv(os.path.join(DATA_DIR, '05_01_disuse_list.csv'), index=False, encoding='utf-8-sig')

# [06-01] 처분 관련
if not df_disposal.empty:
    df_disposal.to_csv(os.path.join(DATA_DIR, '06_01_disposal_list.csv'), index=False, encoding='utf-8-sig')

# [물품상태이력]
df_history.to_csv(os.path.join(DATA_DIR, '99_asset_status_history.csv'), index=False, encoding='utf-8-sig')

print("🎉 [Phase 2] 생애주기 시뮬레이션 및 파일 생성 완료!")
print(f"   - 운용 자산(개별): {len(df_operation)}건")
print(f"   - 운용 신청: {len(df_op_req)}건 (신규 + 재사용)")
print(f"   - 반납 발생: {len(df_return)}건")
print(f"   - 불용 발생: {len(df_disuse)}건")
print(f"   - 처분 발생: {len(df_disposal)}건")
print(f"   - 상태 변경 이력: {len(df_history)}건")