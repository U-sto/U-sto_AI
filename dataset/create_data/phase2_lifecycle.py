import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
fake = Faker('ko_KR')

# Phase 1 결과물 로드
try:
    df_acq = pd.read_csv('03_01_acquisition_master.csv')
    print(f"📂 [Phase 2] 취득 데이터 로드 완료: {len(df_acq)}건")
except FileNotFoundError:
    print("❌ 오류: '03_01_acquisition_master.csv' 파일이 없습니다. Phase 1을 먼저 실행해주세요.")
    exit()

# 사용자/부서 마스터 (Phase 1과 동일하게 유지하거나, 별도 파일로 관리 권장)
# 여기서는 시뮬레이션을 위해 일부 재정의
ADMIN_USER = ("hyl0610", "황팀장")
STAFF_USER = ("badbergjr", "박대리")

# ---------------------------------------------------------
# 1. 데이터 분화 (Explosion) & 고유번호 부여
# ---------------------------------------------------------
# 승인상태가 '확정'인 건만 운용 대장으로 넘어감
df_confirmed = df_acq[df_acq['승인상태'] == '확정'].copy()

# 행 복제 (수량 N -> N개 행)
# Index를 유지하면서 수량만큼 반복
df_operation = df_confirmed.loc[df_confirmed.index.repeat(df_confirmed['수량'])].reset_index(drop=True)

# 수량 1로 초기화 (개별 관리이므로)
df_operation['수량'] = 1

# 물품고유번호 생성 로직
# 규칙: M + 년도(4자리) + 순번(5자리) -> 예: M201500001
def generate_asset_id(row, idx):
    acq_date = pd.to_datetime(row['취득일자'])
    year_str = acq_date.strftime('%Y') # 4자리 년도 (ex: 2015)
    seq_str = f"{idx+1:05d}"           # 5자리 순번
    return f"M{year_str}{seq_str}"

print("⚙️ [Phase 2] 개별 자산 분화 및 고유번호 생성 중...")
# 인덱스를 활용해 고유번호 일괄 생성
df_operation['물품고유번호'] = [generate_asset_id(row, i) for i, row in df_operation.iterrows()]

# 초기 운용 상태 설정
# 정리일자가 있으면 그때부터 '운용중', 아니면 '취득(대기)' 상태일 수 있으나, 
# 시뮬레이션 편의상 확정된 건은 '운용' 또는 '취득'으로 시작
# 매뉴얼상: 취득 -> 운용 -> 반납 -> 불용
df_operation['운용상태'] = '취득' # 초기값

# ---------------------------------------------------------
# 2. 생애주기 시뮬레이션 (Lifecycle)
# ---------------------------------------------------------
# 결과를 담을 리스트들
operation_history_list = [] # 이력 데이터
return_list = [] # 반납 목록
disuse_list = [] # 불용 목록
disposal_list = [] # 처분 목록

# 현재 시점 (Today)
today = datetime.now()

print("⏳ [Phase 2] 자산 생애주기 시뮬레이션 시작 (반납/불용/처분)...")

for idx, row in df_operation.iterrows():
    # -------------------------------------------------------
    # 기본 변수 세팅 (요청하신 스타일대로 Unpacking)
    # -------------------------------------------------------
    # [물품운용대장목록] 관련
    g2b_full_code = row['G2B_목록번호'] # 추가
    g2b_name = row['G2B_목록명']
    asset_id = row['물품고유번호']
    acq_date = pd.to_datetime(row['취득일자'])
    total_amount = row['취득금액'] # 추가

    # 정리일자 Null 처리
    if pd.isna(row['정리일자']) or row['정리일자'] == '':
        clear_date = acq_date # 임시
        clear_date_str = ''
    else:
        clear_date = pd.to_datetime(row['정리일자'])
        clear_date_str = clear_date.strftime('%Y-%m-%d')
    
    dept_name = row['운용부서'] # 추가
    # 운용상태는 루프 내에서 변경 ?
    life_years = row['내용연수']

    # [물품기본정보] 관련
    acq_method = row['취득정리구분'] # 추가
    dept_id = row['운용부서코드'] # 추가
    # 수량 추가...
    remark = row['비고'] # 추가
    
    # -------------------------------------------------------
    # 2-1. 운용 시작 (취득 -> 운용)
    # -------------------------------------------------------
    # 정리일자에 '취득' 상태 기록
    operation_history_list.append({
        '물품고유번호': asset_id,
        '변경일자': clear_date.strftime('%Y-%m-%d'),
        '(이전)운용상태': '-',
        '(변경)운용상태': '취득',
        '사유': '신규 취득',
        '관리자명': STAFF_USER[1], '관리자ID': STAFF_USER[0],
        '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
    })
    
    # 정리일자 직후 '운용'으로 변경된다고 가정
    use_start_date = clear_date + timedelta(days=random.randint(0, 3))
    df_operation.at[idx, '운용상태'] = '운용' # 현재 상태 업데이트
    
    operation_history_list.append({
        '물품고유번호': asset_id,
        '변경일자': use_start_date.strftime('%Y-%m-%d'),
        '(이전)운용상태': '취득',
        '(변경)운용상태': '운용',
        '사유': '부서 배정 및 사용 시작',
        '관리자명': remark if (pd.notna(remark) and remark) else STAFF_USER[1], # 비고에 관리자 있을 수 있음
        '관리자ID': STAFF_USER[0],
        '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
    })
    
    # -------------------------------------------------------
    # 2-2. 반납 시뮬레이션 (운용중 -> 반납)
    # 조건: 취득 후 3년 이상 지난 물품 중 일부(약 30%), 혹은 고장난 물품
    # -------------------------------------------------------
    is_returned = False
    return_date = None
    return_row = None
    item_condition = '중고품'
    return_reason = ''
    
    # 확률적 반납 결정 (내구연한 도래 여부와 관계없이 발생 가능)
    # 오래된 물건일수록 반납 확률 증가
    age_days = (today - acq_date).days

    # 반납 확률 로직
    prob_return = 0.0
    if age_days > 365 * 3: prob_return = 0.3 # 3년 지남
    if age_days > 365 * 5: prob_return = 0.6 # 5년 지남 (내구연한)
    
    if random.random() < prob_return:
        # 반납 발생!
        # 반납 시점: 운용 시작일 ~ 오늘 사이 랜덤, 단 최소 1년은 썼다고 가정
        if age_days > 365:
            return_date = use_start_date + timedelta(days=random.randint(365, age_days))
            if return_date > today: return_date = today # 미래 날짜 방지
            
            # 반납 사유 결정
            return_reason = np.random.choice(
                ['사용연한경과', '고장/파손', '불용결정', '사업종료', '잉여물품'], 
                p=[0.4, 0.2, 0.2, 0.1, 0.1]
            )
            
            # 물품 상태 결정 (사유에 따라)
            if return_reason == '고장/파손': item_condition = '정비필요품'
            elif return_reason == '사용연한경과': item_condition = '폐품'
            elif return_reason == '잉여물품': item_condition = '신품' # 잉여물품은 주로 신품/상태좋음
            else: item_condition = '중고품'

            # 반납 승인 절차 (90:9.5:0.5)
            return_status = np.random.choice(['확정', '대기', '반려'], p=[0.90, 0.095, 0.005])
            
            # 반납 데이터 생성
            return_row = {
                # ---------------반납등록목록-----------------
                '반납일자': return_date.strftime('%Y-%m-%d'),
                '반납확정일자': (return_date + timedelta(days=3)).strftime('%Y-%m-%d') if return_status == '확정' else '',
                '등록자ID': STAFF_USER[0], '등록자명': STAFF_USER[1],
                '승인상태': return_status,
                # 물품 정보
                # ---------------반납물품목록-----------------
                'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                '물품고유번호': asset_id, '취득일자': row['취득일자'], '취득금액': total_amount,
                '정리일자': clear_date_str, # 취득 시 정리일자  
                '운용부서': dept_name, '물품상태': item_condition, '사유': return_reason
            }
            return_list.append(return_row)
            
            if return_status == '확정':
                is_returned = True
                df_operation.at[idx, '운용상태'] = '반납'
                df_operation.at[idx, '운용부서'] = '' # 반납 시 부서 Null 처리

                # 이력 추가
                operation_history_list.append({
                    '물품고유번호': asset_id,
                    '변경일자': return_date.strftime('%Y-%m-%d'), # 반납일자
                    '(이전)운용상태': '운용', '(변경)운용상태': '반납',
                    '사유': return_reason,
                    '관리자명': STAFF_USER[1], '관리자ID': STAFF_USER[0],
                    '등록자명': STAFF_USER[1], '등록자ID': STAFF_USER[0]
                })

    # -------------------------------------------------------
    # 2-3. 불용 시뮬레이션 (반납 -> 불용)
    # 조건: 반납 확정된 물품 중 '폐품', '정비필요품' or 내구연한 경과품
    # -------------------------------------------------------
    is_disused = False
    disuse_date = None
    disuse_row = None
    
    if is_returned and return_row and return_row['반납확정일자']:
        # 잉여물품 + 신품인 경우 불용 스킵(보관) 로직
        skip_disuse = False
        disuse_reason = ''
        
        # 보관 로직 (잉여물품 + 신품 -> 보관)
        if return_reason == '잉여물품' and item_condition == '신품':
            if random.random() > 0.1: # 90% 확률로 보관 (불용 X)
                skip_disuse = True
            else:
                # 10% 확률로 불용 처리 (사유 변경)
                disuse_reason = '활용부서 부재' # 10% 불용 진행
        
        if not skip_disuse:
            disuse_base_date = pd.to_datetime(return_row['반납확정일자'])
            disuse_date = disuse_base_date + timedelta(days=random.randint(30, 180))
            
            # 불용 사유 결정 (4종) - 반납 사유와 매핑
            if not disuse_reason:
                if return_reason == '사용연한경과':
                    disuse_reason = '내구연한 경과'
                elif return_reason == '고장/파손':
                    disuse_reason = '수리비용과다'
                elif return_reason == '사업종료':
                    disuse_reason = '활용부서부재'
                elif return_reason == '잉여물품': # 위의 잉여물품 로직을 통과한 경우
                    disuse_reason = '활용부서부재'
                else: # 불용결정 등
                    disuse_reason = '구형화'
                
            disuse_status = np.random.choice(['확정', '대기', '반려'], p=[0.70, 0.25, 0.05])
                
            # 불용일자 = 불용확정일자 (승인확정 시)
            confirm_date_str = ''
            if disuse_status == '확정':
                confirm_date_str = (disuse_date + timedelta(days=7)).strftime('%Y-%m-%d')
                disuse_date_str = confirm_date_str 
            else:
                disuse_date_str = disuse_date.strftime('%Y-%m-%d') # 확정 전이면 신청일
            
            # 불용 데이터 생성
            disuse_row = {
                # ---------------불용등록목록-----------------
                '불용일자': disuse_date.strftime('%Y-%m-%d'),
                '불용확정일자': confirm_date_str,
                '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1], # 관리자가 보통 처리
                '승인상태': disuse_status,
                # 물품 정보
                # ---------------불용물품목록-----------------
                'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                    '물품고유번호': asset_id, '취득일자': row['취득일자'], '취득금액': total_amount,
                    '정리일자': clear_date_str, # 취득 시 정리일자  
                    '운용부서': '', 
                    '물품상태': return_row['물품상태'], '사유': disuse_reason
            }
            disuse_list.append(disuse_row)
            
            if disuse_status == '확정':
                is_disused = True
                df_operation.at[idx, '운용상태'] = '불용'
                # 이력 추가
                operation_history_list.append({
                    '물품고유번호': asset_id,
                    '변경일자': disuse_date.strftime('%Y-%m-%d'), # 불용일자
                    '(이전)운용상태': '반납', '(변경)운용상태': '불용',
                    '사유': disuse_reason,
                    '관리자명': ADMIN_USER[1], '관리자ID': ADMIN_USER[0],
                    '등록자명': ADMIN_USER[1], '등록자ID': ADMIN_USER[0]
                })

    # -------------------------------------------------------
    # 2-4. 처분 시뮬레이션 (불용 -> 처분)
    # 조건: 불용 확정된 물품은 무조건 처분 (매각/폐기)
    # -------------------------------------------------------
    if is_disused and disuse_row and disuse_row['불용확정일자']:
        disposal_base_date = pd.to_datetime(disuse_row['불용확정일자'])
        disposal_date = disposal_base_date + timedelta(days=random.randint(14, 60))
        
        if disposal_date <= today:
            # [수정 1] 물품 상태에 따른 처분정리구분 결정
            # 상태가 좋음(신품, 중고품) -> 주로 '매각'
            # 상태가 나쁨(정비필요품, 폐품) -> 주로 '폐기'
            current_condition = disuse_row['물품상태']
            
            if current_condition in ['신품', '중고품']:
                # 매각 85%, 폐기 13%, 멸실 1%, 도난 1%
                disposal_method = np.random.choice(
                    ['매각', '폐기', '멸실', '도난'], 
                    p=[0.85, 0.13, 0.01, 0.01]
                )
            else:
                # 폐기 95%, 매각 3%, 멸실 1%, 도난 1%
                disposal_method = np.random.choice(
                    ['폐기', '매각', '멸실', '도난'], 
                    p=[0.95, 0.03, 0.01, 0.01]
                )
            # 처분 사유는 불용 사유와 동일하게 설정 (요청사항 반영)
            disposal_reason = disuse_row['사유']

            # 처분 승인 상태 비율 설정 (확정 90%, 대기 8%, 반려 2%)
            disposal_status = np.random.choice(['확정', '대기', '반려'], p=[0.90, 0.08, 0.02])
            
            # 처분확정일자 생성 로직
            disposal_confirm_date_str = ''
            if disposal_status == '확정':
                # 신청일로부터 3~7일 후 확정
                confirm_date = disposal_date + timedelta(days=random.randint(3, 7))
                if confirm_date > today: confirm_date = today # 미래 날짜 방지
                disposal_confirm_date_str = confirm_date.strftime('%Y-%m-%d')

            disposal_row = {
                # ---------------처분목록-----------------
                '처분일자': disposal_date.strftime('%Y-%m-%d'),
                '처분정리구분': disposal_method,
                '등록자ID': ADMIN_USER[0], '등록자명': ADMIN_USER[1],
                '승인상태': disposal_status,
                # ---------------처분물품목록-----------------
                'G2B_목록번호': g2b_full_code, 'G2B_목록명': g2b_name,
                '물품고유번호': asset_id, '취득일자': row['취득일자'], '취득금액': total_amount,
                '처분방식': disposal_method, '물품상태': disuse_row['물품상태'], '사유': disuse_row['사유'],
                # ---------------처분등록정보-----------------
                '불용일자': disuse_row['불용확정일자'],  # 불용승인확정일과 동일
                '내용연수': life_years,    
                '처분확정일자': disposal_confirm_date_str,
                '정리일자': clear_date_str # 취득 시 정리일자     
            }

            disposal_list.append(disposal_row)
            
            # [중요] '확정'인 경우에만 실제 대장의 상태를 '처분'으로 변경하고 이력을 남김
            if disposal_status == '확정':
                df_operation.at[idx, '운용상태'] = '처분' # 매뉴얼상 처분 완료되면 목록에서 사라지거나 상태 변경
                # 이력 추가
                operation_history_list.append({
                    '물품고유번호': asset_id,
                    '변경일자': disposal_date.strftime('%Y-%m-%d'),
                    '(이전)운용상태': '불용', '(변경)운용상태': '처분',
                    '사유': f"{disposal_method} 완료",
                    '관리자명': ADMIN_USER[1], '관리자ID': ADMIN_USER[0],
                    '등록자명': ADMIN_USER[1], '등록자ID': ADMIN_USER[0]
                })

# ---------------------------------------------------------
# 3. 데이터프레임 변환 및 저장
# ---------------------------------------------------------
df_return = pd.DataFrame(return_list)
df_disuse = pd.DataFrame(disuse_list)
df_disposal = pd.DataFrame(disposal_list)
df_history = pd.DataFrame(operation_history_list)

# 저장
# [04-01] 물품 운용 대장 목록 (최종 상태가 반영된 Main Table)
cols_operation = [
    'G2B_목록번호', 'G2B_목록명', '물품고유번호', '취득일자', '취득금액', '정리일자', 
    '운용부서', '운용상태', '내용연수'
]
df_operation[cols_operation].to_csv('04_01_operation_master.csv', index=False, encoding='utf-8-sig')

# [04-03] 반납 관련
if not df_return.empty:
    df_return.to_csv('04_03_return_list.csv', index=False, encoding='utf-8-sig')

# [05-01] 불용 관련
if not df_disuse.empty:
    df_disuse.to_csv('05_01_disuse_list.csv', index=False, encoding='utf-8-sig')

# [06-01] 처분 관련
if not df_disposal.empty:
    df_disposal.to_csv('06_01_disposal_list.csv', index=False, encoding='utf-8-sig')

# [물품상태이력] (상세 페이지용)
df_history.to_csv('99_asset_status_history.csv', index=False, encoding='utf-8-sig')

print("✅ [Phase 2] 생애주기 시뮬레이션 및 파일 생성 완료!")
print(f"   - 운용 자산(개별): {len(df_operation)}건")
print(f"   - 반납 발생: {len(df_return)}건")
print(f"   - 불용 발생: {len(df_disuse)}건")
print(f"   - 처분 발생: {len(df_disposal)}건")
print(f"   - 상태 변경 이력: {len(df_history)}건")