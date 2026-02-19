import pandas as pd
import numpy as np
import os
from pandas.errors import EmptyDataError

# ---------------------------------------------------------
# 0. 설정 및 데이터 로드
# ---------------------------------------------------------
# 기준일을 시스템 '오늘'로 잡으면 코드를 돌릴 때마다 수명과 잔여일수가 달라져서 
# 모델 재현성(Reproducibility)이 떨어지므로 특정 일자로 고정하는 것이 좋음
FIXED_TODAY_STR = "2026-02-10"
today = pd.to_datetime(FIXED_TODAY_STR).normalize()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOAD_DIR = os.path.join(BASE_DIR, "data_lifecycle")
SAVE_DIR = os.path.join(BASE_DIR, "data_ml")
os.makedirs(SAVE_DIR, exist_ok=True)

print("📂 [Phase 4] AI 학습용 데이터 전처리 시작...")

# 빈 데이터프레임이 로드될 경우를 대비해, 병합 시 KeyError가 나지 않도록 필수 컬럼을 명시
COLS_RT = ['물품고유번호', '반납일자', '반납확정일자', '사유', '승인상태']
COLS_DU = ['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']
COLS_DP = ['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']

def load_csv_safe(filename, required=False, expected_cols=None):
    """안전하게 CSV 파일을 불러오고, 없거나 비어있으면 빈 DataFrame을 반환하는 함수"""
    filepath = os.path.join(LOAD_DIR, filename)
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except EmptyDataError:
            return pd.DataFrame(columns=expected_cols) if expected_cols else pd.DataFrame()
    else:
        if required:
            print(f"❌ 필수 데이터 파일 누락: {filename}")
            exit()
        return pd.DataFrame(columns=expected_cols) if expected_cols else pd.DataFrame()

# 1. 원천 데이터 로드
df_op = load_csv_safe('04_01_operation_master.csv', required=True) 
df_rt = load_csv_safe('04_03_return_list.csv', expected_cols=COLS_RT)      
df_du = load_csv_safe('05_01_disuse_list.csv', expected_cols=COLS_DU)      
df_dp = load_csv_safe('06_01_disposal_list.csv', expected_cols=COLS_DP)    

print(f"   - 원천 데이터 로드 완료: 운용 대장 {len(df_op)}건")

# 하나의 물품이 여러 번 반납/불용 이력을 가질 수 있으므로
# 최신 이력(확정일자 기준 내림차순) 하나만 남기고 중복을 제거해야 1:1 병합이 깔끔하게 됨
def drop_duplicates_safe(df, date_col, conf_date_col):
    if not df.empty:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if conf_date_col in df.columns:
            df[conf_date_col] = pd.to_datetime(df[conf_date_col], errors='coerce')
            df = df.sort_values(by=['물품고유번호', conf_date_col, date_col], ascending=[True, False, False])
        return df.drop_duplicates(subset=['물품고유번호'], keep='first')
    return df

df_rt = drop_duplicates_safe(df_rt, '반납일자', '반납확정일자')
df_du = drop_duplicates_safe(df_du, '불용일자', '불용확정일자')
df_dp = drop_duplicates_safe(df_dp, '처분확정일자', '처분확정일자') # 처분은 일자가 하나뿐이므로 동일하게 처리

# ---------------------------------------------------------
# 1. 데이터 병합 (Master Table 생성)
# ---------------------------------------------------------
print("   1. 생애주기 병합 (운용+반납+불용+처분)...")

# 운용 마스터에 반납, 불용, 처분 이력을 Left Join으로 붙임
df_merged = pd.merge(df_op, df_rt[['물품고유번호', '반납일자', '반납확정일자', '사유', '승인상태']].rename(columns={'승인상태': '반납승인상태', '사유': '반납사유'}), on='물품고유번호', how='left')
df_merged = pd.merge(df_merged, df_du[['물품고유번호', '불용일자', '불용확정일자', '사유', '승인상태']].rename(columns={'사유': '불용사유', '승인상태': '불용승인상태'}), on='물품고유번호', how='left')
df_merged = pd.merge(df_merged, df_dp[['물품고유번호', '처분방식', '처분확정일자', '물품상태', '승인상태']].rename(columns={'승인상태': '처분승인상태'}), on='물품고유번호', how='left')

# ---------------------------------------------------------
# 2. 전처리 및 결측치 보정 
# ---------------------------------------------------------
print("   2. 결측치 보정 및 기준일 산출...")

date_cols = ['취득일자', '반납일자', '불용일자', '반납확정일자', '불용확정일자', '처분확정일자']
for col in date_cols:
    if col in df_merged.columns:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

# 수명 계산의 끝점이 되는 '종료일'을 구하는 로직
# 1순위: 처분 > 2순위: 불용 > 3순위: 반납 순으로 우선순위를 둠.
confirmed_end_date = df_merged.get('처분확정일자').combine_first(df_merged.get('불용확정일자')).combine_first(df_merged.get('반납확정일자'))
valid_ret_date = df_merged['반납일자'].where(df_merged['반납승인상태'] == '확정')
valid_disuse_date = df_merged['불용일자'].where(df_merged['불용승인상태'] == '확정')
fallback_end_date = valid_ret_date.combine_first(valid_disuse_date)

df_merged['최종종료일'] = confirmed_end_date.combine_first(fallback_end_date)
# 종료일이 없으면 '현재 운용 중'이라는 뜻이므로 기준일을 today로 설정
df_merged['기준일'] = df_merged['최종종료일'].fillna(today)

# 머신러닝용 최종 데이터프레임 뼈대 구축
df_final = df_merged[['물품고유번호', '취득금액', '운용부서코드', '캠퍼스', '취득일자', '반납일자', '불용일자', '반납사유', '불용사유', '물품상태', '처분방식', '기준일']].copy()
df_final['G2B목록명'] = df_merged.get('G2B_목록명', df_merged.get('G2B목록명'))
df_final['물품분류명'] = df_merged.get('물품분류명', df_final['G2B목록명'])
df_final['운용부서명'] = df_merged.get('운용부서', df_merged.get('운용부서명'))
df_final['내용연수'] = df_merged.get('내용연수')

# 결측치 보정 (가격이 0이거나 없는 경우 중앙값으로, 내용연수가 없으면 최빈값(보통 5년)으로)
median_price = df_final.loc[df_final['취득금액'] > 0, '취득금액'].median()
df_final['취득금액'] = df_final['취득금액'].fillna(median_price).replace(0, median_price)
df_final['내용연수'] = df_final['내용연수'].fillna(df_final['내용연수'].mode()[0] if not df_final['내용연수'].mode().empty else 5)
df_final = df_final.dropna(subset=['취득일자']) # 시작일이 없으면 수명 계산이 불가하므로 제거

# ---------------------------------------------------------
# 3. 파생 변수 생성 및 타겟 정의
# ---------------------------------------------------------
print("   3. 파생변수 생성 및 이상치 처리...")

# 운용연차 산출 (년 단위 환산)
df_final['운용연차'] = ((df_final['기준일'] - df_final['취득일자']).dt.days.clip(lower=0) / 365.0).round(2)

# 학습에 사용할 "수명이다 한(Dead)" 데이터를 정의하는 로직 (정답지 확보)
is_disposal = df_final['처분방식'].isin(['폐기', '멸실'])
is_sale_eol = (df_final['처분방식'] == '매각') & df_final['불용사유'].isin(['고장/파손', '노후화(성능저하)', '수리비용과다', '구형화', '내구연한 경과(노후화)'])
is_return_repair = df_final['반납일자'].notna() & (df_final['물품상태'] == '정비필요품')

df_final['학습데이터여부'] = np.where(is_disposal | is_sale_eol | is_return_repair, 'Y', 'N')

# --- IQR 기반 이상치(Outlier) 제거 ---
# 이유: 학습 데이터에 수명이 0.1년이거나 50년인 극단적 데이터가 섞여 있으면 모델이 흔들림.
train_cond = df_final['학습데이터여부'] == 'Y'
Q1 = df_final.loc[train_cond, '운용연차'].quantile(0.25)
Q3 = df_final.loc[train_cond, '운용연차'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 예측해야 할 데이터('N')는 살리고, 학습 데이터('Y') 중에서 정상 범주에 있는 것만 남김
valid_data_mask = (df_final['학습데이터여부'] == 'N') | ((df_final['학습데이터여부'] == 'Y') & (df_final['운용연차'] >= lower_bound) & (df_final['운용연차'] <= upper_bound))
removed_count = len(df_final) - valid_data_mask.sum()
df_final = df_final[valid_data_mask].copy()
print(f"      * [이상치 제거] 극단적 수명을 가진 학습 데이터 {removed_count}건 제외 (정상범위: {lower_bound:.2f} ~ {upper_bound:.2f}년)")

# 추가 파생 변수 산출
df_final['잔여내용연수'] = (df_final['내용연수'] - df_final['운용연차']).round(2)

def get_severity(dept_name):
    if pd.isna(dept_name): return 1.0
    dept_str = str(dept_name)
    if any(k in dept_str for k in ['소프트웨어', '공학', '전산', 'AI', '정보','공과', '컴퓨터']): return 1.3
    if any(k in dept_str for k in ['연구', '실험', '과학']): return 1.2
    return 1.0

df_final['부서가혹도'] = df_final['운용부서명'].apply(get_severity)
df_final['누적사용부하'] = (df_final['운용연차'] * df_final['부서가혹도']).round(2)
df_final['고장임박도'] = ((df_final['운용연차'] / df_final['내용연수']) ** 2).clip(0, 1).round(2)

# 예산/구매 관련 지표
df_final['가격민감도'] = (np.log1p(df_final['취득금액']) / np.log1p(100000000)).clip(0, 1).round(2)
df_final['리드타임등급'] = df_final['취득금액'].apply(lambda x: 0 if x < 5000000 else (1 if x < 30000000 else 2))
df_final['장비중요도'] = ((df_final['가격민감도'] * 0.7) + ((df_final['리드타임등급'] * 0.5) * 0.3)).round(2)
df_final['취득월'] = df_final['취득일자'].dt.month

# ---------------------------------------------------------
# 4. 컬럼정의서 기반 결과 Placeholder 세팅 및 인코딩
# ---------------------------------------------------------
# 타겟 세팅 (실제수명: 학습에 쓰일 Y값)
df_final['실제수명'] = np.nan
df_final.loc[df_final['학습데이터여부'] == 'Y', '실제수명'] = df_final['운용연차']

# 컬럼정의서 필수 산출물 빈칸(Placeholder) 처리
# (이 값들은 Phase 6 예측 단계나 LLM 후처리 단계에서 채워짐)
df_final['서비스계수'] = np.nan 
df_final['실제잔여수명'] = np.where(df_final['학습데이터여부'] == 'Y', 0.0, np.nan)
df_final['예측잔여수명'] = np.nan
df_final['(월별)고장예상수량'] = 0
df_final['안전재고'] = 0
df_final['(월별)필요수량'] = 0
df_final['AI예측고장일'] = pd.NaT
df_final['안전버퍼'] = 0.0
df_final['권장발주일'] = pd.NaT
df_final['예측실행일자'] = today.strftime('%Y-%m-%d')

# 범주형 수치화 (모델이 이해할 수 있도록 라벨 인코딩)
for col in ['G2B목록명', '물품분류명', '운용부서코드', '캠퍼스']:
    df_final[col] = df_final[col].fillna('Unknown').astype(str)
    df_final[f'{col}_Code'] = pd.factorize(df_final[col], sort=True)[0]

# ---------------------------------------------------------
# 5. 데이터 분할 및 저장 (Train / Valid / Test / Pred)
# ---------------------------------------------------------
df_train_source = df_final[df_final['학습데이터여부'] == 'Y'].copy().sort_values(by='취득일자')

# 시계열성이 있는 자산 데이터이므로 랜덤 셔플링보다는 취득일자 순으로 잘라서 
# 과거 데이터로 미래를 예측하는 Time-Series Split 방식을 흉내내는 것이 좋음
n_total = len(df_train_source)
n_train, n_valid = int(n_total * 0.7), int(n_total * 0.2)

df_final['데이터세트구분'] = 'Prediction' # 기본값은 예측 대상
df_final.loc[df_train_source.iloc[:n_train].index, '데이터세트구분'] = 'Train'
df_final.loc[df_train_source.iloc[n_train : n_train + n_valid].index, '데이터세트구분'] = 'Valid'
df_final.loc[df_train_source.iloc[n_train + n_valid:].index,  '데이터세트구분'] = 'Test'

# 최종 출력 컬럼 지정 (컬럼정의서 매핑 반영 & 데이터 누수 방지)
output_cols = [
    # 정적 & 기본 정보
    '물품고유번호', 'G2B목록명', '물품분류명', '운용부서코드', '운용부서명', '캠퍼스',
    '취득일자', '반납일자', '불용일자', '처분방식', '물품상태', '불용사유', '반납사유', 
    
    # 파생 변수 (Features)
    '내용연수', '취득금액', '운용연차', '잔여내용연수', '부서가혹도', '누적사용부하',
    '고장임박도', '가격민감도', '장비중요도', '리드타임등급', '취득월',
    'G2B목록명_Code', '물품분류명_Code', '운용부서코드_Code', '캠퍼스_Code',
    
    # 타겟 및 구분
    '실제수명', '학습데이터여부', '데이터세트구분',
    
    # 예측값/결과값 (컬럼정의서 매핑 완벽 대응)
    '서비스계수', '실제잔여수명', '예측잔여수명', '(월별)고장예상수량', '안전재고', 
    '(월별)필요수량', 'AI예측고장일', '안전버퍼', '권장발주일', '예측실행일자'
]

df_export = df_final.reindex(columns=output_cols)
save_path = os.path.join(SAVE_DIR, 'phase4_training_data.csv')
df_export.to_csv(save_path, index=False, encoding='utf-8-sig')

print("-" * 50)
print(f"✅ 처분 완료(학습용) 데이터: {len(df_export[df_export['학습데이터여부']=='Y'])} 건 (Train/Valid/Test 분할 완료)")
print(f"✅ 운용 중(예측용) 데이터 : {len(df_export[df_export['학습데이터여부']=='N'])} 건")
print(f"💾 최종 파일 저장 완료: {save_path}")
print("-" * 50)