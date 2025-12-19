import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 설정
RAW_DATA_PATH = r'c:\Workspaces\SKN22-2nd-4Team\data\01_raw'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
TARGET_COL = 'churn'
RANDOM_STATE = 42

def load_data():
    """학습 및 테스트 데이터를 로드하고 병합합니다."""
    train_path = os.path.join(RAW_DATA_PATH, TRAIN_FILE)
    test_path = os.path.join(RAW_DATA_PATH, TEST_FILE)
    
    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 테스트 세트에 타겟 변수가 있는지 확인
    if TARGET_COL not in df_test.columns:
        print(f"WARNING: '{TARGET_COL}' column missing in {TEST_FILE}. "
              "These rows cannot be used for stratified splitting or evaluation.")
        # 레이블이 없는 데이터는 지도 학습/분할에 사용할 수 없음
        # 하지만 병합을 진행하기 위해 NaN으로 채움
        df_test[TARGET_COL] = np.nan
    
    # 추적을 위한 소스 플래그 추가 (선택 사항)
    df_train['dataset_source'] = 'train'
    df_test['dataset_source'] = 'test'

    # 일관된 전처리를 위해 병합
    df_total = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    print(f"Total dataset shape after merge: {df_total.shape}")
    return df_total

def preprocess_data(df):
    """
    1. 이진 매핑 (yes/no -> 1/0)
    2. 레이블 인코딩 (state, area_code)
    """
    df_processed = df.copy()
    
    # 이진 매핑
    # 참고: 테스트 행의 경우 'churn'이 NaN일 수 있음
    binary_cols = ['international_plan', 'voice_mail_plan']
    for col in binary_cols:
        if col in df_processed.columns:
            # 필요한 경우 먼저 결측치를 채우지만, 이 컬럼들은 깨끗해야 함
            df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
    
    # 타겟을 별도로 매핑하여 NaN이 있는 경우 보존
    if TARGET_COL in df_processed.columns:
        # 'yes'/'no'만 매핑. NaN은 NaN으로 남겨둠.
        df_processed[TARGET_COL] = df_processed[TARGET_COL].map({'yes': 1, 'no': 0})

    # 범주형 특성에 대한 레이블 인코딩
    le = LabelEncoder()
    # 주(State)
    if 'state' in df_processed.columns:
        df_processed['state'] = le.fit_transform(df_processed['state'].astype(str))
        
    # 지역 번호(Area Code)
    if 'area_code' in df_processed.columns:
        df_processed['area_code'] = le.fit_transform(df_processed['area_code'].astype(str))
        
    return df_processed

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    IQR 기반으로 이상치가 포함된 행을 제거합니다.
    참고: 이 함수를 사용하면 데이터셋 크기가 줄어듭니다.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # 필터링
        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & 
            (df_clean[col] <= upper_bound)
        ]
        
    final_rows = len(df_clean)
    print(f"Outlier Removal: Removed {initial_rows - final_rows} rows.")
    return df_clean

def main():
    # 1. 로드
    df = load_data()
    
    # 2. 전처리
    df_processed = preprocess_data(df)
    
    # 3. 이상치 제거 (비활성화됨)
    APPLY_OUTLIER_REMOVAL = False
    if APPLY_OUTLIER_REMOVAL:
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['churn', 'international_plan', 'voice_mail_plan', 'state', 'area_code', 'dataset_source', 'id']
        target_cols = [c for c in numeric_cols if c not in exclude_cols]
        df_processed = remove_outliers_iqr(df_processed, target_cols)
    else:
        print("Skipping outlier removal.")

    # 4. 분할 준비
    # 층화 분할(Stratified split)을 위해 타겟이 NaN인 행은 제거해야 함
    initial_len = len(df_processed)
    df_clean_for_split = df_processed.dropna(subset=[TARGET_COL])
    dropped_rows = initial_len - len(df_clean_for_split)
    
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing target (likely from test.csv).")
        print("참고: 층화 분할에는 알려진 레이블이 필요합니다.")

    # 보조 컬럼 드롭
    drop_cols = [TARGET_COL, 'dataset_source']
    if 'id' in df_clean_for_split.columns:
        drop_cols.append('id')
        
    y = df_clean_for_split[TARGET_COL]
    X = df_clean_for_split.drop(columns=drop_cols)
    
    # 분할 크기 결정
    # 사용자 요청: 학습 4250, 테스트 750. (합계 5000)
    # test.csv에 레이블이 없는 경우 사용 가능한 데이터가 더 적을 수 있음 (총 4250개).
    
    total_samples = len(X)
    print(f"Available labeled samples for split: {total_samples}")
    
    if total_samples == 5000:
        test_size_count = 750
    elif total_samples == 4250:
        # train.csv 데이터만 있는 경우, 비율을 시뮬레이션하기 위해 85/15로 분할
        # 4250 * 0.15 = 637.5 -> 638
        test_size_count = 0.15
        print("Using 0.15 test split ratio on available 4250 samples.")
    else:
        test_size_count = 0.15 # 폴백
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_count, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # 5. 출력 확인
    print("-" * 30)
    print("Final Shapes Verification:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")
    print("-" * 30)
    
    if X_train.shape[0] == 4250 and X_test.shape[0] == 750:
        print("SUCCESS: Data split shapes match the experiment specifications (4250, 750).")
    else:
        print(f"WARNING: Data split shapes do NOT exact match (4250, 750). Received ({X_train.shape[0]}, {X_test.shape[0]}).")
        print("이유: 제공된 test.csv에 레이블이 부족하여 사용 가능한 총 데이터셋이 4250개로 줄어들었을 가능성이 높습니다.")

    # 6. 분할된 데이터 저장
    OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nSaving split data to {OUTPUT_DIR}...")
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'X_train_original.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, 'y_train_original.csv'), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, 'y_test.csv'), index=False)
    print("Files saved: X_train_original.csv, X_test.csv, y_train_original.csv, y_test.csv")



    print("Preprocessing & Splitting Completed.")
if __name__ == "__main__":
    main()