import pandas as pd
import numpy as np
import os
import sys
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split

# 루트에서 실행하지 않을 경우를 대비해 로컬 모듈을 임포트할 수 있도록 src를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 이전 단계에서 임포트
try:
    from src.data.preprocess_and_split import load_data, preprocess_data, TARGET_COL, RANDOM_STATE
except ImportError:
    # src/data에서 직접 실행하는 경우를 위한 폴백
    from preprocess_and_split import load_data, preprocess_data, TARGET_COL, RANDOM_STATE

OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'

def get_sampling_strategies():
    """샘플링 전략 사전(dictionary)을 반환합니다."""
    return {
        'SMOTE': SMOTE(random_state=RANDOM_STATE),
        'SMOTE_Tomek': SMOTETomek(random_state=RANDOM_STATE),
        'SMOTE_ENN': SMOTEENN(random_state=RANDOM_STATE)
    }

def print_class_distribution(y, name="Dataset"):
    """클래스별 개수와 비율을 출력합니다."""
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"\n[{name}] Class Distribution:")
    for label, count in counts.items():
        ratio = count / total
        print(f"  Class {label}: {count} ({ratio:.2%})")

def save_resampled_data(X_res, y_res, method_name):
    """샘플링된 X와 y를 CSV로 저장합니다."""
    # 디렉토리가 존재하는지 확인
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # X 저장
    x_path = os.path.join(OUTPUT_DIR, f"X_train_{method_name.lower()}.csv")
    X_res.to_csv(x_path, index=False)
    
    # y 저장
    y_path = os.path.join(OUTPUT_DIR, f"y_train_{method_name.lower()}.csv")
    y_res.to_csv(y_path, index=False)
    
    print(f"  Saved to: {x_path} & {y_path}")

def main():
    print("=== Starting Sampling Pipeline ===")
    
    # 1. 로드 및 전처리 (로직 재사용)
    df = load_data()
    df_processed = preprocess_data(df)
    
    # 2. 분할 (preprocess_and_split.py와 동일한 로직)
    # 타겟이 없는 행 제거 (레이블이 없는 테스트 세트)
    df_clean = df_processed.dropna(subset=[TARGET_COL])
    
    y = df_clean[TARGET_COL]
    X = df_clean.drop(columns=[TARGET_COL, 'dataset_source', 'id'], errors='ignore')
    
    # 분할 크기 결정
    total_samples = len(X)
    if total_samples == 5000:
        test_size_val = 750
    else:
        test_size_val = 0.15
        
    print(f"Splitting data (Total: {total_samples}, Test Size: {test_size_val})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size_val, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    print(f"Original X_train shape: {X_train.shape}")
    print_class_distribution(y_train, "원본 학습 데이터")
    
    # 3. 샘플링 전략 적용
    strategies = get_sampling_strategies()
    
    for name, sampler in strategies.items():
        print(f"\n--- Applying {name} ---")
        try:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            
            # 확인을 위해 출력
            print_class_distribution(y_res, f"샘플링됨 ({name})")
            
            # 저장
            save_resampled_data(X_res, y_res, name)
            
        except Exception as e:
            print(f"ERROR applying {name}: {e}")

    # 4. 일관성을 위해 원본 테스트 세트 저장
    # 베이스라인 비교를 위해 원본 학습 세트도 저장해야 함
    print("\n--- 원본 세트 저장 중 ---")
    save_resampled_data(X_train, y_train, "original")
    
    # X_test, y_test 저장 (한 번만 수행하면 됨)
    x_test_path = os.path.join(OUTPUT_DIR, "X_test.csv")
    y_test_path = os.path.join(OUTPUT_DIR, "y_test.csv")
    X_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print(f"  Saved Test Set to: {x_test_path} & {y_test_path}")
    
    print("\n=== Sampling Pipeline Completed ===")

if __name__ == "__main__":
    main()
