import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 설정
DATA_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\03_resampled'
OUTPUT_DIR = r'c:\Workspaces\SKN22-2nd-4Team\data\04_results'
RANDOM_STATE = 42

def load_dataset(dataset_name):
    """특정 샘플링 전략의 X_train, y_train과 공통 테스트 세트를 로드합니다."""
    # 규칙에 따라 파일명 생성: X_train_smote.csv
    train_x_path = os.path.join(DATA_DIR, f"X_train_{dataset_name.lower()}.csv")
    train_y_path = os.path.join(DATA_DIR, f"y_train_{dataset_name.lower()}.csv")
    
    test_x_path = os.path.join(DATA_DIR, "X_test.csv")
    test_y_path = os.path.join(DATA_DIR, "y_test.csv")
    
    if not os.path.exists(train_x_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found at {train_x_path}")
        
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel() # 1차원 배열 보장
    X_test = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    
    return X_train, y_train, X_test, y_test

def get_models(use_class_weights=False):
    """모델 사전(dict)을 반환합니다. SVM, LR, ANN은 스케일링을 포함한 파이프라인으로 구성됩니다."""
    
    models = {}
    
    # 가중치 설정
    # 참고: XGBoost는 scale_pos_weight = count(negative) / count(positive)를 사용함
    # 하지만 이 루프의 단순성을 위해 라이브러리가 'balanced' 문자열을 지원하지 않으면 수동 계산이 필요할 수 있음.
    # LightGBM/CatBoost/RF/SVC/LR은 'balanced'를 지원함.
    
    # 필요한 경우 XGBoost/CatBoost에 대한 단순 비율 추정 (약 850 비이탈 / 150 이탈 ~ 5.6)
    # 하지만 지원되는 경우 'balanced' 문자열을 사용하거나 수동 로직을 고수함.
    
    cw_param = 'balanced' if use_class_weights else None
    
    # 1. 트리/앙상블 모델
    # DT
    models['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=cw_param)
    
    # RF
    models['RF'] = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=cw_param)
    
    # XGBoost (보통 'balanced' 문자열을 직접 지원하지 않으며, scale_pos_weight가 필요함)
    # 필요한 경우 대략적인 scale_pos_weight를 계산하거나, 엄격하지 않은 경우 기본값으로 둠.
    # 이 벤치마크에서는 고정 가중치 ~5.9를 사용함 (총 3333 / 483 이탈 ~ 14.5% 이탈. Neg/Pos = 2850/483 ~ 5.9)
    xgb_weight = 5.9 if use_class_weights else 1
    models['XGBoost'] = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, use_label_encoder=False, scale_pos_weight=xgb_weight)
    
    # LightGBM
    models['LightGBM'] = LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, class_weight=cw_param)
    
    # CatBoost (범용 벤치마크)
    # CatBoost는 auto_class_weights='Balanced'를 지원함
    cb_weights = 'Balanced' if use_class_weights else None
    models['CatBoost'] = CatBoostClassifier(verbose=0, random_state=RANDOM_STATE, auto_class_weights=cb_weights)
    
    # 2. 거리/경사 기반 모델 (스케일링 필요)
    # ANN (MLP는 sklearn < 0.24 등에서 class_weight를 엄격하게 지원하지 않음? 보통 class_weight 파라미터가 없음)
    # ANN의 경우 가중치를 생략하거나 그대로 둠.
    models['ANN'] = make_pipeline(StandardScaler(), MLPClassifier(random_state=RANDOM_STATE, max_iter=1000))
    
    # SVM
    models['SVM'] = make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE, class_weight=cw_param))
    
    # LR
    models['LR'] = make_pipeline(StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight=cw_param))
    
    return models

def load_raw_data_for_catboost():
    """
    원본 데이터를 로드하고 벤치마크 데이터셋이 생성된 것과 동일하게 분할하여(random_state=42),
    CatBoost용 범주형 특성을 보존합니다.
    """
    raw_path = os.path.join(r'c:\Workspaces\SKN22-2nd-4Team\data\01_raw', "train.csv")
    df = pd.read_csv(raw_path)
    
    # 간단한 전처리 (Yes/No -> 1/0)
    for col in ['international_plan', 'voice_mail_plan']:
        if col in df.columns:
            df[col] = (df[col] == 'yes').astype(int)
    
    # 타겟
    X = df.drop('churn', axis=1)
    y = df['churn'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # 분할 (X_train_original.csv 생성에 사용된 분할과 일치해야 함)
    # 시드 42를 사용한 표준 80/20 분할이 사용되었다고 가정함.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    return X_train, y_train, X_test, y_test, cat_features

def evaluate_models_on_dataset(dataset_name):
    """
    특정 데이터셋(예: 'smote')에 대해 모든 모델을 훈련하고 평가합니다.
    반환값: 결과 DataFrame 및 ROC 데이터 딕셔너리 리스트.
    """
    print(f"\n[{dataset_name.upper()}] Loading data...")
    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    except FileNotFoundError as e:
        print(e)
        return None, None

    # (SMOTE를 대체하기 위해 요청된 대로) 원본 데이터셋에 대해서만 클래스 가중치 적용
    use_weights = (dataset_name == 'original')
    models = get_models(use_class_weights=use_weights)
    
    results = []
    roc_data = {} # 모델별로 저장
    
    print(f"[{dataset_name.upper()}] Training {len(models)} models (Class Weights={use_weights})...")
    
    for name, model in models.items():
        # (사용자 요청에 따라) 원본 데이터셋에서 네이티브 범주형 지원을 사용하기 위해
        # CatBoost에 대한 특별 처리
        if name == 'CatBoost' and dataset_name == 'original':
            print("  > CatBoost: 네이티브 범주형 처리를 위해 원본 데이터로 전환 중...")
            X_train_cb, y_train_cb, X_test_cb, y_test_cb, cat_features = load_raw_data_for_catboost()
            
            # cat_features 및 클래스 가중치를 사용하여 다시 초기화
            model = CatBoostClassifier(
                verbose=0, 
                random_state=RANDOM_STATE,
                cat_features=cat_features,
                auto_class_weights='Balanced'
            )
            model.fit(X_train_cb, y_train_cb)
            
            y_pred = model.predict(X_test_cb)
            y_prob = model.predict_proba(X_test_cb)[:, 1]
            
            # 지표 (레이블이 동일해야 하는 CatBoost 특정 테스트 세트 사용)
            p = precision_score(y_test_cb, y_pred, zero_division=0)
            r = recall_score(y_test_cb, y_pred, zero_division=0)
            f1 = f1_score(y_test_cb, y_pred, zero_division=0)
            roc = roc_auc_score(y_test_cb, y_prob)
            
            # 시각화를 위한 ROC 데이터 (y_test_cb 사용)
            fpr, tpr, _ = roc_curve(y_test_cb, y_prob)
            roc_data[name] = (fpr, tpr, roc)
            
        else:
            # 표준 경로
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            p = precision_score(y_test, y_pred, zero_division=0)
            r = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc = roc_auc_score(y_test, y_prob)
            
            # 시각화를 위한 ROC 데이터 (표준 y_test 사용)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, roc)
        
        results.append({
            'Model': name,
            'Precision': p,
            'Recall': r,
            'F1-score': f1,
            'ROC AUC': roc
        })
        
        print(f"  > {name}: F1={f1:.4f}, AUC={roc:.4f}")
        
    df_results = pd.DataFrame(results).set_index('Model')
    return df_results, roc_data

def plot_roc_curves(roc_data, dataset_name):
    """모든 모델의 ROC 곡선을 하나의 그림에 그리고 저장합니다."""
    plt.figure(figsize=(10, 8))
    
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name.upper()} Dataset')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(OUTPUT_DIR, f"roc_curve_{dataset_name.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  ROC Plot saved to: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 이전 단계의 4가지 데이터셋 변체
    datasets = ['original', 'smote', 'smote_tomek', 'smote_enn']
    
    all_benchmarks_completed = True
    
    for ds_name in datasets:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {ds_name}")
        print(f"{'='*40}")
        
        results_df, roc_data = evaluate_models_on_dataset(ds_name)
        
        if results_df is not None:
            # 표 저장
            csv_path = os.path.join(OUTPUT_DIR, f"benchmark_{ds_name.lower()}.csv")
            results_df.to_csv(csv_path)
            print(f"  Table saved to: {csv_path}")
            
            # 로그 출력을 위해 프린트
            print(results_df)
            
            # ROC 도식화
            plot_roc_curves(roc_data, ds_name)
        else:
            all_benchmarks_completed = False

    if all_benchmarks_completed:
        print("\nAll benchmarks completed successfully.")
    else:
        print("\nSome benchmarks failed due to missing files.")

if __name__ == "__main__":
    main()
