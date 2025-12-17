import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
from catboost import CatBoostClassifier

# 한글 컬럼명 및 단위 매핑
COLUMN_KOREAN_MAP = {
    # 통화 시간 (분) - 월간 기준
    'total_day_minutes': '주간 통화 시간 (분/월간)',
    'total_eve_minutes': '저녁 통화 시간 (분/월간)',
    'total_night_minutes': '야간 통화 시간 (분/월간)',
    'total_intl_minutes': '국제 통화 시간 (분/월간)',

    # 통화 요금 (Charge) - 월간 기준
    'total_day_charge': '주간 통화 요금 (월간)',
    'total_eve_charge': '저녁 통화 요금 (월간)',
    'total_night_charge': '야간 통화 요금 (월간)',
    'total_intl_charge': '국제 통화 요금 (월간)',

    # 통화 횟수 (회) - 월간 기준
    'total_day_calls': '주간 통화 횟수 (회/월간)',
    'total_eve_calls': '저녁 통화 횟수 (회/월간)',
    'total_night_calls': '야간 통화 횟수 (회/월간)',
    'total_intl_calls': '국제 통화 횟수 (회/월간)',
    'number_customer_service_calls': '고객센터 전화 (회/월간)',

    # 기타
    'account_length': '가입 유지 기간 (일)',
    'number_vmail_messages': '음성메일 수 (개)',
    'international_plan': '국제전화 플랜 가입 여부',
    'voice_mail_plan': '음성메일 플랜 가입 여부',
    'area_code': '지역 코드',
    'state': '주(State)'
}

# --- 1. 페이지 설정 및 모델 로드 ---
st.set_page_config(
    page_title="고객 이탈 예측 대시보드",
    page_icon="📊",
    layout="wide"
)

st.title("📊 고객 이탈 예측 시스템 (Churn Prediction)")
st.markdown("---")

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "churn_model.cbm")
FEATURES_PATH = os.path.join(CURRENT_DIR, "features.pkl")

# 모델 및 데이터 로드 (캐싱 사용)
@st.cache_resource
def load_model_and_features():
    # 1. 모델 로드
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    # 2. Feature Names 로드
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
        
    return model, feature_names

# 로딩 중 표시
with st.spinner("모델 및 데이터를 로딩 중입니다..."):
    # 파일 존재 여부 확인
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error("모델 파일 또는 피처 파일이 없습니다. 'save_model.py'를 먼저 실행해주세요.")
        st.stop()
        
    model, feature_names = load_model_and_features()

# 평균값 로드 (캐싱)
MEAN_VALUES_PATH = os.path.join(CURRENT_DIR, "mean_values.pkl")
@st.cache_resource
def load_mean_values():
    if not os.path.exists(MEAN_VALUES_PATH):
        return None
    with open(MEAN_VALUES_PATH, 'rb') as f:
        return pickle.load(f)

mean_values = load_mean_values()

# --- 2. 사이드바: 사용자 입력 (User Input) ---

def smart_number_input(key, label, min_value, value, max_value=None):
    """
    컬럼명(key)에 따라 step과 format을 동적으로 설정하는 래퍼 함수
    """
    step = 1.0 # 기본값
    fmt = None # 기본값
    
    # 1. 횟수/개수/기간 (정수형)
    if any(x in key for x in ['calls', 'messages', 'account_length']):
        step = 1
        fmt = "%d"
        # 정수형 입력이므로 value와 min_value도 정수로 변환 (안전장치)
        value = int(value)
        min_value = int(min_value)
        
    # 2. 시간 데이터 (분 단위) - 10분 단위 이동
    elif 'minutes' in key:
        step = 10.0
        fmt = "%.1f"
        
    # 3. 요금 데이터 (달러/원) - 1.0 단위 이동
    elif 'charge' in key:
        step = 1.0
        fmt = "%.2f"
        
    return st.number_input(label, min_value=min_value, value=value, max_value=max_value, step=step, format=fmt)

st.sidebar.header("📝 고객 정보 입력")

# 입력값을 저장할 딕셔너리
user_input = {}

# 그룹 1: 기본 가입 정보 (Demographics & Plans)
with st.sidebar.expander("👤 기본 가입 정보", expanded=True):
    # State 선택
    state_options = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN'] # 예시
    user_input['state'] = st.selectbox(COLUMN_KOREAN_MAP['state'], state_options)
    
    user_input['account_length'] = smart_number_input('account_length', COLUMN_KOREAN_MAP['account_length'], min_value=1, value=100)
    user_input['area_code'] = st.selectbox(COLUMN_KOREAN_MAP['area_code'], ["area_code_408", "area_code_415", "area_code_510"])
    
    # Yes/No 입력 -> 1/0 변환
    intl_plan = st.radio(COLUMN_KOREAN_MAP['international_plan'], ["Yes", "No"])
    user_input['international_plan'] = 1 if intl_plan == "Yes" else 0
    
    vmail_plan = st.radio(COLUMN_KOREAN_MAP['voice_mail_plan'], ["Yes", "No"])
    user_input['voice_mail_plan'] = 1 if vmail_plan == "Yes" else 0
    
    user_input['number_vmail_messages'] = smart_number_input('number_vmail_messages', COLUMN_KOREAN_MAP['number_vmail_messages'], min_value=0, value=0)

# 그룹 2: 통화량 정보 (Call Usage)
with st.sidebar.expander("📞 통화 사용량 정보", expanded=False):
    st.markdown("**주간 (Day)**")
    user_input['total_day_minutes'] = smart_number_input('total_day_minutes', COLUMN_KOREAN_MAP['total_day_minutes'], min_value=0.0, value=150.0)
    user_input['total_day_calls'] = smart_number_input('total_day_calls', COLUMN_KOREAN_MAP['total_day_calls'], min_value=0, value=100)
    user_input['total_day_charge'] = smart_number_input('total_day_charge', COLUMN_KOREAN_MAP['total_day_charge'], min_value=0.0, value=25.0)
    
    st.markdown("**저녁 (Evening)**")
    user_input['total_eve_minutes'] = smart_number_input('total_eve_minutes', COLUMN_KOREAN_MAP['total_eve_minutes'], min_value=0.0, value=200.0)
    user_input['total_eve_calls'] = smart_number_input('total_eve_calls', COLUMN_KOREAN_MAP['total_eve_calls'], min_value=0, value=100)
    user_input['total_eve_charge'] = smart_number_input('total_eve_charge', COLUMN_KOREAN_MAP['total_eve_charge'], min_value=0.0, value=17.0)
    
    st.markdown("**야간 (Night)**")
    user_input['total_night_minutes'] = smart_number_input('total_night_minutes', COLUMN_KOREAN_MAP['total_night_minutes'], min_value=0.0, value=200.0)
    user_input['total_night_calls'] = smart_number_input('total_night_calls', COLUMN_KOREAN_MAP['total_night_calls'], min_value=0, value=100)
    user_input['total_night_charge'] = smart_number_input('total_night_charge', COLUMN_KOREAN_MAP['total_night_charge'], min_value=0.0, value=9.0)
    
    st.markdown("**국제 (Intl)**")
    user_input['total_intl_minutes'] = smart_number_input('total_intl_minutes', COLUMN_KOREAN_MAP['total_intl_minutes'], min_value=0.0, value=10.0)
    user_input['total_intl_calls'] = smart_number_input('total_intl_calls', COLUMN_KOREAN_MAP['total_intl_calls'], min_value=0, value=3)
    user_input['total_intl_charge'] = smart_number_input('total_intl_charge', COLUMN_KOREAN_MAP['total_intl_charge'], min_value=0.0, value=2.7)

# 그룹 3: 기타 고객 서비스
with st.sidebar.expander("🎧 고객 서비스 (CS)", expanded=False):
    user_input['number_customer_service_calls'] = smart_number_input('number_customer_service_calls', COLUMN_KOREAN_MAP['number_customer_service_calls'], min_value=0, max_value=20, value=1)


# 입력 데이터를 DataFrame으로 변환
input_df = pd.DataFrame([user_input])

# 중요: 학습된 모델의 Feature 순서와 동일하게 정렬
# 없는 컬럼은 0으로 채우고, 불필요한 컬럼은 제거
# (현재 예시 UI에서는 모든 피처를 다 받지 않았을 수 있으므로 안전장치 추가)
for col in feature_names:
    if col not in input_df.columns:
        # UI에서 입력받지 않은 값이 있다면 기본값 0 처리 (혹은 적절한 값)
        input_df[col] = 0

# 최종적으로 Feature Names 순서대로 정렬
input_df = input_df[feature_names]


# --- 3. 메인 화면: 예측 결과 ---
# 예측 수행
# predict_proba 반환값은 [class0_prob, class1_prob]
prob_churn = model.predict_proba(input_df)[0][1] # 이탈(1) 확률
prob_percent = prob_churn * 100

# --- 3. 메인 화면: 예측 결과 및 분석 ---

# 1. Hero Section: 핵심 지표 (Key Metrics)
st.markdown("### 🔑 핵심 지표 요약")
m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 3]) # 1:1:1:3 비율 (우측 여백)

# 1-1. 예상 월 요금 (Total Bill)
total_bill = (
    user_input['total_day_charge'] + 
    user_input['total_eve_charge'] + 
    user_input['total_night_charge'] + 
    user_input['total_intl_charge']
)
m_col1.metric("예상 월 요금", f"${total_bill:.2f}")

# 1-2. 가입 기간 (Tenure)
tenure = int(user_input['account_length'])
m_col2.metric("가입 기간", f"{tenure}일")

# 1-3. CS 요청 (CS Calls)
cs_calls = int(user_input['number_customer_service_calls'])
if cs_calls >= 3:
    m_col3.metric("CS 요청", f"{cs_calls}회", delta="-주의", delta_color="inverse")
else:
    m_col3.metric("CS 요청", f"{cs_calls}회", delta="정상")

st.markdown("---")

# 2. 메인 컨텐츠 (2단 레이아웃)
# 왼쪽: 진단 및 처방 / 오른쪽: 심층 분석
col1, col2 = st.columns([1, 1.2])

# 🟢 왼쪽 컬럼: 진단 & 처방
with col1:
    st.subheader("1. 상태 진단 (Diagnosis)")
    
    # 이탈 확률에 따른 신호등 시스템
    if prob_percent <= 40:
        st.success(f"✅ 안정권 (Safe)\n\n이탈 확률: {prob_percent:.1f}%")
        st.caption("안정적인 장기 충성 고객입니다.")
    elif prob_percent <= 70:
        st.info(f"🟡 관심 필요 (Attention)\n\n이탈 확률: {prob_percent:.1f}%")
        st.caption("세심한 케어가 필요한 단계입니다.")
    elif prob_percent <= 85:
        st.warning(f"🟠 이탈 주의 (Warning)\n\n이탈 확률: {prob_percent:.1f}%")
        st.caption("강력한 이탈 징후가 감지되었습니다.")
    else:
        st.error(f"🚨 위험 (Critical)\n\n이탈 확률: {prob_percent:.1f}%")
        st.caption("즉각적인 조치가 필요한 위험 고객입니다!")

    st.markdown("---")

    # AI 맞춤 대응 전략 (Action Plan)
    st.subheader("2. 대응 전략 (Action Plan)")
    
    # 카드 스타일 CSS
    st.markdown("""
    <style>
    .action-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        color: white;
        font-weight: bold;
    }
    .urgent { background-color: #ff4b4b; }
    .warning { background-color: #ffa726; }
    .suggestion { background-color: #2196f3; }
    .safe { background-color: #66bb6a; }
    </style>
    """, unsafe_allow_html=True)
    
    actions = []
    
    if prob_percent > 70:
        actions.append({
            "type": "urgent", "icon": "🚨", "title": "방어 코드 발동",
            "msg": "즉시 해피콜을 진행하여 불만 사항을 청취하세요."
        })
        
    if user_input['number_customer_service_calls'] > 3:
        actions.append({
            "type": "warning", "icon": "📞", "title": "불만 케어 필요",
            "msg": f"최근 CS 문의가 많습니다 ({user_input['number_customer_service_calls']}회). 우선 상담하세요."
        })
        
    # 평균값 로드 확인 후 로직 수행
    if mean_values:
        mean_intl = mean_values.get('total_intl_minutes', 10)
        if (user_input['total_intl_minutes'] > mean_intl) and (user_input['international_plan'] == 0):
             actions.append({
                "type": "suggestion", "icon": "💡", "title": "업셀링 기회",
                "msg": "국제전화 사용량이 많습니다. 전용 플랜을 제안하세요."
            })
    
    if prob_percent < 30 and len(actions) == 0:
        actions.append({
            "type": "safe", "icon": "✅", "title": "관계 강화",
            "msg": "장기 혜택 안내 문자를 발송하세요."
        })
        
    for action in actions:
        st.markdown(f"""
        <div class="action-card {action['type']}">
            <div>{action['icon']} {action['title']}</div>
            <div style="font-size: 0.8em; opacity: 0.9; font-weight: normal;">{action['msg']}</div>
        </div>
        """, unsafe_allow_html=True)


# 🔵 오른쪽 컬럼: 심층 분석
with col2:
    st.subheader("3. 심층 분석 (Deep Dive)")
    
    st.markdown("##### 📌 주요 이탈 요인 (Top 7)")
    
    # Feature Importance
    importances = model.get_feature_importance()
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp['Feature_KR'] = feature_imp['Feature'].map(COLUMN_KOREAN_MAP).fillna(feature_imp['Feature'])
    
    # Top 7 추출
    feature_imp = feature_imp.sort_values(by='Importance', ascending=True).tail(7)
    
    fig_bar = px.bar(
        feature_imp, x='Importance', y='Feature_KR', orientation='h',
        color='Importance', color_continuous_scale='Reds'
    )
    
    fig_bar.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis={'categoryorder':'total ascending'},
        xaxis_title=None, yaxis_title=None,
        coloraxis_showscale=False
    )
    fig_bar.update_traces(hovertemplate='<b>%{y}</b><br>중요도: %{x:.2f}<extra></extra>')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    
    # Radar Chart
    st.markdown("##### 🕸️ 고객 성향 비교 (Radar Chart)")
    
    if mean_values:
        chart_features = [
            'total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 
            'total_intl_minutes', 'number_customer_service_calls'
        ]
        params_korean = ['주간 통화', '저녁 통화', '야간 통화', '국제 통화', 'CS 전화']
        
        current_vals = [user_input.get(f, 0) for f in chart_features]
        avg_vals = [mean_values.get(f, 0) for f in chart_features]
        
        # Scale Normalization logic (simple max-based)
        norm_current = []
        norm_avg = []
        
        for c, a in zip(current_vals, avg_vals):
            axis_max = max(c, a) * 1.5 if max(c, a) > 0 else 1.0
            norm_current.append(c / axis_max)
            norm_avg.append(a / axis_max)
            
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_current, theta=params_korean, fill='toself', name='현재 고객',
            line_color='blue'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_avg, theta=params_korean, fill='toself', name='평균 고객',
            line_color='gray', opacity=0.5
        ))
        
        fig_radar.update_layout(
            height=350,
            margin=dict(l=40, r=40, t=40, b=40),
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True,
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("데이터 부족으로 성향 분석 차트를 표시할 수 없습니다.")
