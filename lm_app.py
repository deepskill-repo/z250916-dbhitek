import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 주택 데이터 CSV 파일에서 데이터를 읽어옵니다.
data = pd.read_csv('kc_house_data.csv')

# Streamlit 앱의 제목을 설정합니다.
st.title('선형 회귀 분석: 주택 가격 예측')

# 설명 추가
st.write("이 앱은 선택한 변수들을 사용하여 주택 가격을 예측하는 선형 회귀 모델을 만듭니다.")

# 수치형 변수들만 선택 (ID, date 등 제외)
numeric_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                   'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                   'sqft_living15', 'sqft_lot15']

# X 변수 선택 위젯 (기본값: sqft_living)
st.subheader('독립 변수 선택')
selected_variables = st.multiselect(
    'X 변수를 선택하세요 (여러 개 선택 가능)',
    numeric_columns,
    default=['sqft_living']
)

# 변수가 선택되었을 때만 분석 실행
if selected_variables:
    # 데이터 준비
    X = data[selected_variables]
    y = data['price']
    
    # 결측값 제거
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    # 선형 회귀 모델 훈련
    model = LinearRegression()
    model.fit(X_clean, y_clean)
    
    # 예측값 계산
    y_pred = model.predict(X_clean)
    
    # 모델 성능 지표 계산
    r2 = r2_score(y_clean, y_pred)
    mse = mean_squared_error(y_clean, y_pred)
    rmse = np.sqrt(mse)
    
    # 성능 지표 표시
    st.subheader('모델 성능')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R-squared", f"{r2:.3f}")
    with col2:
        st.metric("RMSE", f"${rmse:,.0f}")
    with col3:
        st.metric("데이터 개수", f"{len(y_clean):,}")
    
    # 회귀 계수 표시
    st.subheader('회귀 계수')
    coef_df = pd.DataFrame({
        '변수': selected_variables,
        '계수': model.coef_
    })
    st.dataframe(coef_df)
    
    # 절편 별도 표시
    st.write(f"**절편 (Intercept):** {model.intercept_:,.2f}")
    
    # 실제값 vs 예측값 산점도
    st.subheader('실제값 vs 예측값')
    
    # 산점도 생성
    fig = go.Figure()
    
    # 산점도 추가
    fig.add_trace(go.Scatter(
        x=y_clean,
        y=y_pred,
        mode='markers',
        name='예측값',
        opacity=0.6,
        marker=dict(size=4)
    ))
    
    # 완벽한 예측선 (y=x) 추가
    min_val = min(y_clean.min(), y_pred.min())
    max_val = max(y_clean.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='완벽한 예측선 (y=x)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='실제 주택 가격 vs 예측 주택 가격',
        xaxis_title='실제 가격 (달러)',
        yaxis_title='예측 가격 (달러)',
        width=700,
        height=500
    )
    
    st.plotly_chart(fig)
    
    # 잔차 분석 (추가 정보)
    st.subheader('잔차 분석')
    residuals = y_clean - y_pred
    
    fig_residuals = px.scatter(
        x=y_pred, 
        y=residuals,
        title='잔차 vs 예측값',
        labels={'x': '예측값', 'y': '잔차 (실제값 - 예측값)'}
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    
    st.plotly_chart(fig_residuals)
    
else:
    st.warning('최소 하나의 독립 변수를 선택해주세요!')
