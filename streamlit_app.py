import streamlit as st
import pandas as pd
import plotly.express as px

# 주택 데이터 CSV 파일에서 데이터를 읽어옵니다.
data = pd.read_csv('kc_house_data.csv')

# 침실 수 목록을 가져옵니다.
categories = sorted(data['bedrooms'].unique())

# Streamlit 앱의 제목을 설정합니다.
st.title('침실 수별 주택 가격 시각화')

# 침실 수 선택 위젯을 생성합니다.
selected_bedrooms = st.selectbox('침실 수 선택', categories)

# 선택한 침실 수에 해당하는 데이터를 필터링합니다.
filtered_data = data[data['bedrooms'] == selected_bedrooms]

# 필터링된 데이터의 가격 분포를 시각화합니다.
fig = px.histogram(filtered_data, x='price', nbins=30,
                   title=f'{selected_bedrooms}개 침실 주택의 가격 분포',
                   labels={'price': '가격 (달러)', 'count': '주택 수'})

# 그래프를 Streamlit에 표시합니다.
st.plotly_chart(fig)
