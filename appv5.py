import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import re

# --- 1. データの読み込み ---
@st.cache_resource
def load_all_resources():
    try:
        with open('real_estate_ai_v4.pkl', 'rb') as f:
            ai_data = pickle.load(f)
        town_list = joblib.load('town_mapping_v4.joblib')
        return ai_data, town_list
    except Exception as e:
        st.error(f"データ読込失敗: {e}")
        return None, None

ai_res, town_options = load_all_resources()

# --- 2. デザイン ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""<style>header[data-testid="stHeader"] { visibility: hidden; } .stApp { background-color: white; }
    .result-card { padding: 25px; border: 1px solid #e2e8f0; border-radius: 15px; background-color: #f8fafc; margin-top: 20px; }
    .main-price { font-size: 32px; font-weight: bold; color: #1e3a8a; }
</style>""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if ai_res and town_options:
    model = ai_res['model']
    model_columns = ai_res['columns'] # AIが知っている全カラム名
    
    # 【ここがポイント】AIが学習した「地点_...」のカラムだけを抽出
    ai_known_locations = [c.replace('地点_', '') for c in model_columns if c.startswith('地点_')]
    
    # データを整理（AIが知っている正確な名前だけを使用）
    df_towns = pd.DataFrame(ai_known_locations, columns=['full_address'])
    df_towns['ward'] = df_towns['full_address'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1) if '区' in x else "不明")
    df_towns['short_name'] = df_towns['full_address'].apply(lambda x: re.sub(r'^東京都.*?区', '', x))
    
    # UI表示
    ward_list = sorted(df_towns['ward'].unique())
    selected_ward = st.selectbox("1. 区を選択してください", options=ward_list)
    
    filtered_df = df_towns[df_towns['ward'] == selected_ward].sort_values('short_name')
    display_map = dict(zip(filtered_df['full_address'], filtered_df['short_name']))
    
    selected_loc = st.selectbox("2. 地点を選択してください", 
                                options=filtered_df['full_address'].tolist(), 
                                format_func=lambda x: display_map.get(x))

    col1, col2 = st.columns(2)
    with col1: area = st.number_input("専有面積 (㎡)", value=60.0, step=0.1)
    with col2: year_built = st.number_input("築年 (西暦)", min_value=1970, max_value=2026, value=2015)

    if st.button("AI精密査定を実行"):
        # AI入力用データ
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        input_df['area'] = area
        input_df['age'] = 2026 - year_built
        
        # 選択された地点のカラム名を100%一致させてスイッチON
        target_col = f'地点_{selected_loc}'
        
        if target_col in model_columns:
            input_df[target_col] = 1
            predicted_price = model.predict(input_df)[0]
            
            # 結果表示（指示通りカッコなし）
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f"### 📍 {selected_ward} {display_map[selected_loc]}")
            st.markdown(f'<div class="main-price">標準AI査定価格: {int(predicted_price):,} 円</div>', unsafe_allow_html=True)
            
            st.divider()
            st.write("#### 💎 ブランドマンション・プレミアム査定")
            c1, c2 = st.columns(2)
            with c1: st.write("**メジャー7価格**"); st.write(f"### {int(predicted_price * 1.25):,} 円")
            with c2: st.write("**ランドマーク価格**"); st.write(f"### {int(predicted_price * 1.15):,} 円")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # 万が一不一致があればエラーを出す
            st.error("AI内部の地点名と一致しません。学習データを確認してください。")
