import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import re

# --- 1. データの読み込み（新・1ファイル直接ロード方式） ---
@st.cache_resource
def load_all_resources():
    try:
        # 分割読み込みを廃止し、新しいv4ファイルを直接読み込みます
        with open('real_estate_ai_v4.pkl', 'rb') as f:
            ai_data = pickle.load(f)
        town_list = joblib.load('town_mapping_v4.joblib')
        return ai_data, town_list
    except Exception as e:
        st.error(f"データ読込失敗: {e}。pklとjoblibがGitHubに存在するか確認してください。")
        return None, None

ai_res, town_options = load_all_resources()

# --- 2. デザイン設定 ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""<style>
    header[data-testid="stHeader"] { visibility: hidden; }
    .stApp { background-color: white; }
    .result-card { padding: 25px; border: 1px solid #e2e8f0; border-radius: 15px; background-color: #f8fafc; margin-top: 20px; }
    .main-price { font-size: 32px; font-weight: bold; color: #1e3a8a; }
    .premium-box { background: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #b45309; margin-top: 10px; }
</style>""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if ai_res and town_options:
    model = ai_res['model']
    model_columns = ai_res['columns']
    
    # 地点データを整理
    df_towns = pd.DataFrame(town_options, columns=['full_address'])
    # 区名の抽出
    df_towns['ward'] = df_towns['full_address'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    # 表示用町名（カッコを含まない）
    df_towns['short_name'] = df_towns['full_address'].apply(lambda x: re.sub(r'^東京都.*?区', '', x))
    
    # --- UI ---
    ward_list = sorted(df_towns['ward'].unique())
    selected_ward = st.selectbox("1. 区を選択してください", options=ward_list)
    
    # 選択された区の地点を表示
    filtered_df = df_towns[df_towns['ward'] == selected_ward]
    display_map = dict(zip(filtered_df['full_address'], filtered_df['short_name']))
    
    selected_loc = st.selectbox(
        "2. 地点を選択してください", 
        options=filtered_df['full_address'].tolist(),
        format_func=lambda x: display_map.get(x)
    )

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("専有面積 (㎡)", value=60.0, step=0.1)
    with col2:
        year_built = st.number_input("築年 (西暦)", min_value=1970, max_value=2026, value=2015)

    if st.button("AI精密査定を実行"):
        # 入力データの準備
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        input_df['area'] = area
        input_df['age'] = 2026 - year_built
        
        # 学習時と同じ「地点_東京都〇〇区△△丁目」という形式で照合
        target_col = f'地点_{selected_loc}'
        
        if target_col in model_columns:
            input_df[target_col] = 1
            predicted_price = model.predict(input_df)[0]
            
            # 結果表示
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f"### 📍 {selected_ward} {display_map[selected_loc]}")
            st.markdown(f'<div class="main-price">標準AI査定価格: {int(predicted_price):,} 円</div>', unsafe_allow_html=True)
            
            st.divider()
            st.write("#### 💎 ブランドマンション・プレミアム査定")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="premium-box">', unsafe_allow_html=True)
                st.write("**メジャー7・トップブランド**")
                st.write(f"### {int(predicted_price * 1.25):,} 円")
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="premium-box">', unsafe_allow_html=True)
                st.write("**地域ランドマーク・高仕様**")
                st.write(f"### {int(predicted_price * 1.15):,} 円")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error(f"地点 '{selected_loc}' がモデルに見つかりません。")