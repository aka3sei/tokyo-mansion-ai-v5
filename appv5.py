import streamlit as st
import pandas as pd
import re

# --- 1. データの読み込み（CSVから直接単価を取得） ---
@st.cache_data
def load_data():
    try:
        # アップロードされたCSVを読み込み
        df = pd.read_csv('chome_model_23ku_v3.csv')
        return df
    except Exception as e:
        st.error(f"CSV読み込み失敗: {e}")
        return None

df_price = load_data()

# --- 2. デザイン ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.title("🏙️ 23区精密エリアAI査定")

if df_price is not None:
    # 所在地（学習地点）から区名を抽出
    df_price['ward'] = df_price['学習地点'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1) if '区' in x else "その他")
    
    # UI
    ward_list = sorted(df_price['ward'].unique())
    selected_ward = st.selectbox("1. 区を選択してください", options=ward_list)
    
    # 選択された区の地点を絞り込み
    filtered_df = df_price[df_price['ward'] == selected_ward].sort_values('学習地点')
    
    # 表示用の名前（区名以降を表示）
    display_options = {row['学習地点']: row['学習地点'].split(selected_ward)[-1] for _, row in filtered_df.iterrows()}
    
    selected_loc = st.selectbox(
        "2. 地点を選択してください", 
        options=list(display_options.keys()),
        format_func=lambda x: display_options[x]
    )

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("専有面積 (㎡)", value=60.0, step=0.1)
    with col2:
        year_built = st.number_input("築年 (西暦)", min_value=1970, max_value=2026, value=2015)

    if st.button("AI精密査定を実行"):
        # 選択された地点の平米単価を取得
        unit_price = filtered_df[filtered_df['学習地点'] == selected_loc]['平均平米単価'].values[0]
        
        # 築年数による減価補正（簡易AIロジック）
        # 築1年ごとに1.5%下落すると仮定（2026年基準）
        age = 2026 - year_built
        age_factor = max(0.4, 1.0 - (age * 0.015)) 
        
        # 最終価格計算
        predicted_price = unit_price * area * age_factor
        
        st.markdown("---")
        st.write(f"### 📍 {selected_ward} {display_options[selected_loc]}")
        # 保存された指示に基づきカッコなしで提示
        st.success(f"標準AI査定価格: {int(predicted_price):,} 円")
        
        # プレミアム査定
        c1, c2 = st.columns(2)
        with c1: st.info(f"メジャー7価格: {int(predicted_price * 1.25):,} 円")
        with c2: st.info(f"ランドマーク価格: {int(predicted_price * 1.15):,} 円")
