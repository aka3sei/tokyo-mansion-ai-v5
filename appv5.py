import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- 1. AIモデル読み込み ---
@st.cache_resource
def load_all():
    try:
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

data = load_all()

# --- 2. スタイル設定 ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""
<style>
    .result-card { padding: 25px; border-radius: 12px; background-color: #f8fafc; border: 1px solid #e2e8f0; margin: 20px 0; }
    .price-large { font-size: 32px; font-weight: bold; color: #1e3a8a; }
    .premium-box { padding: 15px; border-left: 5px solid #b45309; background-color: #fffbeb; margin-bottom: 20px; }
    .premium-title { font-weight: bold; color: #92400e; font-size: 18px; margin-bottom: 5px; }
    .brand-list { font-size: 13px; color: #4b5563; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if data:
    model, cols, base_prices = data['model'], data['columns'], data['base_prices']
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    ward = st.selectbox("1. 区を選択してください", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox("2. 地点を選択してください", loc_options, format_func=lambda x: x.split(ward)[-1])
    
    col1, col2 = st.columns(2)
    with col1: area = st.number_input("専有面積 (㎡)", value=60.0)
    with col2: year_built = st.number_input("築年 (西暦)", value=2015)

    if st.button("AI精密査定を実行"):
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'] = area, 2026 - year_built
        input_df[f'地点_{selected_loc}'] = 1.0
        
        # 計算
        base = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        standard_price = base * ratio * area
        
        st.markdown("---")
        st.markdown(f"### 📍 {selected_loc.replace('東京都','')}")
        
        # 標準査定
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("標準的なマンションのAI査定価格")
        st.markdown(f'<div class="price-large">査定額: {int(standard_price):,} 円</div>', unsafe_allow_html=True)
        st.write("※一般分譲マンション、地元デベロッパー物件等の市場相場です。")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- 詳細版プレミアム査定 ---
        st.write("### 💎 ブランドマンション・プレミアム査定")
        
        # メジャー7最高級
        st.markdown('<div class="premium-box">', unsafe_allow_html=True)
        st.markdown('<div class="premium-title">【最高級ブランド】プレミアム（+25%〜）</div>', unsafe_allow_html=True)
        st.write(f"査定目安: **{int(standard_price * 1.25):,} 円**")
        st.markdown('<div class="brand-list">対象：パークマンション、パークコート、三井不動産レジデンシャル、三菱地所レジデンス、ザ・パークハウスグラン、住友不動産、グランドヒルズ、野村不動産、プラウドタワー、東京建物、ブリリアタワー、東急不動産、ブランズ永田町など。</div>', unsafe_allow_html=True)
        st.caption("各デベロッパーが社運をかけて開発するフラッグシップモデル。最高級の資材と意匠、コンシェルジュサービス等を備え、資産価値が極めて落ちにくいのが特徴です。")
        st.markdown('</div>', unsafe_allow_html=True)

        # ランドマーク/中堅高級
        st.markdown('<div class="premium-box" style="border-left-color: #0369a1; background-color: #f0f9ff;">', unsafe_allow_html=True)
        st.markdown('<div class="premium-title" style="color: #075985;">【ランドマーク・高級】プレミアム（+15%〜）</div>', unsafe_allow_html=True)
        st.write(f"査定目安: **{int(standard_price * 1.15):,} 円**")
        st.markdown('<div class="brand-list">対象：パークホームズ、ザ・パークハウス、プラウド、ブリリア、ブランズ、グランドメゾン、パークタワー、ピアース、ディアナコート、ジオ、など。</div>', unsafe_allow_html=True)
        st.caption("地域を象徴する大規模物件や、大手ブランドの標準的な高級ライン。高い知名度と信頼性により、中古市場でも安定した高値取引が約束されます。")
        st.markdown('</div>', unsafe_allow_html=True)
