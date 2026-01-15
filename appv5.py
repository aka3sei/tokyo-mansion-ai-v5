import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- 1. AIモデルと地点単価データの読み込み ---
@st.cache_resource
def load_all():
    # 前の手順で作成した v5_final.pkl を読み込みます
    try:
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        st.error("モデルファイルが見つかりません。GitHubにpklファイルをアップロードしてください。")
        return None

data = load_all()

# --- 2. 画面デザインとスタイル ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 20px 0;
    }
    .price-large {
        font-size: 28px;
        font-weight: bold;
        color: #1e3a8a;
    }
    .brand-title {
        font-weight: bold;
        color: #b45309;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if data:
    model = data['model']
    cols = data['columns']
    base_prices = data['base_prices']
    
    # 地点リストの整理
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    # --- UI部 ---
    ward = st.selectbox("1. 区を選択してください", sorted(df_towns['ward'].unique()))
    
    # 選択された区の地点を抽出
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox(
        "2. 地点を選択してください", 
        loc_options, 
        format_func=lambda x: x.split(ward)[-1]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("専有面積 (㎡)", value=60.0, step=0.1)
    with col2:
        year_built = st.number_input("築年 (西暦)", value=2015, min_value=1970, max_value=2026)

    if st.button("AI精密査定を実行"):
        # AI入力データ作成
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'] = area
        input_df['age'] = 2026 - year_built
        input_df[f'地点_{selected_loc}'] = 1.0
        
        # 地点単価 × AI建物補正 × 面積 で算出
        base_unit = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        standard_price = base_unit * ratio * area
        
        # --- 結果表示（カッコなし） ---
        st.markdown("---")
        st.markdown(f"### 📍 {selected_loc.replace('東京都','')}")
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("標準的なマンションのAI査定価格")
        st.markdown(f'<div class="price-large">査定額: {int(standard_price):,} 円</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- 詳細版プレミアム査定 ---
        st.write("---")
        st.write("### 💎 ブランドマンション・プレミアム査定")
        
        c1, c2 = st.columns(2)
        
        with c1:
            m7_price = standard_price * 1.25
            st.markdown('<p class="brand-title">【メジャー7】プレミアム</p>', unsafe_allow_html=True)
            st.write(f"査定額: {int(m7_price):,} 円")
            st.caption("三井・三菱・住友・野村・地所レジ・東急・東京建物の7社分譲。圧倒的な安心感と管理体制により、エリア平均を大きく上回るリセールバリューを維持します。")
            
        with c2:
            landmark_price = standard_price * 1.15
            st.markdown('<p class="brand-title">【ランドマーク】プレミアム</p>', unsafe_allow_html=True)
            st.write(f"査定額: {int(landmark_price):,} 円")
            st.caption("100戸以上の大規模、または地域で誰もが知る象徴的な物件。希少性が高く、中古市場でも指名買いが発生するため、高値での成約が期待できます。")
