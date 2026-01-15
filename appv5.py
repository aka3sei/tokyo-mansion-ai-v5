import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

@st.cache_resource
def load_all():
    with open('real_estate_ai_v5_final.pkl', 'rb') as f:
        return pickle.load(f)

data = load_all()

if data:
    model = data['model']
    cols = data['columns']
    base_prices = data['base_prices']
    
    # 地点リストの作成
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    st.title("🏙️ 23区精密エリアAI査定")
    ward = st.selectbox("1. 区を選択", sorted(df_towns['ward'].unique()))
    selected_loc = st.selectbox("2. 地点を選択", df_towns[df_towns['ward']==ward]['full'].tolist(), format_func=lambda x: x.split(ward)[-1])
    
    area = st.number_input("専有面積 (㎡)", value=60.0)
    year = st.number_input("築年 (西暦)", value=2015)

    if st.button("AI精密査定を実行"):
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'] = area, 2026 - year
        input_df[f'地点_{selected_loc}'] = 1.0
        
        # 地点単価を取得し、AIの補正を掛ける
        base = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        total = base * ratio * area
        
        # 賃料予測 (4%利回り)
        rent = (total * 0.04) / 12

        st.markdown("---")
        # 指示通りカッコなし
        st.success(f"📍 {selected_loc.replace('東京都','')}\n\n標準AI査定価格: {int(total):,} 円")
        st.warning(f"想定月額賃料: {int(rent):,} 円")
        
        c1, c2 = st.columns(2)
        with c1: st.info(f"メジャー7価格: {int(total * 1.25):,} 円")
        with c2: st.info(f"ランドマーク価格: {int(total * 1.15):,} 円")
