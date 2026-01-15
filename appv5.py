import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

@st.cache_resource
def load_model():
    try:
        with open('real_estate_ai_final.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

ai_data = load_model()

st.set_page_config(page_title="23区精密エリアAI査定")
st.title("🏙️ 23区精密エリアAI査定")

if ai_data:
    model = ai_data['model']
    cols = ai_data['columns']
    
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    ward = st.selectbox("1. 区を選択", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    # カッコなしで表示
    selected_loc = st.selectbox("2. 地点を選択", loc_options, format_func=lambda x: x.split(ward)[-1])

    area = st.number_input("専有面積 (㎡)", value=60.0)
    year = st.number_input("築年 (西暦)", value=2015)

    if st.button("AI精密査定を実行"):
        input_data = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_data['area'] = area
        input_data['age'] = 2026 - year
        
        target = f'地点_{selected_loc}'
        if target in cols:
            input_data[target] = 1.0
            # 単価予測
            predicted_unit_price = model.predict(input_data)[0]
            total = predicted_unit_price * area
            
            # 賃料予測 (想定利回り4%で算出)
            monthly_rent = (total * 0.04) / 12
            
            st.markdown("---")
            # 指定通りカッコなし
            st.success(f"📍 {selected_loc.replace('東京都','')}\n\n標準AI査定価格: {int(total):,} 円")
            
            # 賃料予測の表示
            st.warning(f"想定月額賃料: {int(monthly_rent):,} 円")
            
            c1, c2 = st.columns(2)
            with c1: st.info(f"メジャー7価格: {int(total * 1.25):,} 円")
            with c2: st.info(f"ランドマーク価格: {int(total * 1.15):,} 円")
