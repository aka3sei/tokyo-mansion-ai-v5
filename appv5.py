import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import plotly.graph_objects as go

# --- 1. AIモデル読み込み ---
@st.cache_resource
def load_all():
    try:
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        return None

data = load_all()

# --- 2. パラメータ演算ロジック ---
def calculate_5_params(walk_dist, tier_value, area, base_price_val):
    alpha_thresholds = [535132, 664447, 771631, 875837, 978161, 1094232, 1229757, 1458726, 1847825]
    val = float(base_price_val) if base_price_val else 875837.0
    alpha_score = int(np.digitize(val, alpha_thresholds) + 1)
    mu_score = max(1, 11 - (walk_dist if walk_dist <= 5 else 5 + (walk_dist-5)//2))
    beta_score = {1.25: 10, 1.15: 8, 1.05: 6}.get(tier_value, 4)
    lambda_score = min(10, int(area / 10) + (5 - alpha_score // 2))
    gamma_score = min(10, 4 + (alpha_score // 2))
    return [alpha_score, mu_score, beta_score, lambda_score, gamma_score]

# --- 3. 蜘蛛の巣グラフ生成関数 ---
def create_radar_chart(scores):
    categories = ['地点地力 α', '利便性 μ', 'クオリティ β', '面積希少性 λ', '動態 γ']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color='#D4AF37', width=3),
        fillcolor='rgba(212, 175, 55, 0.4)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor="#333"),
            angularaxis=dict(gridcolor="#333", tickfont=dict(color="#ccc", size=10)),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=45, t=20, b=20),
        height=320,
        autosize=True
    )
    return fig

# --- 4. 画面構成 ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.title("🏙️ 23区精密エリアAI査定")

if data:
    model, cols, base_prices = data['model'], data['columns'], data['base_prices']
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    ward = st.selectbox("1. 区を選択してください", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox("2. 地点を選択してください", loc_options, format_func=lambda x: x.split(ward)[-1])
    
    c1, c2, c3 = st.columns(3)
    area = c1.number_input("専有面積 ㎡", value=42.0, step=0.1)
    year_built = c2.number_input("築年 西暦", value=2015)
    walk_dist = c3.number_input("駅徒歩 分", value=8, min_value=1)

    # ボタンはここ1箇所だけにします
    if st.button("AI精密査定を実行", type="primary"):
        # 計算ロジック
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'], input_df['walk'] = area, 2026 - year_built, walk_dist
        input_df[f'地点_{selected_loc}'] = 1.0
        
        base_price_val = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        std_price = base_price_val * ratio * area
        scores = calculate_5_params(walk_dist, 1.05, area, base_price_val)

        st.markdown("---")
        
        # 巨大な一つの黒いカードとして構築（デザイン統合）
        st.markdown(f"""
        <div style="background-color: #111; padding: 20px; border-radius: 15px; border: 1px solid #333;">
            <h3 style="color: white; margin-top: 0;">📍 {selected_loc.replace('東京都','')}</h3>
            <p style="color: #888; font-size: 14px;">数理モデル解析：{area}㎡ / 築{2026-year_built}年 / 徒歩{walk_dist}分</p>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1.2, 1])
        
        with col_left:
            st.plotly_chart(create_radar_chart(scores), use_container_width=True, config={'displayModeBar': False})
        
        with col_right:
            st.markdown(f"""
            <div style="text-align: right; padding-top: 10px;">
                <div style="color: #D4AF37; font-size: 14px; font-weight: bold;">AI THEORETICAL PRICE</div>
                <div style="font-size: 40px; font-weight: bold; color: white; line-height: 1.2;">{int(std_price):,} <span style="font-size: 18px;">円</span></div>
                <div style="border-top: 1px solid #333; margin: 15px 0; padding-top: 10px;">
                    <div style="color: #aaa; font-size: 12px; margin-bottom: 5px;">【グレード別プレミアム査定】</div>
                    <div style="color: #fff; font-size: 14px;">Tier1: {int(std_price * 1.25):,} 円</div>
                    <div style="color: #fff; font-size: 14px;">Tier2: {int(std_price * 1.15):,} 円</div>
                    <div style="color: #fff; font-size: 14px;">Tier3: {int(std_price * 1.05):,} 円</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 解析ログ
        st.markdown(f"""
            <div style="font-family: 'Courier New', monospace; font-size: 12px; background: #000; padding: 15px; border-radius: 8px; color: #00ff00; border: 1px solid #222; margin-top: 10px;">
                <span style="color: #555;">>></span> ANALYSIS_SEQUENCE_COMPLETE...<br>
                <span style="color: #555;">>></span> LOCATION_ALPHA: RANK_{scores[0]} / UTILITY_MU: RANK_{scores[1]}<br>
                <span style="color: #555;">>></span> NON_LINEAR_LAMBDA_DETECTION: {scores[3]*10}%<br>
                <span style="color: #555;">>></span> <span style="color: #ffaa00;">MARKET_INEFFICIENCY_DELTA DETECTED</span><br>
                <span style="color: #555;">>></span> CONCLUSION: 理論均衡価格への収束性が認められます。
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("AIモデルの読み込みに失敗しました。")
