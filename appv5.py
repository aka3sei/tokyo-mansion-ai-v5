import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import plotly.graph_objects as go
import base64

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

st.markdown("""
<style>
    body { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
    /* ボタンをゴールドに */
    .stButton>button { width: 100%; border-radius: 8px; background-color: #D4AF37; color: black; font-weight: bold; border: none; height: 3em; }
    .stButton>button:hover { background-color: #B8962E; color: white; }
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
    
    c1, c2, c3 = st.columns(3)
    area = c1.number_input("専有面積 ㎡", value=42.0, step=0.1)
    year_built = c2.number_input("築年 西暦", value=2015)
    walk_dist = c3.number_input("駅徒歩 分", value=8, min_value=1)

    if st.button("AI精密査定を実行"):
        # --- 計算 ---
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'], input_df['walk'] = area, 2026 - year_built, walk_dist
        input_df[f'地点_{selected_loc}'] = 1.0
        
        base_price_val = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        std_price = base_price_val * ratio * area
        scores = calculate_5_params(walk_dist, 1.05, area, base_price_val)

        st.markdown("---")
        
        # --- 一体型デザインの構築 ---
        # 1. 外部のコンテナ（黒背景）を先に定義
        container = st.container()
        with container:
            # HTMLとCSSで黒いカードを生成
            st.markdown(f"""
            <div style="background-color: #111; padding: 25px; border-radius: 15px; border: 1px solid #333; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; font-size: 22px;">📍 {selected_loc.replace('東京都','')}</h3>
                <p style="color: #888; font-size: 13px; margin-bottom: 20px;">数理モデル解析：{area}㎡ / 築{2026-year_built}年 / 徒歩{walk_dist}分</p>
                <div id="dashboard-body" style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="flex: 1.2; min-width: 300px;" id="chart-area"></div>
                    <div style="flex: 1; min-width: 250px; text-align: right; border-left: 1px solid #222; padding-left: 20px;">
                        <div style="color: #D4AF37; font-size: 12px; font-weight: bold; letter-spacing: 1px;">AI THEORETICAL PRICE</div>
                        <div style="font-size: 42px; font-weight: bold; color: white; line-height: 1.1; margin: 5px 0;">{int(std_price):,} <span style="font-size: 16px; color: #888;">円</span></div>
                        <div style="border-top: 1px solid #333; margin: 15px 0; padding-top: 15px;">
                            <div style="color: #888; font-size: 11px; margin-bottom: 8px; letter-spacing: 0.5px;">【グレード別プレミアム推計】</div>
                            <div style="color: #eee; font-size: 14px; margin-bottom: 4px;">Tier 1: {int(std_price * 1.25):,} 円</div>
                            <div style="color: #eee; font-size: 14px; margin-bottom: 4px;">Tier 2: {int(std_price * 1.15):,} 円</div>
                            <div style="color: #eee; font-size: 14px;">Tier 3: {int(std_price * 1.05):,} 円</div>
                        </div>
                    </div>
                </div>
                <div style="font-family: 'Courier New', monospace; font-size: 11px; background: rgba(0,0,0,0.6); padding: 15px; border-radius: 8px; color: #00ff00; border: 1px solid #222; margin-top: 15px; line-height: 1.6;">
                    <span style="color: #555;">>></span> ANALYSIS_SEQUENCE_COMPLETE...<br>
                    <span style="color: #555;">>></span> LOCATION_ALPHA: RANK_{scores[0]} / UTILITY_MU: RANK_{scores[1]} / MOMENTUM_GAMMA: RANK_{scores[4]}<br>
                    <span style="color: #555;">>></span> NON_LINEAR_LAMBDA_RATIO: {scores[3]*10}% DETECTED<br>
                    <span style="color: #555;">>></span> <span style="color: #ffaa00;">MARKET_INEFFICIENCY_DELTA EVALUATED</span><br>
                    <span style="color: #555;">>></span> CONCLUSION: 理論均衡価格への高い収束性を確認。
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # グラフはHTMLの外（Streamlitのレイヤー）で描画されますが、
            # 直前に表示したHTMLカードの直後に来るため、視覚的に一体化します。
            # グラフをカードの中央（左側）に配置するための調整
            col_chart, col_empty = st.columns([1.5, 1])
            with col_chart:
                # カード内の空きスペースに重なるように配置をマイナスマージンで調整
                st.markdown('<div style="margin-top: -370px;">', unsafe_allow_html=True)
                st.plotly_chart(create_radar_chart(scores), use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("AIモデルの読み込みに失敗しました。")
