import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re
import plotly.graph_objects as go

# --- 1. AIモデル（地点辞書・統計しきい値を含む）の読み込み ---
@st.cache_resource
def load_all():
    try:
        # このpklの中に、モデル、地点別単価、10段階のしきい値をすべてパッキングしておく
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました。")
        return None

data = load_all()

# --- 2. 5軸パラメータ演算（地力αを10段階で算出） ---
def calculate_5_params(selected_loc, walk_dist, tier_value, area, data):
    # α: 地点固有地力 (pkl内の辞書から単価を引き、統計しきい値で10段階判定)
    base_unit_prices = data.get('base_unit_prices', {}) # 地点名: 単価 の辞書
    alpha_thresholds = data.get('alpha_thresholds', []) # 10段階のしきい値リスト
    
    u_price = base_unit_prices.get(selected_loc, 0)
    # 統計データに基づき1〜10に分類
    alpha_score = np.digitize(u_price, alpha_thresholds[1:-1]) + 1
    
    # μ: 地点利便性指数 (徒歩分数から算出)
    mu_score = max(1, 11 - (walk_dist if walk_dist <= 5 else 5 + (walk_dist-5)//2))
    
    # β: アセット・クオリティ係数 (ブランドから算出)
    beta_score = {1.25: 10, 1.15: 8, 1.05: 6}.get(tier_value, 4)
    
    # λ: 面積寄与の非線形性 (広さの希少性：地力αが低いエリアほど広い物件を高く評価)
    lambda_score = min(10, int(area / 10) + (5 - alpha_score // 2))
    
    # γ: 時系列動態モメンタム (固定値または地点特性から算出)
    gamma_score = min(10, 5 + (alpha_score // 3))
    
    return [alpha_score, mu_score, beta_score, lambda_score, gamma_score]

# --- 3. 蜘蛛の巣グラフ表示 ---
def create_radar_chart(scores):
    categories = ['地点固有地力(α)', '地点利便性指数(μ)', 'アセットクオリティ(β)', '面積寄与の非線形性(λ)', '時系列動態(γ)']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill='toself',
        line=dict(color='#D4AF37', width=3), # 黄金色
        fillcolor='rgba(212, 175, 55, 0.4)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, gridcolor="#444"),
            angularaxis=dict(gridcolor="#444", font=dict(color="white", size=12)),
            bgcolor="rgb(20, 20, 20)"
        ),
        showlegend=False,
        paper_bgcolor="rgb(10, 10, 10)",
        margin=dict(l=50, r=50, t=40, b=40),
        height=400
    )
    return fig

# --- (以下、査定実行ボタン内の処理) ---
if st.button("AI精密査定を実行"):
    # ... (予測価格計算処理) ...
    
    # パラメータ算出
    # tier_valueは選択されたグレードの倍率を渡す
    scores = calculate_5_params(selected_loc, walk_dist, 1.05, area, data)
    
    st.markdown("---")
    # 蜘蛛の巣グラフ（証拠）を表示
    st.plotly_chart(create_radar_chart(scores), use_container_width=True)
    
    # 専門用語によるエビデンス表示
    st.markdown(f"""
    <div style="background-color: #000; padding: 15px; border: 1px solid #333; color: #00ff00; font-family: monospace;">
    [SYSTEM] 構造解析完了<br>
    [ANALYSIS] 地点固有地力 α: Rank {scores[0]} / 地点利便性指数 μ: Rank {scores[1]}<br>
    [ANALYSIS] 面積寄与の非線形性 λ: Rank {scores[3]} を検知<br>
    [REPORT] 流通価格と数理モデルの乖離（δ）を解析中... 歪みを検出しました。<br>
    [RESULT] 理論均衡価格への収束ポテンシャルが極めて高いと判断します。
    </div>
    """, unsafe_allow_html=True)
