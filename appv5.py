import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- 1. AIモデル読み込み ---
@st.cache_resource
def load_all():
    try:
        # 9MBに軽量化した最新モデルを読み込み
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

data = load_all()

# --- 2. 画面デザイン・スタイル設定 ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""
<style>
    .result-card { padding: 25px; border-radius: 12px; background-color: #f1f5f9; border: 1px solid #cbd5e1; margin: 20px 0; }
    .price-large { font-size: 34px; font-weight: bold; color: #0f172a; }
    .brand-section { margin-top: 30px; border-top: 2px solid #ddd; padding-top: 20px; }
    .tier-card { padding: 18px; border-radius: 10px; margin-bottom: 15px; border-left: 6px solid #ccc; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .tier-top { border-left-color: #b45309; background-color: #fffbeb; }
    .tier-high { border-left-color: #0369a1; background-color: #f0f9ff; }
    .tier-standard { border-left-color: #4b5563; background-color: #f9fafb; }
    .tier-title { font-weight: bold; font-size: 18px; color: #1e293b; margin-bottom: 5px; }
    .brand-names { font-size: 14px; font-weight: bold; color: #334155; margin-bottom: 8px; }
    .brand-desc { font-size: 13px; color: #475569; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if data:
    model, cols, base_prices = data['model'], data['columns'], data['base_prices']
    
    # 地点リストの整理
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    # --- 入力セクション ---
    ward = st.selectbox("1. 区を選択してください", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox("2. 地点を選択してください", loc_options, format_func=lambda x: x.split(ward)[-1])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("専有面積 ㎡", value=42.0, step=0.1)
    with col2:
        year_built = st.number_input("築年 西暦", value=2015, min_value=1970, max_value=2026)
    with col3:
        walk_dist = st.number_input("駅徒歩 分", value=8, min_value=1, max_value=30)

    if st.button("AI精密査定を実行"):
        # 入力データの組み立て
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'] = area
        input_df['age'] = 2026 - year_built
        input_df['walk'] = walk_dist
        input_df[f'地点_{selected_loc}'] = 1.0
        
        # 予測実行（地点ベース単価 × AI補正率 × 面積）
        base = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        std_price = base * ratio * area
        
        # --- 結果表示（カッコなし） ---
        st.markdown("---")
        st.markdown(f"### 📍 {selected_loc.replace('東京都','')}")
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write(f"専有面積 {area}㎡ / 築{2026-year_built}年 / 駅徒歩{walk_dist}分")
        st.write("標準的なマンションのAI査定価格")
        st.markdown(f'<div class="price-large">査定額: {int(std_price):,} 円</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- モダンリビング参照：ブランドグレード別詳細査定 ---
        st.markdown('<div class="brand-section">', unsafe_allow_html=True)
        st.write("### 💎 デベロッパー別・ブランドグレード査定")

        # Tier 1: 最高級
        st.markdown(f"""
        <div class="tier-card tier-top">
            <div class="tier-title">【最高級ブランド】プレミアム査定：{int(std_price * 1.25):,} 円〜</div>
            <div class="brand-names">三井：パークマンション / 三菱：ザ・パークハウス グラン / 住友：グランドヒルズ / 東急：ブランズ ザ・レジデンス</div>
            <div class="brand-desc">
                モダンリビング誌で「不動産芸術」と称されるフラッグシップ。
                都心の超一等地に限定され、究極の資材と意匠を完備。時が経つほどにその希少性が際立つ、別格の資産価値を維持します。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tier 2: 高級・タワー
        st.markdown(f"""
        <div class="tier-card tier-high">
            <div class="tier-title">【高級・タワー】プレミアム査定：{int(std_price * 1.15):,} 円〜</div>
            <div class="brand-names">三井：パークコート・パークタワー / 三菱：ザ・パークハウス（都心） / 野村：プラウドタワー / 東京建物：ブリリアタワー</div>
            <div class="brand-desc">
                エリアの景観を象徴するランドマーク物件。
                優れたデザイン性と充実した共用施設により、中古市場でも指名買いが発生する、信頼と実績のハイエンドラインです。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tier 3: スタンダード大手
        st.markdown(f"""
        <div class="tier-card tier-standard">
            <div class="tier-title">【スタンダード大手】プレミアム査定：{int(std_price * 1.05):,} 円〜</div>
            <div class="brand-names">三井：パークホームズ / 住友：シティハウス / 野村：プラウド / 東急：ブランズ / 東京建物：ブリリア</div>
            <div class="brand-desc">
                安心感と資産性のバランスに優れた大手シリーズ。
                利便性の高い立地に多く、施工品質や管理体制への信頼から、一般物件より一段高い評価で安定して取引されます。
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("モデルファイル real_estate_ai_v5_final.pkl が見つかりません。")
