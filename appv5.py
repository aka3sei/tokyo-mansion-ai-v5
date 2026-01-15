import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- AIモデル読み込み ---
@st.cache_resource
def load_all():
    try:
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

data = load_all()

# --- 画面デザイン ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""
<style>
    .result-card { padding: 25px; border-radius: 12px; background-color: #f8fafc; border: 1px solid #e2e8f0; margin: 20px 0; }
    .price-large { font-size: 32px; font-weight: bold; color: #1e3a8a; }
    .brand-section { margin-top: 30px; border-top: 2px solid #eee; pt: 20px; }
    .brand-tier { padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 6px solid #ccc; }
    .tier-top { border-left-color: #b45309; background-color: #fffbeb; } /* 最高級 */
    .tier-high { border-left-color: #0369a1; background-color: #f0f9ff; } /* 高級 */
    .tier-standard { border-left-color: #4b5563; background-color: #f9fafb; } /* 標準 */
    .tier-title { font-weight: bold; font-size: 17px; margin-bottom: 5px; }
    .brand-names { font-size: 14px; font-weight: bold; color: #334155; margin-bottom: 8px; }
    .brand-desc { font-size: 13px; color: #475569; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if data:
    model, cols, base_prices = data['model'], data['columns'], data['base_prices']
    towns = [c.replace('地点_', '') for c in cols if c.startswith('地点_')]
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1))
    
    ward = st.selectbox("1. 区を選択", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox("2. 地点を選択", loc_options, format_func=lambda x: x.split(ward)[-1])
    
    col1, col2 = st.columns(2)
    with col1: area = st.number_input("専有面積 (㎡)", value=60.0)
    with col2: year_built = st.number_input("築年 (西暦)", value=2015)

    if st.button("AI精密査定を実行"):
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'] = area, 2026 - year_built
        input_df[f'地点_{selected_loc}'] = 1.0
        
        base = base_prices.get(selected_loc, 0)
        ratio = model.predict(input_df)[0]
        std_price = base * ratio * area
        
        st.markdown("---")
        st.markdown(f"### 📍 {selected_loc.replace('東京都','')}")
        
        # 標準査定額
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("標準的なマンション（一般分譲・地元デベ等）のAI査定価格")
        st.markdown(f'<div class="price-large">査定額: {int(std_price):,} 円</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- モダンリビング参照：ブランド別プレミアム査定 ---
        st.markdown('<div class="brand-section">', unsafe_allow_html=True)
        st.write("### 💎 デベロッパー別・ブランドグレード査定")

        # 1. 最高級グレード (+25%〜)
        st.markdown(f"""
        <div class="brand-tier tier-top">
            <div class="tier-title">【最高級ブランド】プレミアム査定：{int(std_price * 1.25):,} 円〜</div>
            <div class="brand-names">三井：パークマンション / 三菱：ザ・パークハウス グラン / 住友：グランドヒルズ / 東急：ブランズ ザ・レジデンス</div>
            <div class="brand-desc">
                モダンリビング誌でも「各社が社運をかけたフラッグシップ」と評されるシリーズ。
                都心の超一等地に限定され、最高級の資材と意匠、ホテルライクなサービスを完備。資産価値が極めて落ちにくい別格の存在です。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. 高級・タワーグレード (+15%〜)
        st.markdown(f"""
        <div class="brand-tier tier-high">
            <div class="tier-title">【高級・タワー】プレミアム査定：{int(std_price * 1.15):,} 円〜</div>
            <div class="brand-names">三井：パークコート・パークタワー / 三菱：ザ・パークハウス（都心）/ 野村：プラウド・プラウドタワー / 東京建物：ブリリアタワー</div>
            <div class="brand-desc">
                「高い顧客満足度と管理体制」を誇るメジャー7の主力ライン。
                タワーマンションや大規模複合開発（パークシティ等）が含まれます。優れたデザイン性と共用施設により、エリアのランドマークとして指名買いが発生します。
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. 一般・ファミリー向け (+5%〜)
        st.markdown(f"""
        <div class="brand-tier tier-standard">
            <div class="tier-title">【スタンダード大手】プレミアム査定：{int(std_price * 1.05):,} 円〜</div>
            <div class="brand-names">三井：パークホームズ / 住友：シティハウス / 東急：ブランズ / 東京建物：ブリリア / 大京：ザ・ライオンズ</div>
            <div class="brand-desc">
                「安心感と資産性のバランス」に優れた大手ブランドの標準シリーズ。
                利便性の高い立地に多く、施工品質やアフターサービスへの信頼から、中古市場でも一般物件より一段高い評価で取引されます。
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
