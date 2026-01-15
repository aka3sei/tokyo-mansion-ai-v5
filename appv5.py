import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- 1. AIモデル（脳）の読み込み ---
@st.cache_resource
def load_ai_model():
    try:
        # 学習時に作成した「final」ファイルを読み込みます
        with open('real_estate_ai_final.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"モデルファイル(real_estate_ai_final.pkl)の読み込みに失敗しました: {e}")
        return None

ai_res = load_ai_model()

# --- 2. デザイン ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.markdown("""<style>
    header[data-testid="stHeader"] { visibility: hidden; }
    .stApp { background-color: white; }
    .result-card { padding: 25px; border: 1px solid #e2e8f0; border-radius: 15px; background-color: #f8fafc; margin-top: 20px; }
    .main-price { font-size: 32px; font-weight: bold; color: #1e3a8a; }
</style>""", unsafe_allow_html=True)

st.title("🏙️ 23区精密エリアAI査定")

if ai_res:
    model = ai_res['model']
    model_columns = ai_res['columns'] # AIが学習した「地点_東京都...」というカラム名のリスト
    
    # 1. AIの学習済みカラムから地点リストを作成
    # これにより、AIが知っている名前とアプリの選択肢が100%一致します
    all_locations = [c.replace('地点_', '') for c in model_columns if c.startswith('地点_')]
    df_towns = pd.DataFrame(all_locations, columns=['full_address'])
    
    # 区名を抽出してグループ化
    df_towns['ward'] = df_towns['full_address'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1) if '区' in x else "その他")
    
    # --- UI ---
    ward_list = sorted(df_towns['ward'].unique())
    selected_ward = st.selectbox("1. 区を選択してください", options=ward_list)
    
    # 選択された区の地点を絞り込み
    filtered_df = df_towns[df_towns['ward'] == selected_ward].sort_values('full_address')
    
    # 表示用：区名より後の住所だけを表示（三宿１丁目など）
    display_map = {row['full_address']: row['full_address'].split(selected_ward)[-1] for _, row in filtered_df.iterrows()}
    
    selected_loc = st.selectbox(
        "2. 地点を選択してください", 
        options=list(display_map.keys()),
        format_func=lambda x: display_map[x]
    )

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("専有面積 (㎡)", value=60.0, step=0.1)
    with col2:
        year_built = st.number_input("築年 (西暦)", min_value=1970, max_value=2026, value=2015)

    if st.button("AI精密査定を実行"):
        # AI入力用の空データフレーム作成
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        input_df['area'] = area
        input_df['age'] = 2026 - year_built
        
        # 選択された地点のフラグを1にする（これで地点ごとの単価が反映されます）
        target_col = f'地点_{selected_loc}'
        
        if target_col in model_columns:
            input_df[target_col] = 1.0
            predicted_price = model.predict(input_df)[0]
            
            # 結果表示（指示通りカッコなし）
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write(f"### 📍 {selected_ward} {display_map[selected_loc]}")
            st.markdown(f'<div class="main-price">標準AI査定価格: {int(predicted_price):,} 円</div>', unsafe_allow_html=True)
            
            st.divider()
            st.write("#### 💎 ブランドマンション・プレミアム査定")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**メジャー7価格**")
                st.write(f"### {int(predicted_price * 1.25):,} 円")
            with c2:
                st.write("**ランドマーク価格**")
                st.write(f"### {int(predicted_price * 1.15):,} 円")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("地点の照合に失敗しました。学習データを確認してください。")
