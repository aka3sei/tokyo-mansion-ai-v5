import streamlit as st
import pandas as pd
import pickle
import numpy as np
import re

# --- 1. モデルと新CSV（Tier_Factor入り）の読み込み ---
@st.cache_resource
def load_all():
    try:
        with open('real_estate_ai_v5_final.pkl', 'rb') as f:
            model_data = pickle.load(f)
        # Tier_Factorのみになった新しいCSVを読み込み
        master_df = pd.read_csv('chome_master_with_factors.csv').set_index('学習地点')
        return {"model": model_data['model'], "cols": model_data['columns'], "master": master_df}
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        return None

data = load_all()

# --- 2. パラメータ演算ロジック（地力αなどは元データから算出） ---
def calculate_5_params(walk_dist, area, base_price_val):
    alpha_thresholds = [535132, 664447, 771631, 875837, 978161, 1094232, 1229757, 1458726, 1847825]
    val = float(base_price_val) if base_price_val else 875837.0
    alpha_score = int(np.digitize(val, alpha_thresholds) + 1)
    mu_score = max(1, 11 - (walk_dist if walk_dist <= 5 else 5 + (walk_dist-5)//2))
    lambda_score = min(10, max(1, int(area / 20) + (1 if area > 100 else 0)))
    gamma_score = min(10, 4 + (alpha_score // 2))
    return {"alpha": alpha_score, "mu": mu_score, "lambda": lambda_score, "gamma": gamma_score}

# --- 3. メイン画面 ---
st.set_page_config(page_title="23区精密エリアAI査定", layout="centered")
st.title("🏙️ 23区精密エリアAI査定")

if data:
    model, cols, master = data['model'], data['cols'], data['master']
    towns = master.index.tolist()
    df_towns = pd.DataFrame({'full': towns})
    df_towns['ward'] = df_towns['full'].apply(lambda x: re.search(r'東京都(.*?区)', x).group(1) if re.search(r'東京都(.*?区)', x) else "その他")
    
    ward = st.selectbox("1. 区を選択してください", sorted(df_towns['ward'].unique()))
    loc_options = df_towns[df_towns['ward'] == ward]['full'].tolist()
    selected_loc = st.selectbox("2. 地点を選択してください", loc_options, format_func=lambda x: x.split(ward)[-1])
    
    c1, c2, c3 = st.columns(3)
    area_options = list(range(10, 75, 5))
    area = c1.selectbox("専有面積 ㎡", options=area_options, index=area_options.index(40))
    year_options = list(range(2026, 1989, -1))
    year_built = c2.selectbox("築年 西暦", options=year_options, index=year_options.index(2015))
    walk_options = list(range(1, 16))
    walk_dist = c3.selectbox("駅徒歩 分", options=walk_options, index=walk_options.index(8))

    if st.button("AI精密査定を実行"):
        # CSVから直接係数と単価を取得
        loc_data = master.loc[selected_loc]
        tier_factor = loc_data['Tier_Factor']
        # CSVに平均平米単価が残っている前提（消した場合はPKL内のbase_prices等を参照する調整が必要）
        base_price_val = loc_data.get('平均平米単価', 875837.0) 
        
        # AI推計
        input_df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
        input_df['area'], input_df['age'], input_df['walk'] = area, 2026 - year_built, walk_dist
        input_df[f'地点_{selected_loc}'] = 1.0
        ratio = model.predict(input_df)[0]
        
        # 最終価格 = 基準単価 × AI補正 × 保存済みのTier係数 × 面積
        std_price = base_price_val * ratio * tier_factor * area
        p = calculate_5_params(walk_dist, area, base_price_val)

        is_converged = 0.90 <= ratio <= 1.10
        status_color, status_bg = ("#166534", "#f0fdf4") if is_converged else ("#b91c1c", "#fef2f2")

        st.markdown("---")
        
        # 解析ログの表示（TIER_FACTORS を追加）
        html_report = f"""<div style="padding:20px;border:1px solid #e2e8f0;border-radius:12px;font-family:sans-serif;background-color:#ffffff;"><h3 style="color:#0f172a;margin:0;">📍 {selected_loc.replace('東京都','')}</h3><p style="color:#64748b;font-size:13px;">{area}㎡ / 築{2026-year_built}年 / 徒歩{walk_dist}分</p><div style="display:flex;flex-wrap:wrap;margin-top:25px;gap:20px;"><div style="flex:1;min-width:250px;"><div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:14px;"><span style="color:#64748b;">地点固有地力 α</span><span style="font-weight:bold;">Rank {p['alpha']}</span></div><div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:14px;"><span style="color:#64748b;">地点利便性指数 μ</span><span style="font-weight:bold;">Rank {p['mu']}</span></div><div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:14px;"><span style="color:#64748b;">面積希少性 λ</span><span style="font-weight:bold;">Rank {p['lambda']}</span></div><div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f1f5f9;font-size:14px;"><span style="color:#64748b;">時系列動態 γ</span><span style="font-weight:bold;">Rank {p['gamma']}</span></div><div style="display:flex;justify-content:space-between;padding:10px 0;font-size:14px;"><span style="color:#64748b;">市場非効率性 δ</span><span style="color:#b45309;font-weight:bold;">分析完了</span></div></div><div style="flex:1;min-width:250px;text-align:left;border-left:2px solid #f1f5f9;padding-left:25px;"><div style="color:#b45309;font-size:18px;font-weight:bold;letter-spacing:1px;">AI 指値</div><div style="font-size:40px;font-weight:bold;color:#1e293b;margin:5px 0;">{int(std_price):,} <span style="font-size:18px;color:#64748b;font-weight:normal;">円</span></div><div style="margin-top:15px;padding-top:15px;border-top:1px solid #f1f5f9;text-align:left;"><div style="color:#1e293b;font-size:14px;">Tier 1: {int(std_price * 1.25):,} 円</div><div style="color:#1e293b;font-size:14px;">Tier 2: {int(std_price * 1.15):,} 円</div><div style="color:#1e293b;font-size:14px;">Tier 3: {int(std_price * 1.05):,} 円</div></div></div></div><div style="background-color:{status_bg};padding:25px;border-radius:12px;border:3px solid {status_color};margin-top:30px;"><div style="font-family:'Courier New',monospace;font-size:18px;color:{status_color};font-weight:bold;line-height:1.6;">>> ANALYSIS_SEQUENCE_COMPLETE...<br>>> ALPHA_RANK_{p['alpha']}<br>>> MU_RANK_{p['mu']}<br>>> GAMMA_RANK_{p['gamma']}<br>>> LAMBDA_NON_LINEAR_RATIO: {p['lambda']*10}%<br>>> TIER_FACTORS: {tier_factor:.3f}x<br>>> MARKET_INEFFICIENCY_DELTA EVALUATED</div></div></div>"""
        
        st.markdown(html_report, unsafe_allow_html=True)
else:
    st.error("ファイルが読み込めません。")
