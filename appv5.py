import pandas as pd
import re
import glob
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor

def train_model():
    all_files = glob.glob("*.csv")
    csv_files = [f for f in all_files if "chome_model" not in f and "23ku_v" not in f]
    
    combined_data = []
    valid_wards = ["千代田区","中央区","港区","新宿区","文京区","台東区","墨田区","江東区","品川区","目黒区","大田区","世田谷区","渋谷区","中野区","杉並区","豊島区","北区","荒川区","板橋区","練馬区","足立区","葛飾区","江戸川区"]

    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='cp932')
            def normalize_address(addr):
                addr = str(addr).replace('　', '').replace(' ', '').strip()
                ward = next((w for w in valid_wards if w in addr), None)
                if not ward: return None
                core_addr = addr.split(ward)[-1]
                match = re.search(r'^([^\d]+(?:\d+丁目|[^0-9]+番町|[^0-9]+町|[^0-9]+))', core_addr)
                return f"東京都{ward}{match.group(1)}" if match else f"東京都{ward}{core_addr}"

            df['地点'] = df['所在地'].apply(normalize_address)
            df = df.dropna(subset=['地点'])
            df['price'] = df['販売価格'].str.replace('万円', '').astype(float) * 10000
            df['area'] = df['専有面積'].str.replace('㎡', '').astype(float)
            df['age'] = 2026 - df['築年月'].str.extract(r'(\d{4})').astype(float)
            combined_data.append(df[['地点', 'area', 'age', 'price']])
        except: continue

    full_df = pd.concat(combined_data).dropna()
    
    # --- 重要：地点の列を固定する ---
    town_list = sorted(full_df['地点'].unique())
    # カテゴリ型として定義することで、get_dummiesの順番を固定
    full_df['地点'] = pd.Categorical(full_df['地点'], categories=town_list)
    
    X = pd.get_dummies(full_df[['地点', 'area', 'age']], columns=['地点'])
    y = full_df['price']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # 保存
    with open('real_estate_ai_v5.pkl', 'wb') as f:
        pickle.dump({'model': model, 'columns': X.columns.tolist()}, f)
    # アプリ用リストも更新
    joblib.dump(town_list, 'town_mapping_v5.joblib')
    print("学習完了：地点の並び順を固定しました。")

if __name__ == "__main__":
    train_model()
