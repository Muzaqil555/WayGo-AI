import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def train():
    print("🤖 [Continuous Learning] Model öyrədilməsinə başlanılır...")
    
    # 1. Datanı Yüklə
    data_path = os.path.join(os.path.dirname(__file__), "traffic_data.csv")
    if not os.path.exists(data_path):
        print("❌ Xəta: traffic_data.csv tapılmadı. Zəhmət olmasa data_generator.py işlədin.")
        return
        
    df = pd.read_csv(data_path)
    print(f"📊 Data yükləndi: {len(df)} sətir.")
    
    # 2. X (Özəlliklər) və Y (Hədəf - Tıxac faizi) ayır
    X = df[['road', 'hour', 'day_of_week', 'is_weekend', 'weather', 'incidents']]
    y = df['congestion_pct']
    
    # Categorical (Mətn) və Numerical (Rəqəm) sütunlar
    categorical_features = ['road', 'weather']
    numeric_features = ['hour', 'day_of_week', 'is_weekend', 'incidents']
    
    # 3. Data Transformasiyası (Boru xətti - Pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # Yüksək Dəqiqlik üçün Random Forest (Təsadüfi Meşə) alqoritmi
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train/Test bölünməsi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Modeli Öyrət (Fit)
    print("⏳ Model datanı analiz edir və öyrənir...")
    model_pipeline.fit(X_train, y_train)
    
    # 5. Modelin Dəqiqliyini Yoxla
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"🎯 Modelin Dəqiqliyi (R2 Score): {r2:.2f}")
    print(f"📉 Orta Xəta Payı (MAE): {mae:.2f}% (Tıxac fərqi)")
    
    # 6. Modeli Yaddaşa Yaz (Pickle)
    model_path = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print(f"✅ Süni İntellekt Beyni (Model) uğurla yadda saxlanıldı: {model_path}")

if __name__ == "__main__":
    train()
