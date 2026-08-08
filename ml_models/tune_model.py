import os
import sys
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def tune_and_train():
    print("🧠 [Tuning] Süni İntellektin Hiperparametr Optimizasiyası Başlayır...")
    
    data_path = os.path.join(os.path.dirname(__file__), "traffic_data.csv")
    if not os.path.exists(data_path):
        print("❌ Xəta: traffic_data.csv tapılmadı.")
        return
        
    df = pd.read_csv(data_path)
    print(f"📊 Datanın həcmi: {len(df)} sətir.")
    
    X = df[['road', 'hour', 'day_of_week', 'is_weekend', 'weather', 'incidents']]
    y = df['congestion_pct']
    
    categorical_features = ['road', 'weather']
    numeric_features = ['hour', 'day_of_week', 'is_weekend', 'incidents']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # GridSearch üçün Parametr Şəbəkəsi (Yüzlərlə beyin variantı)
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    }
    
    print("⚙️ GridSearchCV: Riyazi kombinasiyalar sınaqdan keçirilir (Bir neçə dəqiqə çəkə bilər)...")
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3, # 3 fold cross validation
        n_jobs=-1, # Bütün CPU nüvələrini istifadə et
        scoring='r2',
        verbose=1
    )
    
    # Təlim və Tuning eyni anda
    grid_search.fit(X_train, y_train)
    
    # Ən Yaxşı Beyni seçirik
    best_model = grid_search.best_estimator_
    
    print("\n🏆 Ən Mükəmməl Beyin Parametrləri Tapıldı:")
    best_params = grid_search.best_params_
    for k, v in best_params.items():
        print(f"  - {k.replace('regressor__', '')}: {v}")
    
    # Yoxlama
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n🎯 YENİ DƏQİQLİK (R2 Score): {r2:.4f}")
    
    # Yaddaşa Yazma (Köhnəni əzirik)
    model_path = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"✅ Optimallaşdırılmış Şah Əsər uğurla yadda saxlanıldı: {model_path}")

if __name__ == "__main__":
    tune_and_train()
