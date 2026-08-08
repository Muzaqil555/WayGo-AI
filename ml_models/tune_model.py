import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def evaluate_model_fit(model, X_train, X_test, y_train, y_test, model_name):
    """
    Modelin Underfitting / Overfitting və bütün performans metriklərini tam dondurub analiz edir.
    """
    # 1. Təlim (Train) məlumatlarında performans
    y_train_pred = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)

    # 2. Test məlumatlarında performans (Görmədiyi data)
    y_test_pred = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = root_mean_squared_error(y_test, y_test_pred)

    # 3. Overfitting / Underfitting analizi
    r2_gap = r2_train - r2_test  # Train və Test fərqi
    
    if r2_test < 0.70:
        fit_status = "⚠️ UNDERFITTING (Model zəif öyrənib, zəif cavab verir)"
    elif r2_gap > 0.08:
        fit_status = f"⚠️ OVERFITTING (Əzbərləyib: Train-Test fərqi = {r2_gap*100:.1f}%)"
    elif r2_gap > 0.04:
        fit_status = f"⚡ MÜLAYİM OVERFITTING (Fərq = {r2_gap*100:.1f}%)"
    else:
        fit_status = "✅ MÜKƏMMƏL BALANS (Overfitting/Underfitting yoxdur)"

    return {
        "name": model_name,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "r2_gap": r2_gap,
        "status": fit_status,
        "model": model
    }

def tune_and_train():
    print("==========================================================================")
    print("🧠 [Deep AI Audit] Model Optimizasiyası, Overfitting & Underfitting Analizi")
    print("==========================================================================\n")
    
    data_path = os.path.join(os.path.dirname(__file__), "traffic_data.csv")
    if not os.path.exists(data_path):
        print("❌ Xəta: traffic_data.csv tapılmadı.")
        return
        
    df = pd.read_csv(data_path)
    print(f"📊 Analiz edilən datanın həcmi: {len(df)} sətir.")
    
    X = df[['road', 'hour', 'day_of_week', 'is_weekend', 'weather', 'incidents']]
    y = df['congestion_pct']
    
    categorical_features = ['road', 'weather']
    numeric_features = ['hour', 'day_of_week', 'is_weekend', 'incidents']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # -------------------------------------------------------------------------
    # 1. RANDOM FOREST (Regularization parametrləri ilə - Overfitting-in qarşısı alınır)
    # -------------------------------------------------------------------------
    print("\n⚙️  [1/2] Random Forest (GridSearchCV 5-Fold, Regularization ilə)...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    rf_param_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__max_depth': [8, 12, 16],
        'regressor__min_samples_split': [5, 10],
        'regressor__min_samples_leaf': [2, 4]
    }

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='r2',
        return_train_score=True
    )
    rf_search.fit(X_train, y_train)
    rf_res = evaluate_model_fit(rf_search.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")

    # -------------------------------------------------------------------------
    # 2. XGBOOST (Gradient Boosting - Regularization parametrləri ilə)
    # -------------------------------------------------------------------------
    print("⚙️  [2/2] XGBoost Regressor (GridSearchCV 5-Fold, L1/L2 Regularization)...")
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, verbosity=0))
    ])

    xgb_param_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__max_depth': [4, 6, 8],
        'regressor__learning_rate': [0.03, 0.08, 0.15],
        'regressor__reg_alpha': [0.1, 1.0],   # L1 Regularization (Overfitting əleyhinə)
        'regressor__reg_lambda': [0.1, 1.0]   # L2 Regularization (Overfitting əleyhinə)
    }

    xgb_search = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='r2',
        return_train_score=True
    )
    xgb_search.fit(X_train, y_train)
    xgb_res = evaluate_model_fit(xgb_search.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")

    # -------------------------------------------------------------------------
    # MÜQAYİSƏ CƏDVƏLİ VƏ AUDİT HESABATI
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("📊 İKİ MODELİN HƏRTƏRƏFLİ FİTTİNG VƏ DƏQİQLİK MÜQAYİSƏSİ")
    print("="*80)
    
    results = [rf_res, xgb_res]
    for r in results:
        print(f"\n🔹 Model: {r['name']}")
        print(f"   ├─ Train R² (Təlim dəqiqliyi)   : {r['r2_train']:.4f}")
        print(f"   ├─ Test R²  (Gələcək dəqiqliyi) : {r['r2_test']:.4f}")
        print(f"   ├─ Overfitting Fərqi (R² Gap)  : {r['r2_gap']*100:.2f}%")
        print(f"   ├─ Train MAE / Test MAE         : {r['mae_train']:.2f}% / {r['mae_test']:.2f}%")
        print(f"   ├─ Train RMSE / Test RMSE       : {r['rmse_train']:.2f}% / {r['rmse_test']:.2f}%")
        print(f"   └─ Status                       : {r['status']}")

    # Ən balansı yüksək olan modeli seçirik (Test R2 yüksək, r2_gap kiçik)
    # Qalib: r2_test - (r2_gap * 0.5) sınaq balı ilə
    rf_score = rf_res['r2_test'] - (max(0, rf_res['r2_gap']) * 0.5)
    xgb_score = xgb_res['r2_test'] - (max(0, xgb_res['r2_gap']) * 0.5)
    
    winner = rf_res if rf_score >= xgb_score else xgb_res
    
    print("\n" + "="*80)
    print(f"🏆 YEKUN MÜTLƏQ QALİB: {winner['name']}")
    print(f"   ├─ Seçilmə Səbəbi: Ən yüksək test dəqiqliyi və minimal overfitting riski.")
    print(f"   └─ Test R² Score: {winner['r2_test']:.4f} (~{winner['r2_test']*100:.1f}%)")
    print("="*80)

    # Qalib modeli yaddaşa yazırıq
    model_path = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(winner['model'], f)
        
    print(f"\n✅ Qalib ({winner['name']}) yadda saxlanıldı: {model_path}")

if __name__ == "__main__":
    tune_and_train()
