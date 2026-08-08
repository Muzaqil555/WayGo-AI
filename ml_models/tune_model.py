import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
        fit_status = "⚠️ UNDERFITTING (Model zəif öyrənib)"
    elif r2_gap > 0.08:
        fit_status = f"⚠️ OVERFITTING (Əzbərləyib: Gap = {r2_gap*100:.1f}%)"
    elif r2_gap > 0.04:
        fit_status = f"⚡ MÜLAYİM OVERFITTING (Gap = {r2_gap*100:.1f}%)"
    else:
        fit_status = "✅ MÜKƏMMƏL BALANS"

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
    print("🏆 [ML World Championship] 5 Böyük Mühərrik Arasında Dünya Çempionatı")
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
    
    results = []

    # -------------------------------------------------------------------------
    # 1. RANDOM FOREST
    # -------------------------------------------------------------------------
    print("\n⚙️  [1/5] Random Forest Regressor (5-Fold Cross-Validation)...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rf_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__max_depth': [10, 15],
        'regressor__min_samples_split': [5, 10]
    }
    rf_search = GridSearchCV(rf_pipeline, rf_grid, cv=5, n_jobs=-1, scoring='r2')
    rf_search.fit(X_train, y_train)
    rf_res = evaluate_model_fit(rf_search.best_estimator_, X_train, X_test, y_train, y_test, "Random Forest")
    results.append(rf_res)

    # -------------------------------------------------------------------------
    # 2. XGBOOST
    # -------------------------------------------------------------------------
    print("⚙️  [2/5] XGBoost Regressor (5-Fold Cross-Validation)...")
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, verbosity=0))
    ])
    xgb_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__max_depth': [4, 6],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__reg_alpha': [0.1, 1.0]
    }
    xgb_search = GridSearchCV(xgb_pipeline, xgb_grid, cv=5, n_jobs=-1, scoring='r2')
    xgb_search.fit(X_train, y_train)
    xgb_res = evaluate_model_fit(xgb_search.best_estimator_, X_train, X_test, y_train, y_test, "XGBoost")
    results.append(xgb_res)

    # -------------------------------------------------------------------------
    # 3. LIGHTGBM
    # -------------------------------------------------------------------------
    print("⚙️  [3/5] LightGBM Regressor (5-Fold Cross-Validation)...")
    lgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LGBMRegressor(random_state=42, verbose=-1))
    ])
    lgb_grid = {
        'regressor__n_estimators': [150, 250],
        'regressor__max_depth': [4, 6],
        'regressor__learning_rate': [0.05, 0.1],
        'regressor__num_leaves': [15, 31]
    }
    lgb_search = GridSearchCV(lgb_pipeline, lgb_grid, cv=5, n_jobs=-1, scoring='r2')
    lgb_search.fit(X_train, y_train)
    lgb_res = evaluate_model_fit(lgb_search.best_estimator_, X_train, X_test, y_train, y_test, "LightGBM")
    results.append(lgb_res)

    # -------------------------------------------------------------------------
    # 4. CATBOOST
    # -------------------------------------------------------------------------
    print("⚙️  [4/5] CatBoost Regressor (5-Fold Cross-Validation)...")
    cat_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor(random_state=42, verbose=0))
    ])
    cat_grid = {
        'regressor__iterations': [200, 350],
        'regressor__depth': [4, 6],
        'regressor__learning_rate': [0.05, 0.1]
    }
    cat_search = GridSearchCV(cat_pipeline, cat_grid, cv=5, n_jobs=-1, scoring='r2')
    cat_search.fit(X_train, y_train)
    cat_res = evaluate_model_fit(cat_search.best_estimator_, X_train, X_test, y_train, y_test, "CatBoost")
    results.append(cat_res)

    # -------------------------------------------------------------------------
    # 5. VOTING ENSEMBLE (Hibrid Super-Model: XGBoost + LightGBM + CatBoost)
    # -------------------------------------------------------------------------
    print("⚙️  [5/5] Voting Ensemble (Hibrid Super-Model: XGBoost + LightGBM + CatBoost)...")
    ensemble_model = VotingRegressor(estimators=[
        ('xgb', xgb_search.best_estimator_),
        ('lgb', lgb_search.best_estimator_),
        ('cat', cat_search.best_estimator_)
    ], weights=[1, 1, 1])
    ensemble_model.fit(X_train, y_train)
    ensemble_res = evaluate_model_fit(ensemble_model, X_train, X_test, y_train, y_test, "Voting Ensemble (Hibrid)")
    results.append(ensemble_res)

    # -------------------------------------------------------------------------
    # MÜQAYİSƏ CƏDVƏLİ VƏ YEKUN QALİB
    # -------------------------------------------------------------------------
    print("\n" + "="*95)
    print("📊 5 MÜHƏRRİKİN HƏRTƏRƏFLİ PERFORMANCE VƏ OVERFITTING ANALİZ CƏDVƏLİ")
    print("="*95)
    print(f"{'Model Name':<25} | {'Train R²':<9} | {'Test R²':<9} | {'Gap (R²)':<9} | {'Test MAE':<9} | {'Status'}")
    print("-" * 95)
    
    best_score = -999
    winner_model = None
    
    for r in results:
        # Penallaşdırılmış bal: Test R2 yüksək olsun, Overfitting Gap-i cəzalandırırıq
        net_score = r['r2_test'] - (max(0, r['r2_gap']) * 0.5)
        print(f"{r['name']:<25} | {r['r2_train']:<9.4f} | {r['r2_test']:<9.4f} | {r['r2_gap']*100:<8.2f}% | {r['mae_test']:<8.2f}% | {r['status']}")
        
        if net_score > best_score:
            best_score = net_score
            winner_model = r

    print("="*95)
    print(f"\n🏆 MÜTLƏQ ÇEMPİON MODEL: {winner_model['name']}")
    print(f"   ├─ Test R² Score        : {winner_model['r2_test']:.4f} (~{winner_model['r2_test']*100:.2f}%)")
    print(f"   ├─ Overfitting Gap      : {winner_model['r2_gap']*100:.2f}%")
    print(f"   ├─ Orta Xəta Payı (MAE) : {winner_model['mae_test']:.2f}%")
    print(f"   └─ Kənarlaşma (RMSE)    : {winner_model['rmse_test']:.2f}%")
    print("="*95)

    # Qalib modeli yaddaşa yazırıq
    model_path = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(winner_model['model'], f)
        
    print(f"\n✅ Çempion Model ({winner_model['name']}) yadda saxlanıldı: {model_path}")

if __name__ == "__main__":
    tune_and_train()
