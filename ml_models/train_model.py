import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ML Best Practices: Always split BEFORE applying encoders/scalers
def load_and_split_data(filepath):
    print("Məlumatlar yüklənir...")
    df = pd.read_csv(filepath)
    
    # Features & Target
    X = df[['hour', 'day_of_week', 'is_weekend', 'weather_condition', 'temperature']]
    y = df['congestion_pct']
    
    # Train, Validation, Test splits (Chronological for time series-like data, or random)
    # Here we use random split for simplicity, but chronological is better for strict time series
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_and_evaluate():
    data_path = '../data/raw/baku_traffic_data.csv'
    if not os.path.exists(data_path):
        print("Xəta: Dataset tapılmadı. Öncə data_generator.py-i işə salın.")
        return
        
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(data_path)
    
    print("Məlumatlar emal edilir (Scaling)...")
    scaler = StandardScaler()
    
    # Fit ONLY on training data to prevent leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("Modellər təlim edilir...")
    # Baseline Model: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_val_scaled)
    
    # Advanced Model: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_val_scaled)
    
    print("\n--- Model Performansları (Validation Set) ---")
    print(f"Linear Regression - MAE: {mean_absolute_error(y_val, lr_preds):.2f}, R2: {r2_score(y_val, lr_preds):.2f}")
    print(f"Random Forest     - MAE: {mean_absolute_error(y_val, rf_preds):.2f}, R2: {r2_score(y_val, rf_preds):.2f}")
    
    # Random Forest is likely better for non-linear time relationships (like rush hour peaks)
    print("\nTest seti üzərində ən yaxşı modelin (Random Forest) yoxlanılması...")
    test_preds = rf_model.predict(X_test_scaled)
    print(f"Test Set - MAE: {mean_absolute_error(y_test, test_preds):.2f}, RMSE: {np.sqrt(mean_squared_error(y_test, test_preds)):.2f}")
    
    # Save the best model and scaler
    os.makedirs('../models', exist_ok=True)
    joblib.dump(rf_model, '../models/traffic_prediction_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("Model və Scaler '../models' qovluğunda saxlanıldı.")

if __name__ == "__main__":
    train_and_evaluate()
