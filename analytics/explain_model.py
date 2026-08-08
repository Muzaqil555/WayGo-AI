import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Şəkillərin yadda saxlanılacağı qovluq
analytics_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(analytics_dir)

def explain_ai():
    print("🧠 [XAI - Explainable AI] Modelin düşüncə tərzi analiz edilir...")
    
    # 1. Modeli yüklə
    model_path = os.path.join(base_dir, 'ml_models', 'traffic_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_pipeline = pickle.load(f)
    except FileNotFoundError:
        print("❌ Model tapılmadı!")
        return
        
    # 2. Xüsusiyyətləri (Features) və Onların Əhəmiyyətini (Importance) Çıxart
    rf_model = model_pipeline.named_steps['regressor']
    preprocessor = model_pipeline.named_steps['preprocessor']
    
    # Rəqəmsal sütunların adları (train_model.py ilə eyni)
    numeric_features = ['hour', 'day_of_week', 'is_weekend', 'incidents']
    
    # Kateqoriyalı sütunların adlarını OneHotEncoder-dən alırıq
    ohe = preprocessor.named_transformers_['cat']
    cat_features = ohe.get_feature_names_out(['road', 'weather'])
    
    # Bütün sütun adlarını birləşdir
    all_features = numeric_features + list(cat_features)
    
    # Modelin beyin xəritəsi (importances)
    importances = rf_model.feature_importances_
    
    # Pandas DataFrame yarat
    df_importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    })
    
    # Bizə ancaq ən vacib ilk 15 amil lazımdır (investora göstərmək üçün)
    df_top_15 = df_importances.sort_values(by='Importance', ascending=False).head(15)
    
    # Adları daha vizual etmək
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('road_', 'Yol: ')
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('weather_', 'Hava: ')
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('hour', 'Günün Saatı')
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('incidents', 'Qəzaların Sayı')
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('is_weekend', 'Həftəsonu')
    df_top_15['Feature'] = df_top_15['Feature'].str.replace('day_of_week', 'Həftənin Günü')
    
    # --- Qrafik Çəkmək (SHAP Style Feature Importance) ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Qradient rəng
    sns.barplot(
        data=df_top_15, 
        x='Importance', 
        y='Feature', 
        palette="viridis"
    )
    
    plt.title("🧠 Süni İntellektin Qərar Xəritəsi (Tıxac Nədən Yaranır?)", fontsize=16, fontweight='bold')
    plt.xlabel("Qərara Təsir Faizi (Riyazi Çəki)", fontsize=12)
    plt.ylabel("Amillər", fontsize=12)
    
    # Xətlər
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # PNG kimi yadda saxla
    output_path = os.path.join(analytics_dir, 'ai_brain_map.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ XAI Qrafiki uğurla yaradıldı: {output_path}")
    print("📈 İnvestorlar artıq modelin qərarlarının 'Niyə'sini riyazi olaraq görə biləcəklər.")

if __name__ == "__main__":
    explain_ai()
