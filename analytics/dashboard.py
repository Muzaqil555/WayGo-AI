import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Windows terminali üçün encoding
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Şəkillərin yadda saxlanılacağı qovluğu yoxlayırıq
analytics_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(analytics_dir):
    os.makedirs(analytics_dir)

# Datanın yolu
data_path = os.path.join(os.path.dirname(analytics_dir), 'ml_models', 'traffic_data.csv')

def generate_dashboard():
    print("📊 Datanın Analizi və Qrafiklərin (Dashboard) Hazırlanmasına Başlanılır...")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("❌ Xəta: traffic_data.csv tapılmadı. Zəhmət olmasa ml_models/data_generator.py skriptini işlədin.")
        return

    # Qrafiklərin ümumi dizaynını (Seaborn) təyin edirik
    sns.set_theme(style="whitegrid", palette="muted")
    
    # --- 1. Saatlara görə Orta Tıxac Faizi (Line Chart) ---
    plt.figure(figsize=(10, 6))
    hourly_traffic = df.groupby('hour')['congestion_pct'].mean().reset_index()
    sns.lineplot(data=hourly_traffic, x='hour', y='congestion_pct', marker='o', color='b', linewidth=2.5)
    
    plt.title("⏰ Saatlara Görə Bakı Yollarında Orta Tıxac Sıxlığı", fontsize=14, fontweight='bold')
    plt.xlabel("Günün Saatı (0-23)", fontsize=12)
    plt.ylabel("Orta Tıxac Faizi (%)", fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(analytics_dir, 'saat_tixac_trend.png'), dpi=300)
    plt.close()
    print("✅ 1. 'saat_tixac_trend.png' yaradıldı.")
    
    # --- 2. Hava Şəraitinin Tıxaca Təsiri (Bar Chart) ---
    plt.figure(figsize=(10, 6))
    weather_traffic = df.groupby('weather')['congestion_pct'].mean().reset_index().sort_values(by='congestion_pct', ascending=False)
    sns.barplot(data=weather_traffic, x='weather', y='congestion_pct', hue='weather', palette="Reds_r", legend=False)
    
    plt.title("🌦️ Hava Şəraitinin Tıxaca Təsiri", fontsize=14, fontweight='bold')
    plt.xlabel("Hava Durumu", fontsize=12)
    plt.ylabel("Orta Tixac Faizi (%)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(analytics_dir, 'hava_tixac_tesiri.png'), dpi=300)
    plt.close()
    print("✅ 2. 'hava_tixac_tesiri.png' yaradıldı.")
    
    # --- 3. Ən Təhlükəli (Ən çox qəza olan) Yollar ---
    plt.figure(figsize=(12, 6))
    road_incidents = df.groupby('road')['incidents'].sum().reset_index().sort_values(by='incidents', ascending=False).head(10)
    sns.barplot(data=road_incidents, y='road', x='incidents', hue='road', palette="magma", legend=False)
    
    plt.title("⚠️ Bakının Ən Çox Qəza Və Ləngimə Olan 10 Yolu", fontsize=14, fontweight='bold')
    plt.xlabel("Ümumi Qəza və Ləngimə Sayı", fontsize=12)
    plt.ylabel("Yol/Prospekt", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(analytics_dir, 'tehlukeli_yollar.png'), dpi=300)
    plt.close()
    print("✅ 3. 'tehlukeli_yollar.png' yaradıldı.")
    
    print("\n🎉 Bütün qrafiklər uğurla 'analytics' qovluğuna yadda saxlanıldı! İnvestorlara təqdimata tam hazırıq.")

if __name__ == "__main__":
    generate_dashboard()
