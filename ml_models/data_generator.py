import pandas as pd
import numpy as np
import random
import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Parametrlər
NUM_RECORDS = 10000
ROADS = ["Ziya Bünyadov", "Heydər Əliyev", "Neftçilər", "Babək", "Zərdabi"]
WEATHER_CONDITIONS = ["Açıq", "Yağışlı", "Qarlı", "Dumanlı"]

def generate_traffic_data(num_records=NUM_RECORDS):
    print(f"🤖 Sintetik Data generasiyası başlayır... ({num_records} sətir)")
    
    data = []
    for _ in range(num_records):
        road = random.choice(ROADS)
        hour = random.randint(0, 23)
        day_of_week = random.randint(0, 6) # 0: Bazar ertəsi, 6: Bazar
        is_weekend = 1 if day_of_week >= 5 else 0
        weather = random.choice(WEATHER_CONDITIONS)
        incidents = random.choices([0, 1, 2, 3], weights=[0.8, 0.15, 0.04, 0.01])[0]
        
        # Məntiqi Tıxac (Congestion) Hesablanması (Baku Reallığına uyğun)
        base_congestion = random.randint(10, 30) # Gecə və ya boş vaxtlar
        
        # Pik saatlar (Rush Hour) təsiri (Səhər 7-9, Axşam 17-20)
        if (7 <= hour <= 9) or (17 <= hour <= 20):
            if not is_weekend:
                base_congestion += random.randint(40, 60)
            else:
                base_congestion += random.randint(10, 20)
        elif 10 <= hour <= 16:
            base_congestion += random.randint(10, 30)
            
        # Hava təsiri
        if weather == "Yağışlı":
            base_congestion += random.randint(15, 25)
        elif weather == "Qarlı":
            base_congestion += random.randint(25, 40)
        elif weather == "Dumanlı":
            base_congestion += random.randint(10, 15)
            
        # Qəza təsiri
        if incidents > 0:
            base_congestion += (incidents * random.randint(15, 25))
            
        # Yol spesifikliyi
        if road == "Ziya Bünyadov":
            base_congestion += 10 # Həmişə sıxdır
            
        # 100%-i keçməmək üçün limitləyirik
        congestion_pct = min(base_congestion, 100)
        
        data.append({
            "road": road,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "weather": weather,
            "incidents": incidents,
            "congestion_pct": congestion_pct
        })
        
    df = pd.DataFrame(data)
    
    # Qovluğun mövcudluğunu yoxla
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    
    file_path = os.path.join(os.path.dirname(__file__), "traffic_data.csv")
    df.to_csv(file_path, index=False, encoding='utf-8')
    print(f"✅ Data uğurla yaradıldı və yadda saxlanıldı: {file_path}")
    return df

if __name__ == "__main__":
    generate_traffic_data()
