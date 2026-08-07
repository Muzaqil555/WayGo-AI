import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_baku_traffic_data(num_records=5000):
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_records)]
    
    # Feature generation
    hours = [d.hour for d in dates]
    day_of_weeks = [d.weekday() for d in dates]
    is_weekend = [1 if d > 4 else 0 for d in day_of_weeks]
    
    # Weather conditions (0: Clear, 1: Rain, 2: Snow, 3: Fog)
    weather = np.random.choice([0, 1, 2, 3], size=num_records, p=[0.7, 0.2, 0.05, 0.05])
    temperature = np.random.normal(15, 10, num_records)
    
    # Target: Congestion Percentage (0-100)
    # Base congestion depends on hour
    base_congestion = np.array([
        10 if (h < 6 or h > 22) else 
        80 if (7 <= h <= 9 or 17 <= h <= 19) else 
        40 for h in hours
    ])
    
    # Weekend effect: much lower congestion during peak hours
    weekend_multiplier = np.array([0.5 if w == 1 else 1.0 for w in is_weekend])
    
    # Weather effect: worse weather = more congestion
    weather_multiplier = np.array([1.0 + (w * 0.15) for w in weather])
    
    congestion_pct = base_congestion * weekend_multiplier * weather_multiplier
    # Add random noise
    congestion_pct += np.random.normal(0, 5, num_records)
    # Clip to 0-100
    congestion_pct = np.clip(congestion_pct, 0, 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'hour': hours,
        'day_of_week': day_of_weeks,
        'is_weekend': is_weekend,
        'weather_condition': weather,
        'temperature': temperature,
        'congestion_pct': congestion_pct
    })
    
    os.makedirs('../data/raw', exist_ok=True)
    df.to_csv('../data/raw/baku_traffic_data.csv', index=False)
    print(f"Generated {num_records} records of synthetic traffic data at ../data/raw/baku_traffic_data.csv")

if __name__ == "__main__":
    generate_baku_traffic_data()
