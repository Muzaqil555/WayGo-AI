import os
import pickle
import pandas as pd
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_PATH = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")

# Modeli yaddaşda saxlamaq üçün (Singleton pattern)
_model_pipeline = None

def load_model():
    global _model_pipeline
    if _model_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tapılmadı! Zəhmət olmasa train_model.py faylını işlədin. Yoxlanılan yol: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            _model_pipeline = pickle.load(f)
    return _model_pipeline

def predict_congestion(road: str, hour: int, weather: str, incidents: int = 0) -> float:
    """
    Verilmiş parametrlərə əsasən yoldakı gözlənilən tıxac faizini proqnozlaşdırır.
    """
    model = load_model()
    
    # Bazar günü və ya həftəiçi olduğunu saat/gündən təxmin etmək olar, amma sadələşdirmək üçün 
    # indiki zamanı bazar ertəsi (0) və iş günü qəbul edirik.
    day_of_week = 0 
    is_weekend = 0
    
    # Modelin gözlədiyi DataFrame formatı
    input_data = pd.DataFrame([{
        "road": road,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "weather": weather,
        "incidents": incidents
    }])
    
    # Model proqnoz verir
    prediction = model.predict(input_data)[0]
    
    # Faiz 0-100 arasında olmalıdır
    prediction = max(0.0, min(100.0, float(prediction)))
    return round(prediction, 1)

if __name__ == "__main__":
    # Test
    road = "Ziya Bünyadov"
    hour = 18 # Axşam pik saatı
    weather = "Qarlı"
    
    print(f"🔮 Proqnoz Testi: {road} prospektində saat {hour}:00-da, {weather} havada tıxac necə olacaq?")
    result = predict_congestion(road, hour, weather, incidents=1)
    print(f"📊 AI Proqnozu: Tıxac səviyyəsi {result}% olacaq.")
