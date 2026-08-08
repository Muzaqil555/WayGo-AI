import sys
import os

from dotenv import load_dotenv
load_dotenv()

# Fix for windows terminal emoji printing
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    
from chatbot.agent import stream_chat

def main():
    print("========================================")
    print("🤖 WayGo AI Local Test Mühiti Başladı")
    print("Çıxmaq üçün 'exit' yazın.")
    print("========================================\n")
    
    # Süni vəziyyət (Extreme Case Test)
    fake_stats = {
        "congestion_pct": 85,
        "avg_speed": 15.0,
        "active_vehicles": 12000,
        "weather_cond": "Qar",
        "temp": -2.5,
        "incident_count": 2,
        "anomaly_count": 1
    }
    
    # Süni Sürücü Profili (Fərdiləşdirmə Testi üçün)
    dummy_profile = {
        "driving_style": "Eco/Təhlükəsiz", 
        "car_type": "Electric Vehicle (EV)", 
        "family_status": "Ailəli",
        "eco_score": 150
    }
    
    print(f"📡 Canlı Data Simulyasiyası: Tıxac {fake_stats['congestion_pct']}%, Hava: {fake_stats['weather_cond']} ({fake_stats['temp']}°C), Qəza: {fake_stats['incident_count']}")
    print(f"👤 Sürücü Profili: {dummy_profile['car_type']}, {dummy_profile['family_status']}")
    print("----------------------------------------")
    
    while True:
        try:
            user_input = input("\nSiz: ")
            if user_input.lower() in ['exit', 'quit', 'çixiş', 'cix']:
                print("Test mühiti bağlandı.")
                break
                
            print("WayGo AI: ", end='', flush=True)
            
            # Streaming cavabı alırıq və gələn kimi ekrana yazdırırıq
            for chunk in stream_chat(user_input, fake_stats, session_id="test_engineer_1", user_profile=dummy_profile):
                print(chunk, end='', flush=True)
                
            print("\n" + "-" * 40)
            
        except Exception as e:
            print(f"\nXəta baş verdi: {e}")
            break

if __name__ == "__main__":
    main()
