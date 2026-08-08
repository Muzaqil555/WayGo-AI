# chatbot/tools.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_current_weather_assumption(weather_from_context: str = None) -> str:
    """
    Kontekstdən hava məlumatı gəlirsə onu istifadə edir.
    Gəlmirsə, fəsilə görə ağıllı ehtimal edir.
    """
    if weather_from_context and weather_from_context.lower() not in ("bilinmir", "unknown", ""):
        return weather_from_context
    from datetime import datetime
    month = datetime.now().month
    if month in (12, 1, 2):
        return "Qarlı"
    elif month in (3, 4, 10, 11):
        return "Yağışlı"
    else:
        return "Açıq"


def predict_future_traffic(road_name: str, hours_ahead: int) -> str:
    """
    Gələcəkdə müəyyən bir yolda tıxacın və ya vəziyyətin necə olacağını təxmin etmək üçün istifadə olunur.
    
    Args:
        road_name (str): Yolun, prospektin və ya küçənin adı.
        hours_ahead (int): Neçə saat sonranı təxmin etmək istədiyiniz (məsələn 2).
        
    Returns:
        str: Proqnozlaşdırılmış vəziyyət barədə məlumat.
    """
    from ml_models.predict import predict_congestion
    from datetime import datetime
    
    current_hour = datetime.now().hour
    target_hour = (current_hour + hours_ahead) % 24
    
    try:
        # Fəsilə görə ağıllı hava ehtimalı (gələcəkdə hava API-dən gələcək)
        weather_assumption = _get_current_weather_assumption()
        predicted_pct = predict_congestion(road=road_name, hour=target_hour, weather=weather_assumption, incidents=0)
        return (
            f"Süni intellekt analizi (ML): {hours_ahead} saat sonra ({target_hour}:00 radələrində) "
            f"{road_name} ərazisində tıxacın təqribən {predicted_pct}% olacağı proqnozlaşdırılır. "
            f"(Hava ehtimalı: {weather_assumption})"
        )
    except Exception as e:
        return f"Proqnoz hesablana bilmədi: {e}"

def find_optimal_route(start_location: str, end_location: str) -> str:
    """
    İki nöqtə arasında tıxacsız, ən qısa və optimal marşrutu tapmaq üçün istifadə olunur.
    
    Args:
        start_location (str): Başlanğıc nöqtə (məs: 28 May).
        end_location (str): Son nöqtə (məs: Gənclik Mall).
        
    Returns:
        str: Məsləhət görülən optimal yol.
    """
    from routing.dijkstra import calculate_optimal_route
    
    try:
        # Fəsilə görə ağıllı hava ehtimalı (gələcəkdə real API-dən gələcək)
        weather_assumption = _get_current_weather_assumption()
        route_result = calculate_optimal_route(start_location, end_location, weather=weather_assumption)
        return route_result
    except Exception as e:
        return f"Marşrut hesablanarkən xəta baş verdi: {e}"
