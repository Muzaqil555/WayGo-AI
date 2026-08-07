# chatbot/tools.py

def predict_future_traffic(road_name: str, hours_ahead: int) -> str:
    """
    Gələcəkdə müəyyən bir yolda tıxacın və ya vəziyyətin necə olacağını təxmin etmək üçün istifadə olunur.
    
    Args:
        road_name (str): Yolun, prospektin və ya küçənin adı.
        hours_ahead (int): Neçə saat sonranı təxmin etmək istədiyiniz (məsələn 2).
        
    Returns:
        str: Proqnozlaşdırılmış vəziyyət barədə məlumat.
    """
    # QEYD: Hazırda bu funksiya ML (Machine Learning) modelinə qoşulmayıb deyə simulyasiya qaytarır.
    # Növbəti mərhələdə ml_models qovluğundakı əsl AI modelinə burdan sorğu atacağıq.
    
    return f"Süni intellekt analizi: {hours_ahead} saat sonra {road_name} ərazisində tıxacın 30% artacağı və hərəkətin çətinləşəcəyi proqnozlaşdırılır."

def find_optimal_route(start_location: str, end_location: str) -> str:
    """
    İki nöqtə arasında tıxacsız, ən qısa və optimal marşrutu tapmaq üçün istifadə olunur.
    
    Args:
        start_location (str): Başlanğıc nöqtə (məs: 28 May).
        end_location (str): Son nöqtə (məs: Gənclik Mall).
        
    Returns:
        str: Məsləhət görülən optimal yol.
    """
    # Gələcəkdə routing/dijkstra.py bura qoşulacaq
    return f"Ən optimal yol: {start_location} nöqtəsindən {end_location} nöqtəsinə alternativ olaraq Ziya Bünyadov prospekti ilə getmək vaxta 15 dəqiqə qənaət edəcək."
