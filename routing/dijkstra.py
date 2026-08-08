import heapq
import sys
import os
from datetime import datetime

# ML modelini import edirik
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_models.predict import predict_congestion

# Bakı Şəhərinin Sadələşdirilmiş Qraf Xəritəsi (Node -> {Qonşu: (Yol Adı, Baza Vaxtı)})
# Baza vaxtı dəqiqələrlə, tıxacsız (ideal) halda olan hərəkət vaxtıdır.
BAKU_GRAPH = {
    "28 May": {
        "Gənclik": ("Heydər Əliyev", 5),
        "Nizami": ("Zərdabi", 4)
    },
    "Gənclik": {
        "28 May": ("Heydər Əliyev", 5),
        "Koroğlu": ("Ziya Bünyadov", 12),
        "Nərimanov": ("Ziya Bünyadov", 6)
    },
    "Nərimanov": {
        "Gənclik": ("Ziya Bünyadov", 6),
        "Neftçilər": ("Qara Qarayev", 10),
        "Xətai": ("Babək", 8)
    },
    "Koroğlu": {
        "Gənclik": ("Ziya Bünyadov", 12),
        "Neftçilər": ("Qara Qarayev", 5),
        "Hava Yolları": ("Heydər Əliyev", 15)
    },
    "Neftçilər": {
        "Nərimanov": ("Qara Qarayev", 10),
        "Koroğlu": ("Qara Qarayev", 5),
        "Əhmədli": ("Babək", 7)
    },
    "Xətai": {
        "Nərimanov": ("Babək", 8),
        "Əhmədli": ("Neftçilər", 12) # Prospekt
    },
    "Əhmədli": {
        "Neftçilər": ("Babək", 7),
        "Xətai": ("Neftçilər", 12)
    },
    "Nizami": {
        "28 May": ("Zərdabi", 4)
    },
    "Hava Yolları": {
        "Koroğlu": ("Heydər Əliyev", 15)
    }
}

def calculate_optimal_route(start_node: str, end_node: str, weather: str = "Açıq") -> str:
    """
    ML Modelindən tıxac proqnozlarını alaraq Dijkstra alqoritmi ilə ƏN QISA və SÜRƏTLİ yolu tapır.
    """
    if start_node not in BAKU_GRAPH or end_node not in BAKU_GRAPH:
        return f"Üzr istəyirik, {start_node} və ya {end_node} sistemimizin xəritəsində tapılmadı."
        
    current_hour = datetime.now().hour
    
    # Priority Queue for Dijkstra
    pq = [(0, start_node, [])] # (Cəmi Vaxt, Cari Nöqtə, Keçilən Marşrut)
    visited = set()
    
    while pq:
        current_time, current_node, path = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        path = path + [current_node]
        
        if current_node == end_node:
            # Hədəfə çatdıq
            route_str = " -> ".join(path)
            return f"✅ Tıxaclar nəzərə alınaraq Ən Optimal Marşrut:\n🗺️ {route_str}\n⏱️ Təxmini çatma vaxtı: {int(current_time)} dəqiqə."
            
        # Qonşu nöqtələri yoxlayırıq
        for neighbor, (road_name, base_time) in BAKU_GRAPH[current_node].items():
            if neighbor not in visited:
                # Dinamik Ağırlıq (ML Prediction)
                try:
                    congestion = predict_congestion(road_name, current_hour, weather, incidents=0)
                except:
                    congestion = 0
                    
                # Tıxac artdıqca, vaxt da artır. Məsələn 100% tıxac vaxtı 2 dəfə (və ya daha çox) artırır
                penalty_factor = 1.0 + (congestion / 100.0) * 1.5 
                dynamic_time = base_time * penalty_factor
                
                heapq.heappush(pq, (current_time + dynamic_time, neighbor, path))
                
    return "Təəssüf ki, yol tapılmadı."

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    # Test edək
    print("Mərkəzdən (28 May) Əhmədliyə yol axtarılır...\n")
    print("Normal (Açıq) Hava:")
    print(calculate_optimal_route("28 May", "Əhmədli", "Açıq"))
    
    print("\nQarlı Hava (Ziya Bünyadovda tıxac pik həddə çatır):")
    print(calculate_optimal_route("28 May", "Əhmədli", "Qarlı"))
