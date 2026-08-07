import os
import google.generativeai as genai

# Təhlükəsizlik: API key yalnız .env-dən oxunmalıdır
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def generate_prompt(message: str, stats: dict) -> str:
    """Bakı Mobiliti üçün sistem təlimatı (System Prompt) və kontekst"""
    return f"""Sən Bakı şəhərinin rəsmi akıllı nəqliyyat və hərəkətlilik AI asistenti olan 'WayGo Smart AI'sən.
Vacib Qaydalar:
1. Hər cavabında özünü təqdim etmə, BİRBAŞA istifadəçinin sualına cavab ver.
2. ƏGƏR istifadəçi nəqliyyat, yol, tıxac, hava, qəza, naviqasiya və ya Bakı şəhəri ilə bağlı sual verərsə, aşağıdakı CANLI VƏZİYYƏT məlumatlarından istifadə edərək cavab ver.
3. ƏGƏR istifadəçi tamam fərqli (məsələn, ümumi dünyagörüşü, tarix, idman və s.) sual verərsə, YALNIZ o suala cavab ver. Qətiyyən cavabın sonuna tıxac və ya hava haqqında məlumat ƏLAVƏ ETMƏ.

İSTİFADƏÇİ SUALI: "{message}"

BAKININ CANLI REAL-VAXT VƏZİYYƏTİ (Yalnız ehtiyac olduqda istifadə et):
- Ümumi Tıxac İndeksi: {stats.get('congestion_pct', 0)}%
- Ortalama Axın Sürəti: {stats.get('avg_speed', 0):.0f} km/s
- Aktiv Transponder Avtomobil Sayı: {stats.get('active_vehicles', 0)}
- Sinoptik Hava: {stats.get('weather_cond', 'Bilinmir')}, {stats.get('temp', 0):.1f}°C
- Aktiv Qəza və Yol Maneələri Sayı: {stats.get('incident_count', 0)}
- Z-Score Anomaliyaları Sayı: {stats.get('anomaly_count', 0)}

TƏLƏB: Azərbaycan dilində son dərəcə nəzakətli, peşəkar, aydın və lüks formatda (HTML emojiləri ilə) qısa və dəqiq cavab ver. Özünü təqdim etməyə ehtiyac yoxdur."""

def process_chat(message: str, stats: dict) -> str:
    """LLM modelinə sorğu göndərib cavabı qaytarır"""
    if not api_key:
        return "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = generate_prompt(message, stats)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI Error: {str(e)}")
        raise e
