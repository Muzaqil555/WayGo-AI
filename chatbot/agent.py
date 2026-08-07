import os
import google.generativeai as genai

# Təhlükəsizlik: API key yalnız .env-dən oxunmalıdır
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Sessiyaları (yaddaşı) saxlayan qlobal lüğət
sessions = {}

SYSTEM_INSTRUCTION = """Sən Bakı şəhərinin rəsmi akıllı nəqliyyat və hərəkətlilik AI asistenti olan 'WayGo Smart AI'sən.
Vacib Qaydalar:
1. Hər cavabında özünü təqdim etmə, BİRBAŞA istifadəçinin sualına cavab ver.
2. ƏGƏR istifadəçi nəqliyyat, yol, tıxac, hava, qəza, naviqasiya və ya Bakı şəhəri ilə bağlı sual verərsə, ona ən uyğun şəkildə kömək et. Datanı sistem özü sənə hər mesajın əvvəlində göndərəcək.
3. ƏGƏR istifadəçi tamam fərqli (məsələn, ümumi dünyagörüşü, tarix, idman və s.) sual verərsə, YALNIZ o suala cavab ver. Qətiyyən cavabın sonuna tıxac və ya hava haqqında məlumat ƏLAVƏ ETMƏ.
4. Azərbaycan dilində son dərəcə nəzakətli, peşəkar, aydın və lüks formatda (HTML emojiləri ilə) qısa və dəqiq cavab ver.
"""

def get_or_create_chat(session_id: str):
    """Verilmiş session_id üçün yaddaşı olan chat obyekti qaytarır və ya yaradır"""
    if session_id not in sessions:
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=SYSTEM_INSTRUCTION
        )
        sessions[session_id] = model.start_chat(history=[])
    return sessions[session_id]

def process_chat(message: str, stats: dict, session_id: str = "default_user") -> str:
    """LLM modelinə canlı statistika ilə zənginləşdirilmiş sorğu göndərib cavabı qaytarır"""
    if not api_key:
        return "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        
    try:
        chat = get_or_create_chat(session_id)
        
        # Dinamik kontekst: İstifadəçinin mesajının əvvəlinə görünməz statistika əlavə edirik
        context_str = (
            f"[SİSTEM MƏLUMATI (Yalnız sənin üçün): "
            f"Tıxac: {stats.get('congestion_pct', 0)}%, "
            f"Sürət: {stats.get('avg_speed', 0):.0f} km/s, "
            f"Hava: {stats.get('weather_cond', 'Bilinmir')}, {stats.get('temp', 0):.1f}°C, "
            f"Qəzalar: {stats.get('incident_count', 0)}]\n"
        )
        
        final_message = context_str + "İstifadəçi mesajı: " + message
        
        response = chat.send_message(final_message)
        return response.text
    except Exception as e:
        print(f"AI Error: {str(e)}")
        raise e
