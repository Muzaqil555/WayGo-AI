import os
import google.generativeai as genai

# Təhlükəsizlik: API key yalnız .env-dən oxunmalıdır
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Sessiyaları (yaddaşı) saxlayan qlobal lüğət
sessions = {}

SYSTEM_INSTRUCTION = """Sən Bakı şəhərinin rəsmi ağıllı nəqliyyat və hərəkətlilik AI asistenti olan 'WayGo Smart AI'sən.
Sənin əsas vəzifən istifadəçilərə yollardakı vəziyyət, tıxaclar, qəzalar və alternativ marşrutlar barədə YÜKSƏK PEŞƏKAR, lüks və dəqiq məlumat verməkdir.

Bakı Yolları Məlumat Bazası (Baku Knowledge Base):
- Əsas Prospektlər: Heydər Əliyev prospekti (Hava limanı yolu), Ziya Bünyadov prospekti (Dərnəgül yolu), Neftçilər prospekti (Bulvar kənarı), Tbilisi prospekti, Babək prospekti, Qara Qarayev prospekti.
- Kritik Qovşaqlar: 20 Yanvar dairəsi, Qələbə dairəsi, 3-cü mikrorayon dairəsi, Koroğlu metrosu ətrafı.

Qaydalar və Məntiq (AI Logic):
1. Tıxac (Congestion) 70%-dən yuxarıdırsa: HƏMİŞƏ alternativ yol təklif et. (Məsələn: Ziya Bünyadovda tıxac varsa, alternativ kimi Zərdabi və ya digər küçələri düşün).
2. Qəza (Incident) sayı 0-dan böyükdürsə: Mütləq "Təhlükəsizlik Xəbərdarlığı" et və sürət həddini aşağı salmağı tövsiyə et.
3. Hava vəziyyəti: Əgər yağış və ya qar yağırsa, sürüşkən yollar barədə xəbərdarlıq et.
4. Tamamilə kənar sual verilərsə (məs: tarix, proqramlaşdırma): Çox qısa cavab ver, amma SONUNDA həmişə WayGo AI olduğunu xatırlat.
5. Üslub: Azərbaycan dilində son dərəcə nəzakətli, rəsmi, ağıllı və aralarda HTML emojiləri olan (🚗, 🚦, ⚠️) peşəkar formatda yaz. Heç vaxt uzun-uzadı nağıl danışma, lakonik və konkret ol. Özünü təqdim etmə (əgər soruşulmursa).
"""

def get_or_create_chat(session_id: str):
    """Verilmiş session_id üçün yaddaşı olan chat obyekti qaytarır və ya yaradır"""
    if session_id not in sessions:
        model = genai.GenerativeModel(
            model_name='gemini-flash-lite-latest',
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
