import os
import google.generativeai as genai

# Təhlükəsizlik: API key yalnız .env-dən oxunmalıdır
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# RAG: Bakı Məlumat Bazasını yükləyirik
import json
try:
    kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
    with open(kb_path, 'r', encoding='utf-8') as f:
        knowledge_base = f.read()
except Exception as e:
    knowledge_base = "Baza tapılmadı."

# Sessiyaları (yaddaşı) saxlayan qlobal lüğət
sessions = {}
SYSTEM_INSTRUCTION = """Sən 'WayGo Smart AI' - Bakı şəhəri üçün xüsusi hazırlanmış yüksək səviyyəli (enterprise-grade) naviqasiya və yol hərəkəti köməkçisisən.

### 1. SƏNİN ROLUN VƏ ŞƏXSİYYƏTİN:
- Həmişə peşəkar, dəqiq və nəzakətlisən.
- Müraciət forması: Cavablarına həmişə "Hörmətli sürücü" və ya "Hörmətli istifadəçi" kimi rəsmi formada başla və "Yolunuz açıq olsun!" və ya "Təhlükəsiz səyahətlər!" ilə bitir.
- Format: Məlumatları oxunaqlı etmək üçün siyahılardan (bullet points), qalın şriftlərdən (bold) və uyğun emojilərdən (🚗, 🚦, ⚠️, 🌧️) istifadə et.

### 2. BAKU KNOWLEDGE BASE (RAG - Məlumat Bazası):
- Aşağıdakı məlumatlar sənə Bakının yolları haqqında əsas qaydaları verir. Əgər istifadəçi bu yollardan biri barədə soruşarsa, mütləq sürət həddini və radarı xatırlat:
{knowledge_base}

### 2. MƏLUMAT VƏ MƏNTİQ:
- Həmişə sənə arxa planda [SİSTEM MƏLUMATI] başlığı altında göndərilən canlı statistikalara (tıxac, sürət, hava, qəza) əsaslan. Məlumat uydurma.
- Tıxac Təhlili: Əgər tıxac 70%-dən çoxdursa, HƏMİŞƏ alternativ marşrutlar təklif et (məs: "Ziya Bünyadov əvəzinə Zərdabi prospekti ilə getmək").
- Təhlükəsizlik: Əgər hava şəraiti pisdirsə (Qar, Yağış, Şaxta) və ya yolda qəza varsa, MÜTLƏQ "Təhlükəsizlik Xəbərdarlığı" başlığı altında sürəti azaltmağı tövsiyə et.

### 3. ALƏTLƏRDƏN (TOOLS) İSTİFADƏ QAYDASI:
- Əgər istifadəçi GƏLƏCƏKLƏ bağlı sual verərsə (məs: "2 saat sonra necə olacaq?", "Axşam tıxac olar?"), MÜTLƏQ 'predict_future_traffic' funksiyasını işə sal. Özündən təxmin etmə.
- Əgər istifadəçi iki nöqtə arasında yol axtarırsa, MÜTLƏQ 'find_optimal_route' funksiyasını işə sal.

### 4. SƏRHƏDLƏR (GUARDRAILS) - ÇOX ÖNƏMLİ:
- Sən YALNIZ yol, tıxac, nəqliyyat, naviqasiya və avtomobillərlə bağlı suallara cavab verirsən.
- Əgər istifadəçi kənar mövzularda (siyasət, resept, kodlaşdırma, tarix və s.) sual soruşarsa, ÇOX NƏZAKƏTLƏ bildir ki, sən yalnız WayGo naviqasiya ekspertisən və kənar mövzulara cavab vermirsən.

### 6. ÇOXDİLLİ DƏSTƏK (MULTI-LANGUAGE):
- Əgər istifadəçi səninlə İngilis (English) və ya Rus (Русский) dilində danışarsa, sən də avtomatik olaraq həmin dildə cavab ver. Şüarları (Yolunuz açıq olsun) da həmin dilə uyğunlaşdır.

### 7. EMOSİYA VƏ PSİXOLOJİ TƏHLİL (SENTIMENT ANALYSIS):
- İstifadəçinin yazışma tonunu analiz et. Əgər istifadəçi əsəbidirsə, şikayət edirsə (məs: "bezdim", "bu nə tıxacdır", "çox pis"), ona dəqiq məlumat verməzdən əvvəl MÜTLƏQ YÜKSƏK EMPATİYA göstər və sakitləşdirici, psixoloji dəstəkverici cümlələr (məs: "Sizi çox yaxşı başa düşürəm, tıxacda qalmaq doğrudan da yorucudur...") istifadə et.
"""

def get_or_create_chat(session_id: str):
    """Verilmiş session_id üçün yaddaşı olan chat obyekti qaytarır və ya yaradır"""
    if session_id not in sessions:
        from chatbot.tools import predict_future_traffic, find_optimal_route
        model = genai.GenerativeModel(
            model_name='gemini-flash-lite-latest',
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[predict_future_traffic, find_optimal_route]
        )
        sessions[session_id] = model.start_chat(
            history=[], 
            enable_automatic_function_calling=True
        )
    return sessions[session_id]

def process_chat(message: str, stats: dict, session_id: str = "default_user") -> str:
    """LLM modelinə canlı statistika ilə zənginləşdirilmiş sorğu göndərib tam cavabı qaytarır (Gecikir)"""
    if not api_key:
        return "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        
    try:
        chat = get_or_create_chat(session_id)
        
        context_str = (
            f"[SİSTEM MƏLUMATI (Yalnız sənin üçün): "
            f"Tıxac: {stats.get('congestion_pct', 0)}%, "
            f"Sürət: {stats.get('avg_speed', 0):.0f} km/s, "
            f"Hava: {stats.get('weather_cond', 'Bilinmir')}, {stats.get('temp', 0):.1f}°C, "
            f"Qəzalar: {stats.get('incident_count', 0)}]\n"
        )
        
        final_message = context_str + "İstifadəçi mesajı: " + message
        
        # Sığorta (Retry Logic): İnternet və ya API xətası olarsa 3 dəfəyədək yenidən yoxla
        import time
        for attempt in range(3):
            try:
                response = chat.send_message(final_message)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(1.0)
                
        # Yaddaşın idarəolunması (Memory Leak-in qarşısını almaq): Yalnız son 10 dialoqu saxla
        while len(chat.history) > 10:
            chat.history.pop(0) # İstifadəçi mesajını sil
            chat.history.pop(0) # Model mesajını sil
            
        return response.text
    except Exception as e:
        print(f"AI Error: {str(e)}")
        raise e

def stream_chat(message: str, stats: dict, session_id: str = "default_user"):
    """LLM modelindən cavabı söz-söz (streaming) qaytarır ki, istifadəçi gözləməsin"""
    if not api_key:
        yield "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        return
        
    try:
        chat = get_or_create_chat(session_id)
        
        context_str = (
            f"[SİSTEM MƏLUMATI (Yalnız sənin üçün): "
            f"Tıxac: {stats.get('congestion_pct', 0)}%, "
            f"Sürət: {stats.get('avg_speed', 0):.0f} km/s, "
            f"Hava: {stats.get('weather_cond', 'Bilinmir')}, {stats.get('temp', 0):.1f}°C, "
            f"Qəzalar: {stats.get('incident_count', 0)}]\n"
        )
        
        final_message = context_str + "İstifadəçi mesajı: " + message
        
        # Alətlərdən istifadə üçün automatic function calling aktiv edirik
        # Streaming əvəzinə tam cavabı alıb sonra sürətlə axın edirik (fake stream)
        # Sığorta (Retry Logic) əlavə edildi
        import time
        response = None
        for attempt in range(3):
            try:
                response = chat.send_message(final_message, stream=False)
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(1.0)
        
        # Yaddaşın idarəolunması (Memory Leak): Yalnız son 10 dialoqu saxla
        while len(chat.history) > 10:
            chat.history.pop(0) # İstifadəçi mesajını sil
            chat.history.pop(0) # Model mesajını sil
        
        # Əgər function çağırılıbsa SDK avtomatik həll edir
        text = response.text if response.text else ""
        for word in text.split(' '):
            yield word + ' '
            time.sleep(0.01)
    except Exception as e:
        print(f"AI Error: {str(e)}")
        yield f"\n[Xəta baş verdi: {str(e)}]"
