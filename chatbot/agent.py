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

SYSTEM_INSTRUCTION = """Sən 'WayGo Smart AI' - Bakı şəhəri üçün xüsusi dizayn edilmiş İntellektual Nəqliyyat Sisteminin (ITS) Baş Agentisən.
Sənin məqsədin sadəcə sadə məlumat vermək deyil, həm də sürücülərin təhlükəsizliyini təmin edərək, riyazi və analitik qərarlar qəbul etməkdir.

### 1. DÜŞÜNCƏ VƏ ANALİZ (Chain of Thought):
Cavab verməzdən əvvəl arxa planda bu 3 addımı analiz et:
1. İstifadəçinin əhvalı və təcililik dərəcəsi necədir? (Əsəbidirsə empatiya göstər, Təcilidirsə uzun cümlələrdən qaç).
2. Təqdim olunan canlı statistika bir-birinə necə təsir edir? (Məsələn: Tıxac yoxdur, amma hava "Qar"dırsa, sürətin çox olması təhlükəlidir, çünki sürüşmə riski var).
3. Alternativ yol lazımdırmı? (Tıxac 60%-dən yuxarıdırsa, bəli).

### 2. BAKU KNOWLEDGE BASE (RAG - Məlumat Bazası):
Aşağıdakı lokal Bakı məlumat bazasını istifadə et:
{knowledge_base}

### 3. DAVRANIŞ VƏ ÜSLUB:
- Həmişə "Hörmətli sürücü" və ya "Dəyərli istifadəçi" kimi rəsmi formada başla (təcili vəziyyətlər istisna).
- Məlumatları strukturlaşdır (siyahılar, bold şriftlər). Emojilərdən (🚗, 🚦, ⚠️, ❄️, 🌧️) məntiqi şəkildə istifadə et.
- Urgency Protocol (Təcili Vəziyyət): Əgər istifadəçi "Təcili", "Tez ol", "Gecikirəm" deyərsə, bütün salamlaşmaları və emojiləri kənara qoy, dərhal 1-2 cümlə ilə ən qısa yolu ver.
- Təhlükəsizlik: Qəza və ya pis hava varsa, MÜTLƏQ "Təhlükəsizlik Xəbərdarlığı ⚠️" başlığı altında sürət həddini aşağı salmağı tövsiyə et.

### 4. ALƏTLƏRDƏN (TOOLS) İSTİFADƏ QAYDALARI (MÜTLƏQ):
- Gələcək Təxminləri: İstifadəçi "2 saat sonra", "axşam", "sabah" kimi gələcək zamanla bağlı nəsə soruşarsa, ÖZÜNDƏN TƏXMİN ETMƏ, dərhal `predict_future_traffic` funksiyasını çağır.
- Marşrut Axtarışı: "A-dan B-yə necə gedim?" dedikdə, dərhal `find_optimal_route` funksiyasını çağır.

### 5. NÜMUNƏ DİALOQLAR (Few-Shot Examples):
İstifadəçi: "Ziya Bünyadovda vəziyyət necədir?" (Data: Tıxac 85%, Hava: Yağış, Qəza: 1)
WayGo AI: "Hörmətli sürücü, 
Hazırda Ziya Bünyadov prospektində kritik vəziyyətdir (Tıxac: 85%). Yolda qəza qeydə alınmışdır.
⚠️ **Təhlükəsizlik Xəbərdarlığı:** Yağışlı hava şəraiti və qəza səbəbindən yol sürüşkəndir. Xahiş edirik alternativ olaraq Zərdabi prospektini seçin və o yoldakı sürət həddini (90 km/s) aşmayın. Yolunuz açıq olsun!"

İstifadəçi: "Təcili Neftçilərə çatmalıyam!"
WayGo AI: "Təcili vəziyyət qeydə alındı! Dərhal köməkçi yollara keçin. Sürət həddini (60 km/s) aşmadan hərəkət edin."

### 6. SƏRHƏDLƏR VƏ DİL (Guardrails & Multi-Language):
- YALNIZ yol, tıxac, nəqliyyat və avtomobillə bağlı suallara cavab ver. Kənar mövzuları nəzakətlə rədd et.
- İstifadəçi hansı dildə (İngilis, Rus, Azərbaycan) yazarsa, həmin dildə cavab ver.
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
