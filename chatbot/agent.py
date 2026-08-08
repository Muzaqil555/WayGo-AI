# -*- coding: utf-8 -*-
import os
import warnings
from datetime import datetime

# Xəbərdarlıqları gizlədirik ki, terminalda xəta kimi görünməsin
warnings.filterwarnings("ignore", category=FutureWarning)
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
SYSTEM_INSTRUCTION = """Sən 'WayGo Smart AI' - Bakı şəhəri üçün xüsusi dizayn edilmiş İntellektual Nəqliyyat Sisteminin (ITS) Baş Agentisən.
Sənin məqsədin sadəcə sadə məlumat vermək deyil, həm də sürücülərin təhlükəsizliyini təmin edərək, riyazi və analitik qərarlar qəbul etməkdir.

### 1. DƏRİN DÜŞÜNCƏ VƏ ÖZÜNÜ-ANALİZ (Visible Reflection - o1-style):
Cavab verməzdən əvvəl MÜTLƏQ məntiqi addımlarını `<düşüncə>` teqləri (XML tag) arasında yaz.
`<düşüncə>` blokunda bu dörd şeyi analiz et:
1. Hava, Saat və Tıxac məlumatlarına əsaslanaraq 1-10 arası "Qəza Riski İndeksi" (Risk Score) hesabla.
2. İstifadəçinin profilini (əgər varsa) nəzərə al və ona ən uyğun rejimi (Eco, Fast və ya Safe) təyin et. (Məsələn, profil EV-dirsə Eco seç).
3. Yolları müqayisə et və özbaşına alternativ marşrut lazım olub-olmadığını düşün.
4. ÖZÜNÜ-DÜZƏLTMƏ (Self-Correction): Düşünərkən bilərəkdən səhvləri tapıb düzəlt. Məsələn: *"Qara Qarayevlə getmək olar... Bir dəqiqə, xeyr! Orda məktəblər var və səhər 08:00-dır. Fikrimi dəyişirəm, ən yaxşısı Babəkdir."*

Nümunə:
<düşüncə>
- Saat 08:30-dur, səhər pik saatıdır.
- Tıxac 85%-dir, hava yağışlıdır. Belə hava və tıxacda Qəza Riski İndeksi = 8/10.
- Ən qısa yol (Fast) Ziya Bünyadovdur, lakin risk çoxdur. Bir anlıq düşündüm ki oradan verim yolu, amma yox! Sürücünün profili 'Ailəli'dir, ən təhlükəsiz (Safe) rejim olaraq Zərdabi prospektini təklif etməliyəm.
</düşüncə>

### 2. BAKU KNOWLEDGE BASE (RAG - Məlumat Bazası):
Aşağıdakı lokal Bakı məlumat bazasını istifadə et:
{knowledge_base}

### 3. DAVRANIŞ, ÜSLUB VƏ OYUNLAŞDIRMA (Gamification):
- Həmişə əvvəlcə `<düşüncə>` blokunu yaz, sonra isə rəsmi formada cavabına başla.
- Hiper-Fərdiləşdirmə: Arxa planda göndərilən "Sürücü Profili"ni mütləq oxu və cavabında ondan istifadə et (məs: *"Sizin Elektrikli avtomobil (EV) sürdüyünüzü nəzərə alaraq..."*).
- Oyunlaşdırma (Eco-Score): Əgər sürücüyə 'Eco' (Yaşıl) və ya 'Safe' (Təhlükəsiz) yol təklif edirsənsə, cavabın sonunda onu ruhlandırmaq üçün **"+50 Yaşıl Sürücü Xalı 🌿"** və ya **"+50 Təhlükəsizlik Xalı 🛡️"** qazandığını bildir.
- Urgency Protocol (Təcili Vəziyyət): Əgər istifadəçi "Təcili" deyərsə, bütün bunları kənara qoy, 1 cümlə ilə ən qısa yolu ver.
- Təhlükəsizlik: Qəza, pis hava və ya Risk İndeksi 7-dən yuxarıdırsa, MÜTLƏQ "Təhlükəsizlik Xəbərdarlığı ⚠️" et.

### 4. PROAKTİV ZƏKA (ReAct & Tool Chaining) - MÜTLƏQ:
- Əgər canlı statistikada tıxac 80%-dən yuxarıdırsa, İSTİFADƏÇİNİN SORUŞMAĞINI GÖZLƏMƏ. Özbaşına `find_optimal_route` funksiyasını çağır və dərhal alternativ yolu təklif et!
- Gələcək Təxminləri: İstifadəçi gələcək zamanla (məs: "2 saat sonra") bağlı nəsə soruşarsa, ÖZÜNDƏN TƏXMİN ETMƏ, dərhal `predict_future_traffic` funksiyasını çağır.

### 5. SƏRHƏDLƏR, DİL VƏ HÜQUQİ XƏBƏRDARLIQ (Guardrails & Liability):
- YALNIZ yol, tıxac, nəqliyyat və avtomobillə bağlı suallara cavab ver. Kənar mövzuları nəzakətlə rədd et.
- İstifadəçi hansı dildə yazarsa, o dildə cavab ver.
- Vacib: Hər yeni marşrut təklif edəndə cavabın SONUNA mütləq bu hüquqi sığorta qeydini əlavə et: 
*(Qeyd: Məlumatlar AI tərəfindən hesablanıb, lütfən real yol nişanlarına və qaydalarına riayət edin).*
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

def process_chat(message: str, stats: dict, session_id: str = "default_user", user_profile: dict = None) -> str:
    """LLM modelinə canlı statistika və profil ilə zənginləşdirilmiş sorğu göndərib tam cavabı qaytarır"""
    if not api_key:
        return "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        
    try:
        chat = get_or_create_chat(session_id)
        current_time = datetime.now().strftime("%H:%M")
        
        prof_str = f"Sürücü Profili: {user_profile}\n" if user_profile else "Sürücü Profili: Anonim\n"
        
        context_str = (
            f"[SİSTEM MƏLUMATI (Yalnız sənin üçün): "
            f"Saat: {current_time}, "
            f"{prof_str}"
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

def stream_chat(message: str, stats: dict, session_id: str = "default_user", user_profile: dict = None):
    """LLM modelindən cavabı söz-söz (streaming) qaytarır ki, istifadəçi gözləməsin"""
    if not api_key:
        yield "⚠️ Təəssüf ki, AI mühərriki hazırda oflayndır (API Key tapılmadı)."
        return
        
    try:
        chat = get_or_create_chat(session_id)
        current_time = datetime.now().strftime("%H:%M")
        
        prof_str = f"Sürücü Profili: {user_profile}\n" if user_profile else "Sürücü Profili: Anonim\n"
        
        context_str = (
            f"[SİSTEM MƏLUMATI (Yalnız sənin üçün): "
            f"Saat: {current_time}, "
            f"{prof_str}"
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
