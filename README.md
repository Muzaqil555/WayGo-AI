# 🚀 WayGo AI — Baku Urban Mobility Intelligence Engine

> **Bakı şəhərinin ağıllı hərəkət mühərriki.** Gemini AI, Maşın Öyrənməsi (Random Forest) və Dijkstra Alqoritmini birləşdirən, real vaxtlı nəqliyyat tıxac proqnozu, optimallaşdırılmış naviqasiya və danışıq qabiliyyətli AI Agent sistemi.

---

## ⚡ Əsas İmkanlar

| Xüsusiyyət | Texnologiya | Dəqiqlik / Sürət |
|---|---|---|
| 🧠 Tıxac Proqnozu | Random Forest (Tuned) | **~90% R² Dəqiqlik** |
| 🗺️ Ağıllı Naviqasiya | Dijkstra + ML | Dinamik çəkili qraf |
| 💬 Danışan AI Agent | Google Gemini 1.5 Pro | Çox-mərhələli dialoq |
| 📊 Data Vizualizasiyası | Seaborn + Matplotlib | 3 investora hazır qrafik |
| 🔍 İzah Edilə bilən AI | XAI (Feature Importance) | Şəffaf qərar xəritəsi |
| 📝 Server İzləmə | Python Logging | Gündəlik `.log` faylları |

---

## 🏗️ Sistem Arxitekturası

```
Java Backend / Frontend
         │
         ▼
   ┌─────────────┐
   │   main.py   │  ◄── FastAPI REST API (Port 8000)
   │  /api/chat  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │  chatbot/       │
   │  agent.py       │  ◄── Gemini 1.5 Pro (LLM Beyin)
   │  tools.py       │  ◄── ML + Routing əlaqəsi
   └──────┬──────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌──────────┐  ┌──────────────┐
│ml_models/│  │  routing/    │
│predict.py│  │ dijkstra.py  │
│ (~90% R²)│  │ (Baku Graph) │
└──────────┘  └──────────────┘
```

---

## 📁 Fayl Quruluşu

```
WayGo-AI/
├── main.py                   # FastAPI API Qapısı
├── requirements.txt          # Bütün asılılıqlar
├── .env.example              # API açarı şablonu
│
├── chatbot/
│   ├── agent.py              # Gemini AI Agent (Əsas dialoq mərkəzi)
│   ├── tools.py              # LLM-in əlindəki real funksiyalar
│   └── knowledge_base.json   # YHQ Qaydaları Bazası (RAG)
│
├── ml_models/
│   ├── data_generator.py     # 10,000 sətirlik sintəz data yaradıcısı
│   ├── traffic_data.csv      # Bakı nəqliyyat öyrətmə dataseti
│   ├── train_model.py        # Əsas model öyrətmə skripti
│   ├── tune_model.py         # GridSearchCV ilə hiperparametr optimizasiyası
│   ├── predict.py            # Tıxac proqnozu API funksiyası
│   └── traffic_model.pkl     # Öyrədilmiş model (17 MB, binar)
│
├── routing/
│   └── dijkstra.py           # Bakı yol qrafı + ML çəkili naviqasiya
│
├── analytics/
│   ├── dashboard.py          # Tıxac trend qrafiklərini generasiya edir
│   ├── explain_model.py      # XAI — Süni İntellektin Qərar Xəritəsi
│   └── *.png                 # Investora hazır vizualizasiyalar
│
├── utils/
│   └── logger.py             # Mərkəzləşdirilmiş server loqlaşdırması
│
├── logs/                     # (Gitignore) Günlük .log faylları
│
└── scripts/
    ├── test_chat.py          # İnteraktiv lokal test mühiti
    └── list_models.py        # Gemini model siyahısı yoxlayıcısı
```

---

## 🚀 Sürətli Başlanğıc (Quickstart)

### 1. Asılılıqları yüklə
```bash
pip install -r requirements.txt
```

### 2. API açarını qur
```bash
# .env.example faylını kopyala
cp .env.example .env

# .env faylını aç və Gemini API açarını yaz:
# GEMINI_API_KEY=sizin_açarınız
```
> Gemini API açarını [Google AI Studio](https://aistudio.google.com)-dan əldə edə bilərsiniz.

### 3. ML Modelini Yarat (Birinci dəfə)
```bash
# Datanı yarat
py ml_models/data_generator.py

# Modeli öyrət və optimallaşdır
py ml_models/tune_model.py
```

### 4. Serveri Başlat
```bash
py -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. API-ni Test Et
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Ziya Bünyadovda tıxac varmı?", "session_id": "test_001"}'
```

---

## 📊 Analytics & XAI Dashboard

Aşağıdakı əmrlərlə investora hazır vizualizasiyaları generasiya edin:

```bash
# Tıxac trend qrafiklərini yarat (3 PNG)
py analytics/dashboard.py

# Süni İntellektin qərar xəritəsini yarat
py analytics/explain_model.py
```

---

## 🤖 ML Modeli Haqqında

| Parametr | Dəyər |
|---|---|
| Alqoritm | Random Forest Regressor |
| Optimallaşdırma | GridSearchCV (81 kombinasiya) |
| Ən Yaxşı Dərinlik | `max_depth = 10` |
| Ağac Sayı | `n_estimators = 300` |
| Dəqiqlik (R² Score) | **~90%** |
| Orta Xəta (MAE) | ~5% tıxac fərqi |

---

## 🔗 API Referansı

### `POST /api/chat`
Java Backend-dən gələn sorğuları qəbul edir.

**Request Body:**
```json
{
  "message": "string",
  "session_id": "string",
  "congestion_pct": 85,
  "avg_speed": 22.5,
  "active_vehicles": 1200,
  "weather_cond": "Qar",
  "temp": -2.5,
  "incident_count": 2,
  "anomaly_count": 1
}
```

**Response:**
```json
{
  "reply": "string"
}
```

### `POST /api/chat/stream`
Canlı (Server-Sent Events) cavab axını. Frontend üçün ideal.

---

## 🛠️ Texnologiya Steki

- **AI / LLM:** Google Gemini 1.5 Pro API
- **ML:** Scikit-learn (RandomForestRegressor, GridSearchCV)
- **API:** FastAPI + Uvicorn
- **Data:** Pandas, NumPy
- **Vizualizasiya:** Matplotlib, Seaborn
- **Loqlaşdırma:** Python `logging` modulu

---

## 👥 Komanda Əlaqəsi

Bu repozitoriya **WayGo** Startapının Süni İntellekt (AI & Data Science) Komandasına aiddir.

- 📌 **Scope:** Yalnız `WayGo-AI/` qovluğu (AI, ML, Routing, Analytics)
- 🔗 **Backend:** Java Spring Boot komandası ilə `POST /api/chat` üzərindən əlaqə
- 🎨 **Frontend:** React/Flutter komandası ilə `/stream` endpoint üzərindən əlaqə

---

*Son yenilənmə: 2026-08-08 | WayGo AI Komandası*
