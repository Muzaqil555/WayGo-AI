from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ətraf mühit dəyişənlərini yükləyirik
load_dotenv()

# Daxili modullar (AI komandası)
from chatbot.agent import process_chat, stream_chat
from fastapi.responses import StreamingResponse

app = FastAPI(title="WayGo AI Engine", description="Baku Urban Mobility AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"
    user_profile: dict = None
    congestion_pct: int = 0
    avg_speed: float = 0.0
    active_vehicles: int = 0
    weather_cond: str = "Bilinmir"
    temp: float = 0.0
    incident_count: int = 0
    anomaly_count: int = 0

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Java Backend-dən gələn sorğuları qarşılayır və Chatbot agentinə ötürür.
    """
    try:
        stats = {
            "congestion_pct": request.congestion_pct,
            "avg_speed": request.avg_speed,
            "active_vehicles": request.active_vehicles,
            "weather_cond": request.weather_cond,
            "temp": request.temp,
            "incident_count": request.incident_count,
            "anomaly_count": request.anomaly_count
        }
        
        reply_text = process_chat(
            message=request.message, 
            stats=stats, 
            session_id=request.session_id,
            user_profile=request.user_profile
        )
        return ChatResponse(reply=reply_text)
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="AI xətası baş verdi")

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Backend və Frontend üçün canlı (hissə-hissə) chat cavabı qaytarır.
    """
    stats = {
        "congestion_pct": request.congestion_pct,
        "avg_speed": request.avg_speed,
        "active_vehicles": request.active_vehicles,
        "weather_cond": request.weather_cond,
        "temp": request.temp,
        "incident_count": request.incident_count,
        "anomaly_count": request.anomaly_count
    }
    
    # Server-Sent Events (SSE) və ya təmiz text axını
    return StreamingResponse(
        stream_chat(
            message=request.message, 
            stats=stats, 
            session_id=request.session_id,
            user_profile=request.user_profile
        ), 
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    # Backend ilə əlaqə üçün Port 8000-də dinləyirik
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
