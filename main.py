import os
from typing import List, Dict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class Message(BaseModel):
    role: str
    parts: List[str]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, List[str]]] = []

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    if not api_key:
        return {"response": "Lỗi: Chưa cấu hình GEMINI_API_KEY trên Render!"}
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-3.6-flash",
            system_instruction="Bạn là một AI Gia sư dạy lập trình từ con số 0. Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống."
        )
        # Khởi tạo phiên chat với lịch sử tin nhắn cũ
        chat_session = model.start_chat(history=req.history)
        response = chat_session.send_message(req.message)
        
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Lỗi từ Google: {str(e)}"}
