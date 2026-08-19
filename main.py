import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

api_key = os.environ.get("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    if not api_key:
        return {"response": "Lỗi: Chưa cấu hình GEMINI_API_KEY trên Render!"}
    
    # Endpoint chuẩn hỗ trợ trực tiếp các phiên bản Gemini
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "system_instruction": {
            "parts": [{"text": "Bạn là một AI Gia sư dạy lập trình từ con số 0. Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống."}]
        },
        "contents": [
            {
                "parts": [{"text": req.message}]
            }
        ]
    }
    
    try:
        res = requests.post(url, json=payload, timeout=30)
        data = res.json()
        
        if res.status_code == 200:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": text}
        else:
            error_msg = data.get("error", {}).get("message", res.text)
            return {"response": f"Lỗi xử lý AI ({res.status_code}): {error_msg}"}
            
    except Exception as e:
        return {"response": f"Lỗi kết nối: {str(e)}"}
