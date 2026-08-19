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
    
    # Danh sách các URL endpoint khả thi để thử lần lượt
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    ]
    
    payload = {
        "contents": [
            {
                "parts": [{"text": f"Bạn là một AI Gia sư dạy lập trình từ con số 0. Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống.\n\nNgười học hỏi: {req.message}"}]
            }
        ]
    }
    
    last_error = ""
    for url in endpoints:
        try:
            res = requests.post(url, json=payload, timeout=30)
            data = res.json()
            
            if res.status_code == 200:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"response": text}
            else:
                last_error = data.get("error", {}).get("message", res.text)
        except Exception as e:
            last_error = str(e)
            
    return {"response": f"Lỗi xử lý AI: {last_error}"}
