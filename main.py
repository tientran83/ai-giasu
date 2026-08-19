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
    
    try:
        # Bước 1: Lấy danh sách các model khả dụng cho API Key này
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        list_res = requests.get(list_url, timeout=10)
        
        valid_model_name = None
        if list_res.status_code == 200:
            models = list_res.json().get("models", [])
            for m in models:
                # Tìm model có hỗ trợ generateContent
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    valid_model_name = m.get("name") # Ví dụ: models/gemini-2.0-flash
                    break
        
        if not valid_model_name:
            # Fallback nếu không lấy được danh sách
            valid_model_name = "models/gemini-2.0-flash"

        # Bước 2: Gửi request chat đến model tìm được
        chat_url = f"https://generativelanguage.googleapis.com/v1beta/{valid_model_name}:generateContent?key={api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": f"Bạn là một AI Gia sư dạy lập trình từ con số 0. Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống.\n\nNgười học hỏi: {req.message}"}]
                }
            ]
        }
        
        res = requests.post(chat_url, json=payload, timeout=30)
        data = res.json()
        
        if res.status_code == 200:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"response": text}
        else:
            error_msg = data.get("error", {}).get("message", res.text)
            return {"response": f"Lỗi từ Google ({res.status_code}): {error_msg}"}
            
    except Exception as e:
        return {"response": f"Lỗi hệ thống: {str(e)}"}
