import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# Cấu hình API Key
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

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
        # Sử dụng mô hình Gemini 2.0 Flash mới nhất
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="Bạn là một AI Gia sư dạy lập trình từ con số 0. Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống."
        )
        response = model.generate_content(req.message)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Lỗi từ Google: {str(e)}"}
