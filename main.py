import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
        sys_instruct = (
            "Bạn là một AI Gia sư dạy lập trình từ con số 0. "
            "Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống."
        )
        
        # Sử dụng thư viện chuẩn google-generativeai
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=sys_instruct
        )
        
        response = model.generate_content(req.message)
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Lỗi xử lý AI: {str(e)}"}
