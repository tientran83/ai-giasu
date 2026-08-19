import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# Lấy API Key từ biến môi trường của Render
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
def chat(req: ChatRequest):
    if not client:
        return {"response": "Lỗi: Chưa cấu hình GEMINI_API_KEY trên Render!"}
    
    try:
        sys_instruct = (
            "Bạn là một AI Gia sư dạy lập trình từ con số 0. "
            "Hãy giải thích ngắn gọn, dễ hiểu, dùng ví dụ đời sống."
        )
        
        # Cập nhật tên model chính xác sang gemini-3.6-flash
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=req.message,
            config={"system_instruction": sys_instruct}
        )
        
        return {"response": response.text}
    except Exception as e:
        return {"response": f"Lỗi xử lý AI: {str(e)}"}
