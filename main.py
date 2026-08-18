from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

# Cấu hình CORS để gọi API không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lấy API Key từ biến môi trường Render
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Định nghĩa cấu trúc dữ liệu gửi lên
class Message(BaseModel):
    role: str  # "user" hoặc "model"
    text: str

class ChatRequest(BaseModel):
    history: list[Message]
    message: str

# Prompt hệ thống định hình phong cách Gia sư Socratic
SYSTEM_PROMPT = """
Bạn là một AI Gia sư dạy học theo phương pháp Socratic.
Quy tắc vàng của bạn:
1. KHÔNG BAO GIỜ cho đáp án trực tiếp ngay lập tức.
2. Luôn đặt câu hỏi gợi mở, ngắn gọn, từng bước một để học sinh tự suy luận.
3. Khen ngợi nhẹ nhàng khi học sinh trả lời đúng.
4. Nếu học sinh trả lời sai, hãy chỉ ra điểm mâu thuẫn trong câu trả lời của họ bằng một câu hỏi khác.
5. Giữ giọng văn thân thiện, kiên nhẫn, gần gũi như một người anh/chị hướng dẫn.
"""

@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Không tìm thấy file index.html!</h1>"

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Chưa cài đặt GEMINI_API_KEY")
    
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        
        # Chuyển đổi lịch sử chat sang định dạng của Gemini
        formatted_history = []
        for msg in req.history:
            formatted_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.text]
            })
            
        chat_session = model.start_chat(history=formatted_history)
        response = chat_session.send_message(req.message)
        
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
