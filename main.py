import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. Định nghĩa cấu trúc JSON đầu ra
class StepHint(BaseModel):
    step_number: int = Field(description="Số thứ tự của bước gợi ý")
    explanation: str = Field(description="Lời giải thích hoặc câu hỏi gợi mở")

class TutorResponse(BaseModel):
    praise_or_encouragement: str = Field(description="Lời khen ngợi, động viên")
    concept_explanation: str = Field(description="Giải thích ngắn gọn khái niệm")
    hints: List[StepHint] = Field(description="Danh sách các bước gợi ý")
    guided_question: str = Field(description="Câu hỏi dẫn dắt ở cuối")

class StudentQuery(BaseModel):
    session_id: str = Field(default="hoc_sinh_01", description="ID phiên học")
    subject: str = Field(default="Toán học", description="Môn học")
    grade_level: str = Field(default="Lớp 9", description="Lớp")
    question: str = Field(..., description="Câu hỏi của học sinh")

app = FastAPI(title="AI Tutor API")

# 2. Khởi tạo Gemini Client (Điền API Key thật của bạn vào đây)
client = genai.Client(api_key="AQ.Ab8RN6J6MVVLxhLVat_9BxU8KW9sdB3W87-Km_8DhlkgpJnA-Q")

sessions_db: Dict[str, List[str]] = {}

# Route phục vụ giao diện trang Web Chat
@app.get("/", response_class=HTMLResponse)
async def get_index():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Không tìm thấy file index.html!</h1>"

@app.post("/api/tutor/chat", response_model=TutorResponse)
async def ask_tutor(query: StudentQuery):
    session_id = query.session_id
    if session_id not in sessions_db:
        sessions_db[session_id] = []

    sessions_db[session_id].append(f"Học sinh: {query.question}")

    system_instruction = (
        f"Bạn là AI Gia sư môn {query.subject} trình độ {query.grade_level}. "
        "Hãy dùng phương pháp Socratic để gợi mở, không cho ngay đáp án trực tiếp. "
        "Dựa vào lịch sử hội thoại để đưa ra phản hồi phù hợp."
    )

    full_prompt = "\n".join(sessions_db[session_id])

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=TutorResponse,
            ),
        )
        
        if response.parsed:
            sessions_db[session_id].append(f"Gia sư: {response.parsed.guided_question}")
            return response.parsed
        else:
            raise HTTPException(status_code=500, detail="Mô hình không trả về dữ liệu đúng định dạng.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)