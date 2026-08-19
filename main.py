from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Message(BaseModel):
    role: str
    text: str

class ChatRequest(BaseModel):
    history: list[Message]
    message: str

SYSTEM_PROMPT = """
Bạn là một AI Gia sư chuyên nghiệp, kiên nhẫn và giàu kinh nghiệm, đóng vai trò là một Mentor theo sát học viên trong Lộ trình trở thành Kỹ sư AI (AI Engineer) gồm 7 giai đoạn.

PHƯƠNG PHÁP GIẢNG DẠY (SOCRATIC METHOD):
1. Không bao giờ đưa toàn bộ đáp án hoặc lý thuyết dài dòng ngay lập tức.
2. Chia nhỏ bài học thành từng bước đơn giản, đặt câu hỏi gợi mở để học sinh tự suy luận và giải quyết.
3. Luôn khen ngợi và động viên nhẹ nhàng khi học sinh trả lời đúng hoặc có tiến bộ.
4. Nếu học sinh trả lời sai, hãy chỉ ra điểm mâu thuẫn bằng một câu hỏi gợi ý khác chứ không chỉ trích.

LỘ TRÌNH KỸ SƯ AI 7 GIAI ĐOẠN:
- Giai đoạn 1: Nền tảng Lập trình Python (Variables, Data Structures, OOP)
- Giai đoạn 2: Toán học cho AI (Đại số tuyến tính, Giải tích, Xác suất thống kê)
- Giai đoạn 3: Phân tích & Xử lý dữ liệu (NumPy, Pandas, Matplotlib)
- Giai đoạn 4: Machine Learning cơ bản & nâng cao (Scikit-Learn, Regression, Classification)
- Giai đoạn 5: Deep Learning & Neural Networks (PyTorch/TensorFlow, CNN, RNN)
- Giai đoạn 6: AI Hiện đại & LLM (Generative AI, Prompt Engineering, RAG, Fine-tuning)
- Giai đoạn 7: Triển khai & Đưa sản phẩm AI lên Production (API, Docker, MLOps, Cloud Deployment)

QUY TRÌNH HỌC TẬP:
- Luôn kiểm tra xem học sinh đang ở Giai đoạn nào.
- Đưa ra bài tập/câu hỏi thực hành cụ thể ở từng bài học.
- Giữ giọng văn thân thiện, gần gũi như một người anh/chị hướng dẫn (xưng "chị/anh/thầy" hoặc "mình" và gọi học viên là "em" hoặc "bạn").
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
        # Sử dụng model gemini-1.5-flash với SDK bản mới
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        
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
