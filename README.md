# 🩺 SERENA AI – Healthcare Chatbot with Gemini + Voice & File Support

SERENA AI is an intelligent healthcare assistant built using Google's Gemini API, Gradio, and speech recognition. It helps users with health-related questions, supports file uploads (PDF, DOCX, Images, Audio), and offers speech-to-text functionality for a more accessible experience.

---

## 🚀 Features

- 💬 Natural language healthcare chatbot powered by **Gemini 2.0 Flash Thinking** and **Gemini Flash**
- 🎙️ **Speech-to-text** using Google Speech Recognition
- 📎 Supports medical file uploads: PDFs, DOCX, images, and audio (MP3, WAV, M4A)
- ⚕️ Validates that conversations are **strictly health-related**
- 📚 Auto-saves and loads **chat history**
- 🧠 Understands symptoms, conditions, treatments, medications, and wellness topics
- 💡 Empathetic and medically responsible responses with disclaimers

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Gayatridandgal/Serena-health-chatbot.git
cd serena-health-chatbot

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

#### Create a .env file with your Gemini API key:
GEMINI_API_KEY=your_google_gemini_api_key_here

#### python app.py

