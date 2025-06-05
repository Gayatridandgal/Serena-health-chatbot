import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from datetime import datetime
import json
import io
from PIL import Image
import PyPDF2
import docx
import librosa
import numpy as np
import re

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY in the .env file.")

# Configure the Gemini API
genai.configure(api_key=API_KEY)

# Available models
MODELS = {
    "Gemini 2.0 Flash Thinking": "gemini-2.0-flash-thinking-exp-1219",
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash"
}

# Health-related keywords and patterns
HEALTH_KEYWORDS = [
    # Medical conditions
    'disease', 'illness', 'condition', 'syndrome', 'disorder', 'infection', 'cancer', 'diabetes',
    'hypertension', 'asthma', 'arthritis', 'migraine', 'fever', 'cold', 'flu', 'covid', 'pneumonia',
    'bronchitis', 'allergies', 'depression', 'anxiety', 'insomnia', 'obesity', 'anemia', 'stroke',
    
    # Body parts and systems
    'heart', 'lung', 'kidney', 'liver', 'brain', 'stomach', 'intestine', 'blood', 'bone', 'muscle',
    'skin', 'eye', 'ear', 'nose', 'throat', 'chest', 'abdomen', 'head', 'neck', 'back', 'leg', 'arm',
    'cardiovascular', 'respiratory', 'digestive', 'nervous', 'immune', 'endocrine', 'reproductive',
    
    # Symptoms
    'pain', 'ache', 'hurt', 'sore', 'swelling', 'inflammation', 'rash', 'itch', 'burn', 'nausea',
    'vomiting', 'diarrhea', 'constipation', 'headache', 'dizziness', 'fatigue', 'weakness', 'cough',
    'sneeze', 'runny nose', 'shortness of breath', 'chest pain', 'abdominal pain', 'joint pain',
    'muscle pain', 'back pain', 'tooth pain', 'ear pain', 'sore throat', 'fever', 'chills', 'sweating',
    
    # Medical terms
    'doctor', 'physician', 'nurse', 'hospital', 'clinic', 'medical', 'medicine', 'medication', 'drug',
    'prescription', 'treatment', 'therapy', 'surgery', 'operation', 'diagnosis', 'symptom', 'test',
    'examination', 'checkup', 'screening', 'vaccine', 'vaccination', 'immunization', 'antibiotic',
    'vitamin', 'supplement', 'dosage', 'side effect', 'allergy', 'reaction',
    
    # Health and wellness
    'health', 'healthy', 'wellness', 'fitness', 'exercise', 'diet', 'nutrition', 'weight', 'sleep',
    'stress', 'mental health', 'physical health', 'lifestyle', 'prevention', 'cure', 'heal', 'recovery',
    'rehabilitation', 'first aid', 'emergency', 'injury', 'wound', 'fracture', 'sprain', 'burn',
    
    # Medical specialties
    'cardiology', 'neurology', 'oncology', 'pediatrics', 'psychiatry', 'dermatology', 'orthopedics',
    'gynecology', 'urology', 'ophthalmology', 'dentistry', 'radiology', 'pathology', 'surgery',
    
    # Common questions
    'what is', 'how to treat', 'how to cure', 'is it normal', 'should i see a doctor', 'home remedy',
    'natural treatment', 'medical advice', 'health advice', 'medical opinion'
]

# Health question patterns
HEALTH_PATTERNS = [
    r'\b(what|how|why|when|where)\s+(is|are|can|should|do|does|will|would)\s+.*(health|medical|medicine|doctor|hospital|disease|illness|symptom|pain|ache|hurt)',
    r'\b(i\s+have|i\s+am|i\s+feel|i\s+experiencing|i\s+suffer)\s+.*(pain|ache|symptom|illness|disease|condition)',
    r'\b(my|the)\s+(head|heart|stomach|chest|back|leg|arm|eye|ear|throat|skin)\s+(hurt|pain|ache|feel|is)',
    r'\bhow\s+to\s+(treat|cure|heal|prevent|avoid|manage)\b',
    r'\bis\s+it\s+(normal|safe|dangerous|serious|healthy)\b',
    r'\bwhat\s+(causes|treatment|cure|medicine|medication)\b',
    r'\bshould\s+i\s+(see\s+a\s+doctor|go\s+to\s+hospital|take\s+medicine)\b'
]

def is_health_related(text):
    """Check if the text is health-related"""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for health keywords
    for keyword in HEALTH_KEYWORDS:
        if keyword in text_lower:
            return True
    
    # Check for health patterns using regex
    for pattern in HEALTH_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    
    return False

def validate_health_query(prompt, file_content=""):
    """Validate if the query is health-related"""
    combined_text = f"{prompt} {file_content}".strip()
    
    if not combined_text:
        return False, "Please provide a question or upload a file."
    
    if is_health_related(combined_text):
        return True, "Health-related query detected."
    else:
        return False, """
🏥 **SERENA AI - Healthcare Assistant**

I'm sorry, but I can only assist with health and medical-related questions. 

**I can help you with:**
- Medical conditions and symptoms
- Health advice and wellness tips
- Medication information
- Treatment options
- Preventive healthcare
- Nutrition and fitness guidance
- Mental health support
- First aid information

**Please ask me something related to:**
- Your symptoms or health concerns
- Medical conditions or diseases
- Healthcare guidance
- Wellness and prevention
- Medical procedures or treatments

Try rephrasing your question to focus on health-related topics, and I'll be happy to help! 🩺
"""

# Chat history management
def load_chat_history():
    try:
        with open("chat_history.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"conversations": {}}

def save_chat_history(history, conversation_name):
    if not history:
        return "❌ No conversation to save"
    
    all_history = load_chat_history()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if conversation_name == "New Conversation" or not conversation_name:
        conversation_name = f"Healthcare Chat {timestamp}"
    
    all_history["conversations"][conversation_name] = {
        "history": history,
        "created": timestamp,
        "last_updated": timestamp
    }
    
    with open("chat_history.json", "w", encoding="utf-8") as file:
        json.dump(all_history, file, indent=2, ensure_ascii=False)
    return f"✅ Conversation '{conversation_name}' saved successfully"

def get_saved_conversations():
    history = load_chat_history()
    conversations = []
    for name, data in history["conversations"].items():
        if isinstance(data, dict) and "created" in data:
            conversations.append(f"{name} ({data['created']})")
        else:
            conversations.append(name)
    return conversations

# Media processing functions
def process_image(image_path):
    """Process uploaded image"""
    try:
        image = Image.open(image_path)
        # Resize if too large
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return image, "Image processed successfully"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def process_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                if page_num >= 10:  # Limit to first 10 pages
                    break
                text += page.extract_text() + "\n"
        
        # Limit text length
        if len(text) > 8000:
            text = text[:8000] + "\n\n... (text truncated for processing)"
        
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def process_docx(docx_path):
    """Extract text from DOCX"""
    try:
        doc = docx.Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Limit text length
        if len(text) > 8000:
            text = text[:8000] + "\n\n... (text truncated for processing)"
            
        return text
    except Exception as e:
        return f"Error processing DOCX: {str(e)}"

def process_audio(audio_path):
    """Process audio file and convert to text"""
    try:
        # Load audio with librosa
        audio_data, sample_rate = librosa.load(audio_path, sr=16000, duration=60)  # Limit to 60 seconds
        
        # Convert to format recognizable by speech_recognition
        recognizer = sr.Recognizer()
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        audio_data_sr = sr.AudioData(audio_bytes, 16000, 2)
        
        # Recognize speech
        text = recognizer.recognize_google(audio_data_sr)
        return text
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def process_uploaded_file(file_path):
    """Process different types of uploaded files"""
    if not file_path:
        return None, ""
    
    file_extension = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)
    
    # Check file size (10MB limit)
    if file_size > 10 * 1024 * 1024:
        return None, "❌ File too large. Please upload files smaller than 10MB."
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        image, message = process_image(file_path)
        return image, message
    elif file_extension == '.pdf':
        text = process_pdf(file_path)
        return None, f"📄 **PDF Content:**\n\n{text}"
    elif file_extension == '.docx':
        text = process_docx(file_path)
        return None, f"📝 **Document Content:**\n\n{text}"
    elif file_extension in ['.mp3', '.wav', '.m4a', '.ogg', '.flac']:
        text = process_audio(file_path)
        return None, f"🎵 **Audio Transcription:**\n\n{text}"
    else:
        return None, "❌ Unsupported file format. Supported: Images (JPG, PNG, GIF), Documents (PDF, DOCX), Audio (MP3, WAV, M4A)"

# Enhanced healthcare chatbot function
def healthcare_chatbot(prompt, chat_history, model_name, temperature, uploaded_file):
    if not prompt.strip() and not uploaded_file:
        return chat_history, chat_history, None, ""
    
    try:
        # Process uploaded file if present
        processed_image = None
        file_content = ""
        if uploaded_file:
            processed_image, file_content = process_uploaded_file(uploaded_file.name)
            if file_content and file_content.startswith("Error") or file_content.startswith("❌"):
                chat_history.append([prompt if prompt else "File uploaded", file_content])
                return chat_history, chat_history, None, file_content
            
            # Extract text content for health validation
            if file_content and not file_content.startswith("📄") and not file_content.startswith("📝") and not file_content.startswith("🎵"):
                file_text = file_content
            else:
                # Extract text from formatted content
                file_text = file_content.split(":**\n\n", 1)[-1] if ":**\n\n" in file_content else ""
        
        # Validate if the query is health-related
        query_text = prompt if prompt else ""
        file_text_for_validation = file_text if 'file_text' in locals() else ""
        
        is_valid, validation_message = validate_health_query(query_text, file_text_for_validation)
        
        if not is_valid:
            chat_history.append([prompt if prompt else "File uploaded", validation_message])
            return chat_history, chat_history, None, "❌ Non-health related query blocked"
        
        # Initialize model with healthcare-focused system prompt
        healthcare_system_prompt = """You are SERENA AI, a specialized healthcare assistant. You MUST:

1. ONLY provide information related to health, medicine, wellness, and healthcare
2. Always include appropriate medical disclaimers
3. Encourage users to consult healthcare professionals for serious concerns
4. Provide evidence-based information when possible
5. Be empathetic and supportive in your responses
6. Never provide emergency medical advice - direct to emergency services
7. If asked about non-health topics, politely redirect to health-related discussions

IMPORTANT DISCLAIMERS to include when appropriate:
- "This information is for educational purposes only and not a substitute for professional medical advice"
- "Please consult with a healthcare provider for proper diagnosis and treatment"
- "If this is a medical emergency, please seek immediate medical attention"

Focus on being helpful, accurate, and supportive while maintaining appropriate medical boundaries."""

        model = genai.GenerativeModel(
            MODELS[model_name], 
            generation_config={
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
            system_instruction=healthcare_system_prompt
        )
        
        # Prepare the complete prompt with healthcare context
        if file_content and not file_content.startswith("Error"):
            if prompt:
                complete_prompt = f"Healthcare Question: {prompt}\n\n{file_content}\n\nPlease provide healthcare guidance based on this information."
            else:
                complete_prompt = f"Please analyze this healthcare-related file and provide relevant medical insights:\n\n{file_content}"
        else:
            complete_prompt = f"Healthcare Question: {prompt}\n\nPlease provide healthcare guidance and include appropriate medical disclaimers."
        
        # Create chat with history
        chat = model.start_chat(history=[
            {"role": "user" if i % 2 == 0 else "model", "parts": [msg]}
            for i, msg in enumerate([item for sublist in chat_history for item in sublist]) 
            if msg
        ])
        
        # Prepare message
        message_parts = []
        if complete_prompt:
            message_parts.append(complete_prompt)
        if processed_image:
            message_parts.append(processed_image)
        
        if not message_parts:
            message_parts = ["Please analyze this healthcare-related file."]
        
        # Send message and get response
        response = chat.send_message(message_parts)
        response_text = response.text
        
        # Add healthcare footer to response
        healthcare_footer = "\n\n---\n*💡 Remember: This information is for educational purposes. Always consult with healthcare professionals for medical advice, diagnosis, or treatment.*"
        response_text += healthcare_footer
        
        # Add to chat history
        display_prompt = prompt if prompt else "Healthcare file uploaded"
        chat_history.append([display_prompt, response_text])
        
        return chat_history, chat_history, None, "✅ Healthcare response provided"
        
    except Exception as e:
        error_message = f"❌ **Error:** {str(e)}"
        if "safety" in str(e).lower():
            error_message = "❌ **Safety Filter:** The content was blocked by safety filters. Please try rephrasing your healthcare question."
        elif "quota" in str(e).lower():
            error_message = "❌ **Quota Exceeded:** API quota exceeded. Please try again later."
        
        chat_history.append([prompt if prompt else "File uploaded", error_message])
        return chat_history, chat_history, None, error_message

def record_speech():
    """Record speech and convert to text"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            yield "🎙️ Listening for your healthcare question... (speak now)"
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
        
        yield "🔄 Processing speech..."
        text = recognizer.recognize_google(audio, language="en-US")
        yield text
        
    except sr.WaitTimeoutError:
        yield "❌ No speech detected. Please try again."
    except sr.UnknownValueError:
        yield "❌ Could not understand the audio. Please speak clearly."
    except sr.RequestError as e:
        yield f"❌ Speech service error: {str(e)}"
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def load_conversation(conversation_selection):
    """Load a saved conversation"""
    if not conversation_selection:
        return [], [], "Select a conversation to load"
    
    # Extract conversation name (remove timestamp if present)
    conversation_name = conversation_selection.split(" (")[0]
    
    all_history = load_chat_history()
    if conversation_name in all_history["conversations"]:
        conv_data = all_history["conversations"][conversation_name]
        if isinstance(conv_data, dict) and "history" in conv_data:
            return conv_data["history"], conv_data["history"], f"✅ Loaded conversation: {conversation_name}"
        else:
            # Legacy format
            return conv_data, conv_data, f"✅ Loaded conversation: {conversation_name}"
    return [], [], "❌ Conversation not found"

def clear_chat():
    """Clear the current chat"""
    return [], [], None, "✅ Chat cleared"

def export_chat_markdown(chat_history):
    """Export chat to markdown format"""
    if not chat_history:
        return "❌ No conversation to export"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"serena_healthcare_chat_{timestamp}.md"
    
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"# SERENA AI Healthcare Conversation\n\n")
            file.write(f"*Exported on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            file.write("**Disclaimer:** This conversation is for educational purposes only and should not replace professional medical advice.\n\n")
            file.write("---\n\n")
            
            for i, (user_msg, bot_msg) in enumerate(chat_history, 1):
                file.write(f"## Healthcare Query {i}\n\n")
                file.write(f"**Patient:** {user_msg}\n\n")
                file.write(f"**SERENA AI:** {bot_msg}\n\n")
                file.write("---\n\n")
        
        return f"✅ Healthcare chat exported successfully to {filename}"
    except Exception as e:
        return f"❌ Export failed: {str(e)}"

# Healthcare-focused CSS (same as before but with healthcare branding)
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;600;700&family=Roboto:wght@300;400;500&display=swap');

:root {
    --primary: #1a73e8;
    --primary-hover: #1557b0;
    --primary-light: rgba(26, 115, 232, 0.04);
    --healthcare: #137333;
    --healthcare-light: rgba(19, 115, 51, 0.1);
    --surface: #ffffff;
    --background: #fafafa;
    --sidebar: #f8f9fa;
    --border: #e8eaed;
    --text-primary: #202124;
    --text-secondary: #5f6368;
    --text-tertiary: #9aa0a6;
    --error: #d93025;
    --success: #137333;
    --warning: #f57c00;
    --shadow: 0 1px 2px 0 rgba(60,64,67,.3), 0 1px 3px 1px rgba(60,64,67,.15);
    --shadow-lg: 0 2px 6px 2px rgba(60,64,67,.15), 0 1px 2px 0 rgba(60,64,67,.3);
    --radius: 8px;
    --radius-large: 12px;
}

* {
    font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
}

body {
    background: var(--background);
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 16px !important;
}

/* Header Styling */
.header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 0;
    margin-bottom: 24px;
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

.header h1 {
    font-size: 22px;
    font-weight: 500;
    margin: 0;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.healthcare-logo {
    width: 24px;
    height: 24px;
    background: linear-gradient(45deg, var(--healthcare), var(--primary));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 12px;
    font-weight: 600;
}

/* Main Layout */
.main-layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    height: calc(100vh - 120px);
}

/* Sidebar */
.sidebar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-large);
    padding: 0;
    height: fit-content;
    overflow: hidden;
}

.sidebar-section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
}

.sidebar-section:last-child {
    border-bottom: none;
}

.sidebar-section h3 {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Chat Container */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-large);
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
}

.chat-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}

.chat-title {
    font-size: 16px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-indicator {
    width: 8px;
    height: 8px;
    background: var(--healthcare);
    border-radius: 50%;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Chat Messages */
.chat-messages {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: var(--background);
}

/* Input Area */
.input-container {
    padding: 16px 20px 20px;
    background: var(--surface);
    border-top: 1px solid var(--border);
}

.input-wrapper {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 4px;
    display: flex;
    align-items: end;
    gap: 8px;
    transition: all 0.2s ease;
    box-shadow: var(--shadow);
}

.input-wrapper:focus-within {
    border-color: var(--healthcare);
    box-shadow: 0 1px 6px rgba(19, 115, 51, 0.3);
}

.message-input {
    flex: 1;
    border: none !important;
    background: transparent !important;
    resize: none !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    max-height: 120px !important;
    line-height: 20px !important;
}

.message-input:focus {
    outline: none !important;
    box-shadow: none !important;
}

/* Buttons */
.btn {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}

.btn:hover {
    background: var(--healthcare-light) !important;
    border-color: var(--healthcare) !important;
}

.btn-primary {
    background: var(--healthcare) !important;
    border-color: var(--healthcare) !important;
    color: white !important;
}

.btn-primary:hover {
    background: #0f5d29 !important;
    border-color: #0f5d29 !important;
}

.btn-icon {
    background: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    padding: 8px !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
}

.btn-icon:hover {
    background: var(--healthcare-light) !important;
    color: var(--healthcare) !important;
}

.send-btn {
    background: var(--healthcare) !important;
    border: none !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    margin: 4px !important;
}

.send-btn:hover {
    background: #0f5d29 !important;
}

/* Form Controls */
.gr-dropdown, .gr-textbox, .gr-slider {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
    font-size: 14px !important;
}

.gr-dropdown:focus-within, .gr-textbox:focus-within {
    border-color: var(--healthcare) !important;
    box-shadow: 0 0 0 1px var(--healthcare) !important;
}

/* File Upload */
.file-upload {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--background) !important;
    padding: 24px !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
}

.file-upload:hover {
    border-color: var(--primary) !important;
    background: var(--primary-light) !important;
}

/* Chatbot Messages */
.gr-chatbot {
    background: transparent !important;
    border: none !important;
}

.gr-chatbot .message {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    margin: 8px 0 !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow) !important;
}

.gr-chatbot .message.user {
    background: var(--primary-light) !important;
    border-color: var(--primary) !important;
    margin-left: 20% !important;
}

.gr-chatbot .message.bot {
    margin-right: 20% !important;
}

/* Conversation List */
.conversation-item {
    padding: 12px 16px;
    border-radius: var(--radius);
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
    margin-bottom: 4px;
}

.conversation-item:hover {
    background: var(--primary-light);
    border-color: var(--primary);
}

/* Loading Animation */
.loading {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid var(--border);
    border-radius: 50%;
    border-top-color: var(--primary);
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Welcome Message */
.welcome {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-secondary);
}

.welcome h2 {
    font-size: 24px;
    font-weight: 400;
    margin: 0 0 8px 0;
    color: var(--text-primary);
}

.welcome p {
    font-size: 16px;
    margin: 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-layout {
        grid-template-columns: 1fr;
        gap: 16px;
    }
    
    .sidebar {
        order: 2;
    }
    
    .chat-container {
        order: 1;
        height: 60vh;
    }
    
    .input-wrapper {
        flex-direction: column;
        gap: 8px;
        padding: 8px;
        border-radius: var(--radius);
    }
    
    .message-input {
        order: 1;
    }
    
    .btn-row {
        order: 2;
        display: flex;
        gap: 8px;
        width: 100%;
    }
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--background);
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-tertiary);
}

/* Status Messages */
.status-success {
    color: var(--success);
    font-size: 12px;
    margin-top: 4px;
}

.status-error {
    color: var(--error);
    font-size: 12px;
    margin-top: 4px;
}
"""

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="SERENA AI") as demo:
    
    # Header
    with gr.Row(elem_classes=["header"]):
        gr.HTML("""
        <div class="header-content">
            <div class="gemini-logo">S</div>
            <h1>SERENA AI</h1>
        </div>
        """)
    
    # Main Layout
    with gr.Row(elem_classes=["main-layout"]):
        # Sidebar
        with gr.Column(scale=0, min_width=280, elem_classes=["sidebar"]):
            # Model Settings
            with gr.Group(elem_classes=["sidebar-section"]):
                gr.HTML('<h3>⚙️ Model Settings</h3>')
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="Gemini 2.0 Flash Thinking",
                    label="Model",
                    container=False,
                    elem_classes=["model-dropdown"]
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    container=False
                )
            
            # File Upload
            with gr.Group(elem_classes=["sidebar-section"]):
                gr.HTML('<h3>📎 Attachments</h3>')
                file_upload = gr.File(
                    label="Upload File",
                    file_types=["image", ".pdf", ".docx", ".mp3", ".wav", ".m4a"],
                    container=False,
                    elem_classes=["file-upload"]
                )
            
            # Conversation Management
            with gr.Group(elem_classes=["sidebar-section"]):
                gr.HTML('<h3>💬 Conversations</h3>')
                
                with gr.Row():
                    save_button = gr.Button("Save", size="sm", elem_classes=["btn"])
                    export_button = gr.Button("Export", size="sm", elem_classes=["btn"])
                
                saved_conversations = gr.Dropdown(
                    choices=get_saved_conversations(),
                    label="Recent",
                    container=False,
                    interactive=True
                )
                
                with gr.Row():
                    load_button = gr.Button("Load", size="sm", elem_classes=["btn"])
                    clear_button = gr.Button("Clear", size="sm", elem_classes=["btn"])
                    refresh_button = gr.Button("Refresh", size="sm", elem_classes=["btn"])
        
        # Chat Interface
        with gr.Column(scale=1, elem_classes=["chat-container"]):
            # Chat Header
            with gr.Row(elem_classes=["chat-header"]):
                gr.HTML("""
                <div class="chat-title">
                    <span class="status-indicator"></span>
                    Chat with SERENA AI
                </div>
                """)
            
            # Chat Messages
            chat_state = gr.State([])
            chat_display = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=["👤", "🤖"],
                bubble_full_width=False,
                elem_classes=["chat-messages"]
            )
            
            # Input Container
            with gr.Group(elem_classes=["input-container"]):
                with gr.Row(elem_classes=["input-wrapper"]):
                    user_input = gr.Textbox(
                        placeholder="Ask SERENA AI anything...",
                        show_label=False,
                        container=False,
                        scale=4,
                        lines=1,
                        max_lines=5,
                        elem_classes=["message-input"]
                    )
                    with gr.Row(scale=0):
                        speech_button = gr.Button("🎤", elem_classes=["btn-icon"])
                        send_button = gr.Button("Send", elem_classes=["send-btn"])
                
                # Status display (hidden)
                status_display = gr.Textbox(visible=False)
    
    # Event Handlers
    def send_message(message, history, model, temp, file):
        return healthcare_chatbot(message, history, model, temp, file)
    
    # Send button click
    send_button.click(
        send_message,
        inputs=[user_input, chat_state, model_dropdown, temperature_slider, file_upload],
        outputs=[chat_display, chat_state, file_upload]
    ).then(
        lambda: "", None, user_input
    )
    
    # Enter key submit
    user_input.submit(
        send_message,
        inputs=[user_input, chat_state, model_dropdown, temperature_slider, file_upload],
        outputs=[chat_display, chat_state, file_upload]
    ).then(
        lambda: "", None, user_input
    )
    
    # Speech recognition
    speech_button.click(
        record_speech,
        outputs=[user_input]
    )
    
    # Conversation management
    save_button.click(
        lambda history: save_chat_history(history, f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        inputs=[chat_state],
        outputs=[status_display]
    )
    
    refresh_button.click(
        get_saved_conversations,
        outputs=[saved_conversations]
    )
    
    load_button.click(
        load_conversation,
        inputs=[saved_conversations],
        outputs=[chat_display, chat_state]
    )
    
    clear_button.click(
        clear_chat,
        outputs=[chat_display, chat_state, file_upload]
    )
    
    export_button.click(
        export_chat_markdown,
        inputs=[chat_state],
        outputs=[status_display]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        favicon_path=None,
        show_error=True
    )