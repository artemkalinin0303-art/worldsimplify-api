import os
import json
import time
import hashlib
import sqlite3
import mimetypes 
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Библиотеки для чтения файлов
import pypdf
import docx

# Библиотека Google Gemini
from google.genai import Client

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
API_KEY = os.getenv("GOOGLE_API_KEY")
CLIENT = Client(api_key=API_KEY) if API_KEY else None

# Используем модели: 2.0-flash (быстрая) или 1.5-pro (умная)
MODEL_CANDIDATES = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
UPLOAD_DIR = "uploads"
DB_PATH = "worldsimplify.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- БАЗА ДАННЫХ ---
def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs(
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        plain_text TEXT,
        created_at INTEGER
    )""")
    con.commit()
    con.close()

db_init()

# --- УТИЛИТЫ ---
def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text_from_file(filepath: str, filename: str, content_type: str = None) -> str:
    """
    Извлекает текст из PDF, DOCX или КАРТИНОК (через Gemini Vision).
    """
    # 1. Сначала верим явному типу от клиента
    mime_type = content_type
    # 2. Если клиент не прислал тип, пытаемся угадать по файлу
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(filepath)
    
    # 3. Страховка: если расширение jpg/png, но тип не определился
    if not mime_type and (filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png')):
        mime_type = 'image/jpeg'

    print(f"📂 Processing File: {filename} | Type: {mime_type}") # Лог для проверки
    text = ""
    
    try:
        # 1. КАРТИНКА (OCR через Gemini)
        if mime_type and mime_type.startswith('image'):
            if CLIENT:
                print(f"📷 Sending image to AI OCR...")
                with open(filepath, "rb") as f:
                    image_data = f.read()
                try:
                    resp = CLIENT.models.generate_content(
                        model="gemini-2.0-flash", 
                        contents=[
                            "Transcribe the text from this contract image exactly as is. Do not summarize. If text is blurry, try your best.", 
                            {"mime_type": mime_type, "data": image_data}
                        ]
                    )
                    text = resp.text if resp.text else ""
                    print(f"✅ OCR Success. Chars extracted: {len(text)}")
                except Exception as img_err:
                    print(f"❌ OCR Error: {img_err}")
                    text = ""
            else:
                text = ""

        # 2. PDF
        elif filename.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        
        # 3. DOCX
        elif filename.lower().endswith(".docx"):
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        # 4. Текст
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
        
    return text.strip()

# --- 🔥 УНИВЕРСАЛЬНЫЙ ПРОМПТ (IRELAND/UK LOCALIZED) ---
READABLE_PROMPT_TEMPLATE = """
Act as a Senior Legal Risk Auditor.
Your goal is to protect the Client (the person signing or receiving the contract) from financial loss, legal traps, and unfair terms.

CONTEXT & JURISDICTION:
1. IF ENGLISH (en):
   - **PRIMARY JURISDICTION:** Irish Law (Republic of Ireland) & UK Common Law.
   - **LOOK FOR KEYWORDS:** "RTB", "Residential Tenancies Board", "Dublin", "PPSN", "Revenue", "WRC", "Cork", "Galway".
   - **APPLY:** GDPR (Data Protection), Consumer Rights Act, Employment Law (WRC standards if employment), Residential Tenancies Act (RTB rules if rental).
   - If the text explicitly mentions "California", "NY", "Delaware", switch to US Law.
2. IF RUSSIAN/UKRAINIAN -> Local laws (RF/Ukraine).
3. OTHERS -> Local laws based on language/location.

INSTRUCTIONS:
Step 1: IDENTIFY THE CONTRACT TYPE immediately (e.g., Employment Offer, NDA, Used Car Sale, Rental Agreement, Service Contract, Loan).

Step 2: Based on the type, look for SPECIFIC "Silent Killers" for that category:
   - **IF RENTAL:** Look for Rent Pressure Zone caps, RTB registration (Crucial for Ireland!), unfair deposit retention.
   - **IF EMPLOYMENT:** Look for "Unpaid Overtime", "Non-Compete", unfair probation periods, WRC compliance.
   - **IF FREELANCE/SERVICE:** Look for "Unlimited Revisions", "Intellectual Property Transfer" (who owns the code?), payment terms.
   - **IF CAR/SALES:** Look for "Sold as Seen" (hiding defects), warranty exclusions.
   - **IF NDA:** Look for "Perpetual duration" (forever), excessive penalties.

Step 3: ANALYZE and score the risk.

RETURN JSON ONLY. NO MARKDOWN. NO ```json TAGS.
Structure:
{{
  "risk_score": integer (0-100),
  "contract_type": "Detected type (e.g., Employment Contract - Ireland)",
  "summary": "Direct verdict in {language}. Start by confirming what document this is.",
  "risks": [
    {{
      "text": "Risk Title: Description in {language}",
      "severity": "High", 
      "original_clause": "Quote or '[MISSING CLAUSE]'"
    }}
  ]
}}
"""

REWRITE_PROMPT_TEMPLATE = """
Rewrite this contract clause to be safe and fair for the Client.
Language: {language}.
Context: Irish/UK/International Common Law (depends on contract type).
Be concise. Remove ambiguity.
Output ONLY the new clause text.
"""

# --- ФУНКЦИИ GEMINI ---
def call_gemini(template, content, language="en"): # ⚠️ DEFAULT IS NOW ENGLISH
    final_prompt = template.format(language=language)
    
    if not CLIENT: 
        return None 
    
    for model in MODEL_CANDIDATES:
        try:
            resp = CLIENT.models.generate_content(
                model=model, 
                contents=f"SYSTEM: {final_prompt}\n\nUSER CONTENT:\n{content}",
                config={"response_mime_type": "text/plain"}
            )
            if resp.text: return resp.text.strip()
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    return None

# --- API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzeReq(BaseModel):
    text: str
    language: str = "en" # ⚠️ DEFAULT EN
    format: str = "readable" 

class AnalyzeDocReq(BaseModel):
    doc_id: str
    language: str = "en" # ⚠️ DEFAULT EN

class RewriteReq(BaseModel):
    clause: str
    language: str = "en" # ⚠️ DEFAULT EN

@app.post("/analyze_one")
def analyze_one(req: AnalyzeReq):
    # ⚠️ ЗАЩИТА ОТ ПУСТОТЫ / РАЗМЫТЫХ ФОТО
    if not req.text or len(req.text.strip()) < 10:
        return JSONResponse(content={
            "risk_score": 0, 
            "summary": "Could not read text. Please upload a clearer image or PDF.", 
            "risks": []
        })

    raw_response = call_gemini(READABLE_PROMPT_TEMPLATE, req.text, req.language)
    
    if not raw_response:
        return JSONResponse(content={"risk_score": 0, "summary": "Error: AI Service Unavailable", "risks": []})

    clean_json = raw_response.replace("```json", "").replace("```", "").strip()
    
    try:
        data = json.loads(clean_json)
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        return JSONResponse(content={
            "risk_score": 0, 
            "summary": "Error parsing AI response. Please try again.", 
            "risks": []
        })

@app.post("/rewrite_clause")
def rewrite_clause(req: RewriteReq):
    res = call_gemini(REWRITE_PROMPT_TEMPLATE, req.clause, req.language)
    return {"safe_clause": res or "Error generating fix."}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    doc_id = file_sha256(temp_path)
    
    # 🔥 ПЕРЕДАЕМ content_type ЧТОБЫ СЕРВЕР ПОНЯЛ, ЧТО ЭТО КАРТИНКА
    text = extract_text_from_file(temp_path, file.filename, content_type=file.content_type)
    
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT OR REPLACE INTO docs (doc_id, filename, plain_text, created_at) VALUES (?, ?, ?, ?)", 
                (doc_id, file.filename, text, int(time.time())))
    con.commit()
    con.close()
    
    # 2. ПРОВЕРКА КАЧЕСТВА (PRE-FLIGHT CHECK)
    is_valid = len(text.strip()) > 50
    
    return {
        "doc_id": doc_id, 
        "valid": is_valid, 
        "char_count": len(text.strip()),
        "preview": text[:200] if is_valid else "Text unreadable"
    }

@app.post("/analyze_by_doc_id")
def analyze_by_doc_id(req: AnalyzeDocReq):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT plain_text FROM docs WHERE doc_id=?", (req.doc_id,)).fetchone()
    con.close()
    
    if not row: raise HTTPException(404, "File not found")
    
    raw_response = call_gemini(READABLE_PROMPT_TEMPLATE, row[0], req.language)
    clean_json = raw_response.replace("```json", "").replace("```", "").strip() if raw_response else "{}"
    
    try:
        data = json.loads(clean_json)
        return JSONResponse(content=data)
    except:
        return JSONResponse(content={"risk_score": 0, "summary": "Error parsing result", "risks": []})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)