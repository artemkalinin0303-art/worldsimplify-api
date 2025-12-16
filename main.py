import os
import json
import time
import hashlib
import sqlite3
import mimetypes 
import logging # Добавим логирование для отладки
from typing import List

# 👇 ЭТА БИБЛИОТЕКА НУЖНА ДЛЯ POSTGRES
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

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
DATABASE_URL = os.getenv("DATABASE_URL") # 👈 Читаем ссылку на базу Render

CLIENT = Client(api_key=API_KEY) if API_KEY else None
MODEL_CANDIDATES = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
UPLOAD_DIR = "uploads"
DB_PATH = "worldsimplify.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 🔌 УПРАВЛЕНИЕ БАЗОЙ ДАННЫХ (HYBRID MODE) ---
def get_db_connection():
    """
    Автоматически выбирает базу:
    1. Если есть DATABASE_URL -> подключаемся к PostgreSQL (Render).
    2. Если нет -> создаем локальный файл sqlite (для тестов дома).
    """
    if DATABASE_URL and psycopg2:
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            return conn, "POSTGRES"
        except Exception as e:
            logger.error(f"Postgres connection failed: {e}")
            # Если база упала, падаем на SQLite (резерв)
            return sqlite3.connect(DB_PATH), "SQLITE"
    else:
        return sqlite3.connect(DB_PATH), "SQLITE"

def db_init():
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    
    # Разница в типах данных: Postgres любит TEXT/BIGINT, SQLite прощает всё
    # Мы используем универсальный SQL
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs(
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        plain_text TEXT,
        created_at BIGINT
    )""")
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized using: {db_type}")

# Запускаем создание таблицы при старте
db_init()

# --- УТИЛИТЫ ---
def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text_from_file(filepath: str, filename: str, content_type: str = None) -> str:
    mime_type = content_type
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(filepath)
    
    if not mime_type and (filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
        mime_type = 'image/jpeg'

    logger.info(f"📂 Processing File: {filename} | Type: {mime_type}")
    text = ""
    
    try:
        if mime_type and mime_type.startswith('image'):
            if CLIENT:
                logger.info(f"📷 Sending image to AI OCR...")
                with open(filepath, "rb") as f:
                    image_data = f.read()
                try:
                    resp = CLIENT.models.generate_content(
                        model="gemini-2.0-flash", 
                        contents=[
                            "Transcribe text exactly.", 
                            {"mime_type": mime_type, "data": image_data}
                        ]
                    )
                    text = resp.text if resp.text else ""
                except Exception as img_err:
                    logger.error(f"OCR Error: {img_err}")
                    text = ""

        elif filename.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(filepath)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        
        elif filename.lower().endswith(".docx"):
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""
        
    return text.strip()

# --- ПРОМПТЫ ---
READABLE_PROMPT_TEMPLATE = """
Act as a Senior Legal Risk Auditor.
Your goal is to protect the Client.

CONTEXT: Irish Law & UK Common Law (unless specified otherwise).

INSTRUCTIONS:
Step 1: IDENTIFY THE CONTRACT TYPE.
Step 2: Look for "Silent Killers" (Rent Pressure Zones, Unpaid Overtime, IP Transfer).
Step 3: ANALYZE risks.

RETURN JSON ONLY:
{{
  "risk_score": integer (0-100),
  "contract_type": "string",
  "summary": "string",
  "risks": [ {{ "text": "string", "severity": "High|Medium|Low", "original_clause": "string" }} ]
}}
"""

REWRITE_PROMPT_TEMPLATE = """
Rewrite this contract clause to be safe and fair for the Client.
Language: {language}.
Context: Irish/UK/International Common Law.
Output ONLY the new clause text.
"""

def call_gemini(template, content, language="en"):
    final_prompt = template.format(language=language)
    if not CLIENT: return None
    
    for model in MODEL_CANDIDATES:
        try:
            resp = CLIENT.models.generate_content(
                model=model, 
                contents=f"SYSTEM: {final_prompt}\n\nUSER CONTENT:\n{content}",
                config={"response_mime_type": "text/plain"}
            )
            if resp.text: return resp.text.strip()
        except Exception as e:
            logger.error(f"Model {model} failed: {e}")
            continue
    return None

# --- API ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzeReq(BaseModel):
    text: str
    language: str = "en"

class AnalyzeDocReq(BaseModel):
    doc_id: str
    language: str = "en"

class RewriteReq(BaseModel):
    clause: str
    language: str = "en"

@app.post("/analyze_one")
def analyze_one(req: AnalyzeReq):
    if not req.text or len(req.text.strip()) < 10:
        return JSONResponse(content={"risk_score": 0, "summary": "Text unclear.", "risks": []})

    raw = call_gemini(READABLE_PROMPT_TEMPLATE, req.text, req.language)
    if not raw:
        return JSONResponse(content={"risk_score": 0, "summary": "Service Unavailable", "risks": []})

    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return JSONResponse(content=json.loads(clean))
    except:
        return JSONResponse(content={"risk_score": 0, "summary": "Error parsing AI", "risks": []})

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
    text = extract_text_from_file(temp_path, file.filename, content_type=file.content_type)
    
    # 👇 МАГИЯ HYBRID SQL: Выбираем правильный запрос
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    
    created_at = int(time.time())
    
    try:
        if db_type == "POSTGRES":
            # Синтаксис для PostgreSQL (%s и ON CONFLICT)
            query = """
                INSERT INTO docs (doc_id, filename, plain_text, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE 
                SET filename = EXCLUDED.filename, plain_text = EXCLUDED.plain_text;
            """
            cur.execute(query, (doc_id, file.filename, text, created_at))
        else:
            # Синтаксис для SQLite (? и INSERT OR REPLACE)
            query = "INSERT OR REPLACE INTO docs (doc_id, filename, plain_text, created_at) VALUES (?, ?, ?, ?)"
            cur.execute(query, (doc_id, file.filename, text, created_at))
            
        conn.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")
    finally:
        conn.close()
    
    is_valid = len(text.strip()) > 2
    return {
        "doc_id": doc_id, 
        "valid": is_valid, 
        "preview": text[:200] if is_valid else "Unreadable"
    }

@app.post("/analyze_by_doc_id")
def analyze_by_doc_id(req: AnalyzeDocReq):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    
    # Для чтения синтаксис почти одинаковый, но Placeholder разный
    placeholder = "%s" if db_type == "POSTGRES" else "?"
    
    cur.execute(f"SELECT plain_text FROM docs WHERE doc_id={placeholder}", (req.doc_id,))
    row = cur.fetchone()
    conn.close()
    
    if not row: raise HTTPException(404, "File not found")
    
    # В Postgres row это кортеж, в sqlite тоже, если не использовать row_factory
    text_content = row[0]
    
    raw = call_gemini(READABLE_PROMPT_TEMPLATE, text_content, req.language)
    try:
        clean = raw.replace("```json", "").replace("```", "").strip() if raw else "{}"
        return JSONResponse(content=json.loads(clean))
    except:
        return JSONResponse(content={"risk_score": 0, "summary": "Error parsing result", "risks": []})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)