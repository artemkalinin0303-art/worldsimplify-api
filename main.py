import os
import json
import time
import hashlib
import sqlite3
import mimetypes 
import logging
from typing import List, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pypdf
import docx
from google.genai import Client

load_dotenv()

# --- КОНФИГУРАЦИЯ ---
API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") 

CLIENT = Client(api_key=API_KEY) if API_KEY else None
MODEL_CANDIDATES = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
UPLOAD_DIR = "uploads"
DB_PATH = "worldsimplify.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- БАЗА ДАННЫХ ---
def get_db_connection():
    if DATABASE_URL and psycopg2:
        try:
            conn = psycopg2.connect(DATABASE_URL, sslmode='require')
            return conn, "POSTGRES"
        except Exception as e:
            logger.error(f"Postgres connection failed: {e}")
            return sqlite3.connect(DB_PATH), "SQLITE"
    else:
        return sqlite3.connect(DB_PATH), "SQLITE"

def db_init():
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    
    # Создаем таблицу
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs(
        doc_id TEXT PRIMARY KEY,
        user_id TEXT,
        filename TEXT,
        plain_text TEXT,
        created_at BIGINT,
        risk_score INTEGER,
        summary TEXT,
        full_report TEXT
    )""")
    
    # ⚡️ МИГРАЦИЯ: Добавляем колонку full_report, если её нет
    columns_to_add = [
        ("user_id", "TEXT"),
        ("risk_score", "INTEGER"),
        ("summary", "TEXT"),
        ("full_report", "TEXT") # 👈 Новая колонка для полного отчета
    ]
    
    for col_name, col_type in columns_to_add:
        try:
            if db_type == "POSTGRES":
                cur.execute(f"ALTER TABLE docs ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
            else:
                try: cur.execute(f"ALTER TABLE docs ADD COLUMN {col_name} {col_type};")
                except: pass 
        except Exception as e:
            logger.warning(f"Migration warning for {col_name}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Database initialized: {db_type}")

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

    text = ""
    try:
        if mime_type and mime_type.startswith('image'):
            if CLIENT:
                with open(filepath, "rb") as f:
                    image_data = f.read()
                try:
                    resp = CLIENT.models.generate_content(
                        model="gemini-2.0-flash", 
                        contents=["Transcribe text exactly.", {"mime_type": mime_type, "data": image_data}]
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
Step 2: Look for "Silent Killers".
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

@app.post("/rewrite_clause")
def rewrite_clause(req: RewriteReq):
    res = call_gemini(REWRITE_PROMPT_TEMPLATE, req.clause, req.language)
    return {"safe_clause": res or "Error generating fix."}

@app.get("/history/{user_id}")
def get_history(user_id: str):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    query = "SELECT doc_id, filename, created_at, risk_score, summary FROM docs WHERE user_id = %s ORDER BY created_at DESC"
    if db_type == "SQLITE": query = query.replace("%s", "?")
    cur.execute(query, (user_id,))
    rows = cur.fetchall()
    conn.close()
    history = []
    for r in rows:
        history.append({
            "doc_id": r[0],
            "filename": r[1],
            "date": time.strftime('%Y-%m-%d', time.localtime(r[2])) if r[2] else "Unknown",
            "risk_score": r[3], 
            "summary": r[4]
        })
    return history

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    doc_id = file_sha256(temp_path)
    text = extract_text_from_file(temp_path, file.filename, content_type=file.content_type)
    
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    created_at = int(time.time())
    try:
        if db_type == "POSTGRES":
            query = """
                INSERT INTO docs (doc_id, user_id, filename, plain_text, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE 
                SET filename = EXCLUDED.filename, plain_text = EXCLUDED.plain_text, user_id = EXCLUDED.user_id;
            """
            cur.execute(query, (doc_id, user_id, file.filename, text, created_at))
        else:
            query = "INSERT OR REPLACE INTO docs (doc_id, user_id, filename, plain_text, created_at) VALUES (?, ?, ?, ?, ?)"
            cur.execute(query, (doc_id, user_id, file.filename, text, created_at))
        conn.commit()
    except Exception as e:
        logger.error(f"DB Error: {e}")
    finally:
        conn.close()
    return {"doc_id": doc_id, "valid": len(text.strip()) > 2, "preview": text[:200]}

@app.post("/analyze_one")
def analyze_one(req: AnalyzeReq):
    raw = call_gemini(READABLE_PROMPT_TEMPLATE, req.text, req.language)
    try:
        return JSONResponse(content=json.loads(raw.replace("```json", "").replace("```", "").strip()))
    except:
        return JSONResponse(content={"risk_score": 0, "summary": "Error parsing AI", "risks": []})

# 👇 ГЛАВНАЯ ФИШКА: УМНЫЙ КЕШ
@app.post("/analyze_by_doc_id")
def analyze_by_doc_id(req: AnalyzeDocReq):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    placeholder = "%s" if db_type == "POSTGRES" else "?"
    
    # 1. Проверяем: есть ли уже ГОТОВЫЙ отчет?
    cur.execute(f"SELECT plain_text, full_report FROM docs WHERE doc_id={placeholder}", (req.doc_id,))
    row = cur.fetchone()
    
    if not row: 
        conn.close()
        raise HTTPException(404, "File not found")
    
    plain_text = row[0]
    existing_report = row[1]

    # ✅ ЕСЛИ ОТЧЕТ ЕСТЬ — ВОЗВРАЩАЕМ МГНОВЕННО
    if existing_report and len(existing_report) > 10:
        logger.info(f"🚀 Serving cached report for {req.doc_id}")
        conn.close()
        return JSONResponse(content=json.loads(existing_report))

    # 2. Если нет — спрашиваем ИИ
    logger.info(f"🤖 Calling AI for {req.doc_id}")
    raw = call_gemini(READABLE_PROMPT_TEMPLATE, plain_text, req.language)
    
    try:
        clean = raw.replace("```json", "").replace("```", "").strip() if raw else "{}"
        result_json = json.loads(clean)
        
        # 3. Сохраняем отчет в базу НАВЕЧНО
        risk_score = result_json.get("risk_score", 0)
        summary = result_json.get("summary", "")
        full_report_str = json.dumps(result_json) # Превращаем JSON в строку для сохранения
        
        try:
            if db_type == "POSTGRES":
                update_q = "UPDATE docs SET risk_score = %s, summary = %s, full_report = %s WHERE doc_id = %s"
            else:
                update_q = "UPDATE docs SET risk_score = ?, summary = ?, full_report = ? WHERE doc_id = ?"
            
            cur.execute(update_q, (risk_score, summary, full_report_str, req.doc_id))
            conn.commit()
        except Exception as db_err:
            logger.error(f"Failed to save report: {db_err}")

        conn.close()
        return JSONResponse(content=result_json)
        
    except:
        conn.close()
        return JSONResponse(content={"risk_score": 0, "summary": "Error parsing result", "risks": []})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)