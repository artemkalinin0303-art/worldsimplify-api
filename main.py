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
    # Миграции
    columns = [("user_id", "TEXT"), ("risk_score", "INTEGER"), ("summary", "TEXT"), ("full_report", "TEXT")]
    for col, type_ in columns:
        try:
            if db_type == "POSTGRES": cur.execute(f"ALTER TABLE docs ADD COLUMN IF NOT EXISTS {col} {type_};")
            else: cur.execute(f"ALTER TABLE docs ADD COLUMN {col} {type_};")
        except: pass
    conn.commit()
    conn.close()

db_init()

# --- УТИЛИТЫ ---
def file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()

def extract_text_from_file(filepath: str, filename: str, content_type: str = None) -> str:
    mime = content_type
    if not mime or mime == 'application/octet-stream': mime, _ = mimetypes.guess_type(filepath)
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    is_image = (mime and mime.startswith('image')) or ext in ['jpg', 'jpeg', 'png', 'heic', 'webp']
    is_pdf = (mime and 'pdf' in mime) or ext == 'pdf'
    
    text = ""
    try:
        if is_image and CLIENT:
            with open(filepath, "rb") as f: image_data = f.read()
            try:
                resp = CLIENT.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=["Transcribe ALL text exactly.", {"mime_type": "image/jpeg", "data": image_data}]
                )
                text = resp.text if resp.text else ""
            except: pass
        elif is_pdf:
            try:
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages: text += (page.extract_text() or "") + "\n"
            except: pass
        elif ext == 'docx':
            doc = docx.Document(filepath)
            for para in doc.paragraphs: text += para.text + "\n"
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
    except: return ""
    return text.strip()

# 👇 ФИНАЛЬНЫЙ ПРОМПТ: УМНАЯ ЮРИСДИКЦИЯ
READABLE_PROMPT_TEMPLATE = """
ROLE: Senior Legal Risk Auditor (Shark-style).
TASK: Analyze the contract to protect the Client.
TARGET LANGUAGE: {language}.

PHASE 1: DETECT JURISDICTION (CRITICAL!)
1. IF LANGUAGE is 'ru' -> Force Jurisdiction: Russian Federation (Civil Code, Labor Code).
2. IF LANGUAGE is 'uk' -> Force Jurisdiction: Ukraine (Constitution, Labor Code).
3. IF LANGUAGE is 'en' or other -> DETECT FROM TEXT:
   - Look for "Governing Law" clause (e.g., "Laws of California", "Laws of England and Wales").
   - Look for Locations (e.g., "Dublin" -> Ireland, "London" -> UK, "New York" -> USA).
   - Look for Currency (€ -> EU/Ireland, £ -> UK, $ -> USA).
   - IF UNSURE: Default to "General Common Law" but warn the user.

PHASE 2: ANALYZE RISKS
- Apply the detected laws strictly.
- For USA: Watch out for "At-will employment" (normal there, but risky).
- For UK/Ireland: Watch out for "Unfair Dismissal" rights violations.
- For Russia/Ukraine: Watch out for "Illegal Fines" and "Data Processing" violations.

PHASE 3: REPORT GENERATION (IN {language})
- Translate Summary and Risks to {language}.
- Keep original quotes in original language.

STRICT JSON OUTPUT:
{{
  "risk_score": integer (0-100),
  "contract_type": "string",
  "summary": "string (Start with: 'Jurisdiction detected: [Country]...')", 
  "risks": [
    {{
      "text": "string (Risk title in {language})",
      "severity": "High|Medium|Low",
      "original_clause": "string",
      "explanation": "string (Explain WHY it is bad under [Country] law)"
    }}
  ]
}}
"""

# 👇 НОВЫЙ ПРОМПТ ДЛЯ ПЕРЕВОДА ГОТОВОГО JSON
TRANSLATE_JSON_TEMPLATE = """
TASK: Translate the values in this JSON object to {language}.
Do NOT translate keys (like "risk_score", "risks", "text").
Only translate the content strings (summary, text, explanation, contract_type).
Keep the structure exactly the same.
JSON:
{json_content}
"""

REWRITE_PROMPT_TEMPLATE = """
Rewrite clause to be SAFE. Language: {language}. Output ONLY new text.
"""

def call_gemini(template, content, language="en", json_mode=False):
    prompt = template.format(language=language, json_content=content) if json_mode else template.format(language=language)
    user_content = "Translate this JSON." if json_mode else content

    if not CLIENT: return None
    for model in MODEL_CANDIDATES:
        try:
            resp = CLIENT.models.generate_content(
                model=model, 
                contents=f"SYSTEM: {prompt}\n\nDATA:\n{user_content}",
                config={"response_mime_type": "application/json" if not "Rewrite" in template else "text/plain"}
            )
            return resp.text.strip()
        except: continue
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

@app.delete("/delete/{doc_id}")
def delete_document(doc_id: str):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    try:
        q = "DELETE FROM docs WHERE doc_id = %s" if db_type == "POSTGRES" else "DELETE FROM docs WHERE doc_id = ?"
        cur.execute(q, (doc_id,))
        conn.commit()
        return {"status": "ok"}
    except: raise HTTPException(500, "Failed")
    finally: conn.close()

@app.get("/history/{user_id}")
def get_history(user_id: str):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    q = "SELECT doc_id, filename, created_at, risk_score, summary FROM docs WHERE user_id = %s ORDER BY created_at DESC"
    if db_type == "SQLITE": q = q.replace("%s", "?")
    cur.execute(q, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"doc_id": r[0], "filename": r[1], "date": time.strftime('%Y-%m-%d', time.localtime(r[2])) if r[2] else "?", "risk_score": r[3], "summary": r[4]} for r in rows]

@app.post("/rewrite_clause")
def rewrite_clause(req: RewriteReq):
    res = call_gemini(REWRITE_PROMPT_TEMPLATE, req.clause, req.language)
    return {"safe_clause": res or "Error generating fix."}

@app.post("/upload")
async def upload(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f: f.write(await file.read())
    doc_id = file_sha256(temp_path)
    text = extract_text_from_file(temp_path, file.filename, content_type=file.content_type)
    
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    created_at = int(time.time())
    try:
        q = """
            INSERT INTO docs (doc_id, user_id, filename, plain_text, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE 
            SET filename = EXCLUDED.filename, plain_text = EXCLUDED.plain_text, user_id = EXCLUDED.user_id;
        """ if db_type == "POSTGRES" else "INSERT OR REPLACE INTO docs (doc_id, user_id, filename, plain_text, created_at) VALUES (?, ?, ?, ?, ?)"
        final_text = text if text else ""
        cur.execute(q.replace("%s", "?") if db_type == "SQLITE" else q, (doc_id, user_id, file.filename, final_text, created_at))
        conn.commit()
    except Exception as e: logger.error(f"DB Error: {e}")
    finally: conn.close()
    
    is_valid = len(text.strip()) > 1
    return {"doc_id": doc_id, "valid": is_valid, "preview": text[:200] if is_valid else "Could not read text."}

@app.post("/analyze_by_doc_id")
def analyze_by_doc_id(req: AnalyzeDocReq):
    conn, db_type = get_db_connection()
    cur = conn.cursor()
    ph = "%s" if db_type == "POSTGRES" else "?"
    cur.execute(f"SELECT plain_text, full_report FROM docs WHERE doc_id={ph}", (req.doc_id,))
    row = cur.fetchone()
    
    if not row: 
        conn.close()
        raise HTTPException(404, "File not found")
        
    plain_text, existing_report = row[0], row[1]

    # ✅ УМНЫЙ ПЕРЕВОД КЕША
    # Если отчет уже есть, мы не анализируем файл заново.
    # Но мы просим ИИ перевести этот JSON на нужный язык.
    if existing_report and len(existing_report) > 10:
        logger.info(f"🔄 Translating cached report for {req.doc_id} to {req.language}")
        try:
            # 1. Пробуем перевести готовый JSON
            translated_raw = call_gemini(TRANSLATE_JSON_TEMPLATE, existing_report, req.language, json_mode=True)
            translated_json = json.loads(translated_raw.replace("```json", "").replace("```", "").strip())
            conn.close()
            return JSONResponse(content=translated_json)
        except Exception as e:
            logger.error(f"Translation failed, returning original: {e}")
            conn.close()
            return JSONResponse(content=json.loads(existing_report))

    # Если отчета нет — полный анализ
    logger.info(f"🤖 Full Analysis for {req.doc_id}")
    raw = call_gemini(READABLE_PROMPT_TEMPLATE, plain_text, req.language)
    
    try:
        clean = raw.replace("```json", "").replace("```", "").strip() if raw else "{}"
        result_json = json.loads(clean)
        
        # Сохраняем в базу (первичный язык)
        risk_score = result_json.get("risk_score", 0)
        summary = result_json.get("summary", "")
        full_report = json.dumps(result_json)
        
        q = "UPDATE docs SET risk_score=%s, summary=%s, full_report=%s WHERE doc_id=%s" if db_type == "POSTGRES" else "UPDATE docs SET risk_score=?, summary=?, full_report=? WHERE doc_id=?"
        try:
            cur.execute(q, (risk_score, summary, full_report, req.doc_id))
            conn.commit()
        except: pass
        
        conn.close()
        return JSONResponse(content=result_json)
    except:
        conn.close()
        return JSONResponse(content={"risk_score": 0, "summary": "AI Error", "risks": []})

@app.post("/analyze_one")
def analyze_one(req: AnalyzeReq):
    raw = call_gemini(READABLE_PROMPT_TEMPLATE, req.text, req.language)
    try: return JSONResponse(content=json.loads(raw.replace("```json", "").replace("```", "").strip()))
    except: return JSONResponse(content={"risk_score": 0, "summary": "Error", "risks": []})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)