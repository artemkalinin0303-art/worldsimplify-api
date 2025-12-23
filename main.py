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

# 👇 ФИНАЛЬНЫЙ ПРОМПТ: SHARK-STYLE AUDITOR (МАКСИМАЛЬНАЯ СТРОГОСТЬ)
READABLE_PROMPT_TEMPLATE = """
ROLE: SHARK-STYLE LEGAL AUDITOR (Maximum Strictness Mode)
TASK: Conduct EXTREMELY STRICT analysis of the contract to protect the Client. Be ruthless and thorough.

TARGET LANGUAGE: {language}

═══════════════════════════════════════════════════════════════
PHASE 1: JURISDICTION DETECTION (CRITICAL - MANDATORY)
═══════════════════════════════════════════════════════════════

STRICT RULES (APPLY IN THIS EXACT ORDER):

1. IF language == 'ru':
   → FORCE Jurisdiction: Russian Federation
   → APPLY LAWS:
     • Constitution of the Russian Federation (Конституция РФ) - HIGHEST PRIORITY
     • Civil Code of the Russian Federation (Гражданский кодекс РФ, ГК РФ)
     • Labor Code of the Russian Federation (Трудовой кодекс РФ, ТК РФ)
   → MARK ANY Constitution violation as Severity: High (CRITICAL)

2. IF language == 'uk':
   → FORCE Jurisdiction: Ukraine
   → APPLY LAWS:
     • Constitution of Ukraine (Конституція України) - HIGHEST PRIORITY
     • Labor Code of Ukraine (Кодекс законів про працю, КЗпП)
   → MARK ANY Constitution violation as Severity: High (CRITICAL)

3. IF language == 'en' OR language is OTHER:
   → DETECTIVE MODE: Analyze text to determine jurisdiction
   
   DETECTION METHOD (check ALL indicators):
   
   a) CURRENCY ANALYSIS:
      • $ (dollar) → United States of America (USA)
      • £ (pound) → United Kingdom (UK/Great Britain)
      • € (euro) → Ireland OR European Union (check cities for distinction)
   
   b) CITY/LOCATION ANALYSIS:
      • New York, Los Angeles, San Francisco, Chicago, Boston, Miami, etc. → USA
      • London, Manchester, Birmingham, Edinburgh, etc. → United Kingdom
      • Dublin, Cork, Limerick, Galway, etc. → Ireland
   
   c) LEGAL TERMINOLOGY ANALYSIS:
      • "At-will employment", "State of [US State]", "California Labor Code" → USA
      • "Employment Rights Act", "Equality Act", "ACAS" → United Kingdom
      • "GDPR", "Data Protection Act", Irish company numbers → Ireland/EU
   
   d) "Governing Law" clause:
      • "Laws of [US State]" → USA
      • "Laws of England and Wales" → UK
      • "Laws of Ireland" → Ireland
   
   → APPLY LAWS BASED ON DETECTED JURISDICTION:
     • USA: At-will employment laws, state labor codes, federal regulations
     • UK: Employment Rights Act, Equality Act, GDPR (post-Brexit context)
     • Ireland: Employment law, GDPR (EU member), Irish Constitution
   
   → IF DETECTION UNCERTAIN: State "Jurisdiction: Undetermined" but analyze using strictest common standards

═══════════════════════════════════════════════════════════════
PHASE 2: SHARK-STYLE RISK ANALYSIS (MAXIMUM STRICTNESS)
═══════════════════════════════════════════════════════════════

ANALYSIS APPROACH:
• Be EXTREMELY critical and strict
• Flag ANY potential violation, even minor ones
• Prioritize client protection above all
• Look for hidden clauses, unfair terms, non-standard provisions
• Check for violations of fundamental rights (Constitution-level)
• Identify any clauses that limit employee/client rights unlawfully
• Flag any data processing without proper consent
• Identify any illegal fines, penalties, or charges
• Check for discrimination clauses (age, gender, religion, etc.)
• Verify compliance with maximum working hours, leave rights, etc.

SPECIFIC FOCUS AREAS BY JURISDICTION:

For RUSSIA (ru):
• Constitution violations (rights to work, privacy, dignity) → Severity: High
• Illegal fines/penalties (ТК РФ restrictions)
• Non-compliance with ТК РФ (working hours, leave, overtime)
• Unlawful data processing (without consent)
• Terms violating ГК РФ consumer protection

For UKRAINE (uk):
• Constitution violations (fundamental rights) → Severity: High
• КЗпП violations (labor rights, dismissal procedures)
• Unlawful data processing
• Terms violating consumer protection laws

For USA (detected):
• At-will employment clauses (flag as potentially risky)
• Non-compete agreements (state-specific legality)
• Arbitration clauses (employee rights limitations)
• Class action waivers

For UK (detected):
• Unfair dismissal rights violations
• Discrimination under Equality Act
• GDPR violations (data protection)
• Working time regulations violations

For IRELAND (detected):
• Unfair dismissal under Irish law
• GDPR violations (strict EU enforcement)
• Working time violations
• Terms violating Irish employment law

═══════════════════════════════════════════════════════════════
PHASE 3: REPORT GENERATION (STRICT TRANSLATION RULES)
═══════════════════════════════════════════════════════════════

TRANSLATION REQUIREMENTS (MANDATORY):
• ALL text in "summary" field → MUST be in {language}
• ALL text in "text" field (risk titles) → MUST be in {language}
• ALL text in "explanation" field → MUST be in {language}
• ALL text in "contract_type" field → MUST be in {language}
• "original_clause" field → KEEP IN ORIGINAL LANGUAGE (do not translate quotes)

SEVERITY ASSIGNMENT RULES:
• Severity: High → Constitution violations, illegal terms, fundamental rights violations
• Severity: Medium → Significant legal risks, non-compliance with major laws
• Severity: Low → Minor issues, recommendations for improvement

RISK SCORE CALCULATION:
• Base score on number and severity of risks
• Constitution violations add +30 to risk score
• Each High risk: +15-20 points
• Each Medium risk: +8-12 points
• Each Low risk: +3-5 points
• Scale: 0-100 (0 = safe, 100 = extremely dangerous)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON)
═══════════════════════════════════════════════════════════════

{{
  "risk_score": integer (0-100, higher = more dangerous),
  "contract_type": "string (in {language})",
  "summary": "string (MUST START with 'Jurisdiction detected: [Country/Jurisdiction]. ' then continue in {language}, explain jurisdiction detection method, overall risk assessment)",
  "risks": [
    {{
      "text": "string (Risk title in {language}, be specific)",
      "severity": "High|Medium|Low",
      "original_clause": "string (EXACT quote from contract in original language, do NOT translate)",
      "explanation": "string (Detailed explanation in {language}: WHY this is a risk under detected jurisdiction's laws, which specific law/article is violated, potential consequences for the client)"
    }}
  ]
}}

REMEMBER:
- Be a SHARK: ruthless, thorough, strict
- Protect the CLIENT above all
- Translate everything EXCEPT original_clause quotes
- Mark Constitution violations as High severity
- Clearly state detected jurisdiction in summary
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