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

def get_mime_type_for_image(filename: str) -> str:
    """Определяет правильный MIME type для изображения"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    mime_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'heic': 'image/heic',
        'webp': 'image/webp',
        'gif': 'image/gif',
        'bmp': 'image/bmp'
    }
    return mime_map.get(ext, 'image/jpeg')

def extract_text_from_file(filepath: str, filename: str, content_type: str = None) -> str:
    """
    Universal Reader: Извлекает текст из файлов разных форматов.
    Для PDF: сначала pypdf, если текста <50 символов - Gemini Vision OCR.
    Для изображений: сразу Gemini Vision OCR (лучше видит рукописный текст и плохие фото).
    """
    mime = content_type
    if not mime or mime == 'application/octet-stream': 
        mime, _ = mimetypes.guess_type(filepath)
    ext = filename.lower().split('.')[-1] if '.' in filename else ""
    is_image = (mime and mime.startswith('image')) or ext in ['jpg', 'jpeg', 'png', 'heic', 'webp', 'gif', 'bmp']
    is_pdf = (mime and 'pdf' in mime) or ext == 'pdf'
    
    text = ""
    
    try:
        # === ОБРАБОТКА ИЗОБРАЖЕНИЙ: СРАЗУ Gemini Vision ===
        if is_image:
            if not CLIENT:
                logger.warning("Gemini client not available for image OCR")
                return ""
            
            with open(filepath, "rb") as f: 
                image_data = f.read()
            
            # Определяем правильный MIME type для изображения
            image_mime = get_mime_type_for_image(filename)
            
            try:
                logger.info(f"🔍 Using Gemini Vision OCR for image: {filename} (MIME: {image_mime})")
                resp = CLIENT.models.generate_content(
                    model="gemini-2.0-flash", 
                    contents=[
                        "Extract ALL text from this image. Preserve formatting, line breaks, and structure. Transcribe exactly as shown, including handwritten text if present.",
                        {"mime_type": image_mime, "data": image_data}
                    ]
                )
                text = resp.text if resp.text else ""
                logger.info(f"✅ Extracted {len(text)} characters from image")
            except Exception as e:
                logger.error(f"❌ Gemini Vision OCR failed for {filename}: {e}")
                text = ""
        
        # === ОБРАБОТКА PDF: Сначала pypdf, затем Gemini Vision если нужно ===
        elif is_pdf:
            # Шаг 1: Пробуем извлечь текст через pypdf
            try:
                reader = pypdf.PdfReader(filepath)
                for page in reader.pages: 
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                
                text = text.strip()
                logger.info(f"📄 Extracted {len(text)} characters from PDF via pypdf")
                
                # Шаг 2: Если текста мало (<50 символов), используем Gemini Vision OCR
                if len(text) < 50 and CLIENT:
                    logger.info(f"⚠️ PDF has little text ({len(text)} chars), trying Gemini Vision OCR...")
                    try:
                        with open(filepath, "rb") as f:
                            pdf_data = f.read()
                        
                        # Отправляем PDF напрямую в Gemini Vision
                        resp = CLIENT.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[
                                "Extract ALL text from this PDF document. This appears to be a scanned document. Transcribe ALL text exactly, preserve formatting and structure.",
                                {"mime_type": "application/pdf", "data": pdf_data}
                            ]
                        )
                        ocr_text = resp.text if resp.text else ""
                        if len(ocr_text) > len(text):
                            text = ocr_text
                            logger.info(f"✅ Gemini Vision OCR extracted {len(text)} characters from PDF")
                        else:
                            logger.warning(f"⚠️ Gemini Vision OCR didn't improve extraction ({len(ocr_text)} vs {len(text)} chars)")
                    except Exception as e:
                        logger.error(f"❌ Gemini Vision OCR for PDF failed: {e}, using pypdf result")
            
            except Exception as e:
                logger.error(f"❌ PDF extraction failed: {e}")
                text = ""
        
        # === ОБРАБОТКА DOCX ===
        elif ext == 'docx':
            try:
                doc = docx.Document(filepath)
                for para in doc.paragraphs: 
                    text += para.text + "\n"
                text = text.strip()
            except Exception as e:
                logger.error(f"❌ DOCX extraction failed: {e}")
                text = ""
        
        # === ОБРАБОТКА ТЕКСТОВЫХ ФАЙЛОВ ===
        else:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f: 
                    text = f.read()
                text = text.strip()
            except Exception as e:
                logger.error(f"❌ Text file reading failed: {e}")
                text = ""
    
    except Exception as e:
        logger.error(f"❌ General extraction error for {filename}: {e}")
        return ""
    
    return text.strip()

# 👇 ФИНАЛЬНЫЙ ПРОМПТ: AGGRESSIVE DEFENSE LAWYER (МАТРИЦА РИСКОВ)
READABLE_PROMPT_TEMPLATE = """
ROLE: AGGRESSIVE DEFENSE LAWYER (Агрессивный защитник клиента)
TASK: Conduct COMPREHENSIVE analysis of the contract to identify ALL risks for the Client. Your job is to find EVERYTHING that could harm the client - legal violations, financial traps, power imbalances, vague terms, and toxic clauses. Be aggressive, thorough, and protective.

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
PHASE 2: MATRIX OF RISKS - 5 CATEGORIES ANALYSIS
═══════════════════════════════════════════════════════════════

ANALYSIS APPROACH:
You must check the contract against ALL 5 categories below. Flag EVERY risk you find, no matter how small. Your goal is to protect the client from ANYTHING that could harm them.

═══════════════════════════════════════════════════════════════
🛑 CATEGORY 1: LEGAL VIOLATIONS
═══════════════════════════════════════════════════════════════

Check for violations of laws in the detected jurisdiction:

For RUSSIA (ru):
• Constitution violations (rights to work, privacy, dignity, freedom) → Severity: High
• ТК РФ violations: illegal fines/penalties, non-compliance with working hours, leave, overtime
• ГК РФ violations: consumer protection, unfair contract terms
• Unlawful data processing without consent
• Discrimination clauses (age, gender, religion, etc.)

For UKRAINE (uk):
• Constitution violations (fundamental rights) → Severity: High
• КЗпП violations: labor rights, dismissal procedures, working conditions
• Unlawful data processing
• Terms violating consumer protection laws
• Discrimination clauses

For USA (detected):
• Violations of federal/state labor laws
• At-will employment clauses (flag as potentially risky)
• Non-compete agreements (check state-specific legality)
• Arbitration clauses that limit employee rights
• Class action waivers
• Violations of ADA, Title VII, FLSA

For UK (detected):
• Unfair dismissal rights violations
• Discrimination under Equality Act
• GDPR violations (data protection)
• Working time regulations violations
• Consumer Rights Act violations

For IRELAND (detected):
• Unfair dismissal under Irish law
• GDPR violations (strict EU enforcement)
• Working time violations
• Terms violating Irish employment/consumer law

═══════════════════════════════════════════════════════════════
💰 CATEGORY 2: FINANCIAL TRAPS
═══════════════════════════════════════════════════════════════

Look for ANY financial risks that could cost the client money:

• HIDDEN FEES / Скрытые платежи:
  - Fees mentioned in fine print but not in main price
  - "Administrative fees", "Processing fees", "Service fees" without clear disclosure
  - Fees that appear only after signing
  → Severity: High if significant, Medium if minor

• UNCAPPED PENALTIES / Штрафы без лимита:
  - Penalties without maximum limit
  - "Penalty of X% per day" without cap
  - Compound interest on penalties
  - Penalties that can exceed principal amount
  → Severity: High

• PRICE INCREASES WITHOUT CONSENT / Повышение цены без согласия:
  - "We reserve the right to change prices"
  - "Prices may vary" without notice period
  - Automatic price increases
  - Price changes without client's explicit consent
  → Severity: High

• AUTO-RENEWAL TRAPS / Автопродление без уведомления:
  - Automatic renewal without clear opt-out
  - Renewal at higher price without notice
  - "Contract renews automatically unless cancelled 30 days before" (unfair notice period)
  - Hidden auto-renewal clauses
  → Severity: High if no easy cancellation, Medium if difficult cancellation

• OTHER FINANCIAL RISKS:
  - Early termination fees that are excessive
  - "Liquidated damages" clauses that are punitive
  - Payment terms that favor the other party unfairly
  - Currency conversion fees
  → Severity: Medium to High depending on impact

═══════════════════════════════════════════════════════════════
⚖️ CATEGORY 3: IMBALANCE OF POWER
═══════════════════════════════════════════════════════════════

Look for clauses that create unfair power imbalance:

• ONE-SIDED TERMINATION RIGHTS:
  - "We can cancel at any time, you cannot"
  - "We reserve the right to terminate without cause"
  - Client termination requires 90 days notice, provider requires 7 days
  → Severity: High

• ONE-SIDED LIABILITY:
  - "We are not liable for anything, you are liable for everything"
  - Broad liability waivers for the other party
  - Client assumes all risks, provider assumes none
  - "As-is" clauses that remove all warranties
  → Severity: High

• UNFAIR TERMINATION CLAUSES:
  - Client can only terminate for "material breach" but definition is vague
  - Provider can terminate for minor reasons
  - No refund upon termination by provider
  → Severity: High

• OTHER POWER IMBALANCES:
  - One party can modify terms unilaterally
  - Dispute resolution favors one party
  - "Entire agreement" clauses that prevent client from relying on promises
  → Severity: Medium to High

═══════════════════════════════════════════════════════════════
🌫 CATEGORY 4: VAGUE DEFINITIONS
═══════════════════════════════════════════════════════════════

Look for vague, ambiguous terms that could be interpreted against the client:

• TIME-RELATED VAGUENESS:
  - "Reasonable time" without definition
  - "Immediately" without specific timeframe
  - "As soon as possible" without deadline
  - "Within a reasonable period" - what is reasonable?
  → Severity: Medium (can become High if used in critical clauses)

• EFFORT-RELATED VAGUENESS:
  - "Reasonable efforts" without metrics
  - "Best efforts" - what does this mean?
  - "Commercially reasonable" without definition
  → Severity: Medium

• QUALITY-RELATED VAGUENESS:
  - "Satisfactory quality" without standards
  - "Professional standards" without specification
  - "Industry standard" - which industry? which standard?
  → Severity: Medium to Low

• OTHER VAGUE TERMS:
  - "Material breach" without definition
  - "Substantial performance" without metrics
  - "Force majeure" defined too broadly
  → Severity: Medium

═══════════════════════════════════════════════════════════════
☠️ CATEGORY 5: TOXIC CLAUSES
═══════════════════════════════════════════════════════════════

Look for clauses that are particularly harmful to the client:

• NON-COMPETE CLAUSES:
  - Non-compete longer than 6 months (excessive)
  - Geographic scope too broad (e.g., "worldwide")
  - Industry scope too broad
  - Non-compete for low-level positions
  → Severity: High if > 6 months or too broad, Medium if reasonable but still restrictive

• LOSS OF INTELLECTUAL PROPERTY (IP) RIGHTS:
  - "All work product becomes our property"
  - "You assign all IP rights to us"
  - "Work for hire" clauses that are too broad
  - Client loses rights to their own creations
  → Severity: High

• FORCED ARBITRATION:
  - Mandatory arbitration clauses
  - Arbitration in inconvenient location
  - Arbitration rules favor the other party
  - Waiver of class action rights
  → Severity: High (especially if combined with other restrictions)

• OTHER TOXIC CLAUSES:
  - "No poaching" clauses that prevent hiring
  - Confidentiality clauses that are too broad
  - "Gag orders" preventing client from speaking
  - Clauses that prevent client from working with competitors
  → Severity: Medium to High depending on scope

═══════════════════════════════════════════════════════════════
PHASE 3: SEVERITY ASSIGNMENT & RISK SCORE
═══════════════════════════════════════════════════════════════

SEVERITY ASSIGNMENT RULES (CRITICAL):

HIGH Severity - Assign when:
• Client loses money (significant financial loss, uncapped penalties, hidden fees)
• Client loses IP rights (work becomes property of other party)
• Client goes to court (forced arbitration, class action waivers)
• Client cannot cancel contract (unfair termination, auto-renewal traps)
• Constitution/fundamental rights violations
• Illegal terms that could be unenforceable but still risky

MEDIUM Severity - Assign when:
• Unfair terms (power imbalance, one-sided liability)
• Vague deadlines/definitions that could be interpreted against client
• Annoying penalties (capped but still significant)
• Restrictive but not illegal clauses (short non-competes, reasonable confidentiality)

LOW Severity - Assign when:
• Missing definitions (minor gaps in clarity)
• Minor typos or formatting issues
• Recommendations for improvement (best practices not followed)
• Minor ambiguities that are unlikely to cause harm

RISK SCORE CALCULATION:
• Start with base score of 0
• Constitution violations: +30 points
• Each HIGH risk: +15-20 points
• Each MEDIUM risk: +8-12 points
• Each LOW risk: +3-5 points
• Maximum score: 100 (extremely dangerous contract)
• Scale: 0-100 (0 = safe, 100 = extremely dangerous)

═══════════════════════════════════════════════════════════════
PHASE 4: REPORT GENERATION (STRICT TRANSLATION RULES)
═══════════════════════════════════════════════════════════════

TRANSLATION REQUIREMENTS (MANDATORY):
• ALL text in "summary" field → MUST be in {language}
• ALL text in "text" field (risk titles) → MUST be in {language}
• ALL text in "explanation" field → MUST be in {language}
• ALL text in "contract_type" field → MUST be in {language}
• "original_clause" field → KEEP IN ORIGINAL LANGUAGE (do not translate quotes)

REPORT STRUCTURE:
• Summary must START with: "Jurisdiction detected: [Country/Jurisdiction]. "
• Then provide overall risk assessment in {language}
• Explain which categories of risks were found
• Mention total number of risks and their severity distribution

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON)
═══════════════════════════════════════════════════════════════

{{
  "risk_score": integer (0-100, higher = more dangerous),
  "contract_type": "string (in {language}, e.g., 'Employment Contract', 'Service Agreement', 'Purchase Agreement')",
  "summary": "string (MUST START with 'Jurisdiction detected: [Country/Jurisdiction]. ' then continue in {language}, explain jurisdiction detection, overall risk assessment across all 5 categories, total risks found)",
  "risks": [
    {{
      "text": "string (Risk title in {language}, be specific and mention category if relevant, e.g., 'Financial Trap: Uncapped Penalties', 'Imbalance of Power: One-Sided Termination Rights')",
      "severity": "High|Medium|Low",
      "original_clause": "string (EXACT quote from contract in original language, do NOT translate)",
      "explanation": "string (Detailed explanation in {language}: WHY this is a risk, which category it belongs to, potential consequences for the client, specific law/article violated if applicable)"
    }}
  ]
}}

REMEMBER:
- You are an AGGRESSIVE DEFENSE LAWYER - find EVERYTHING that could harm the client
- Check ALL 5 categories: Legal Violations, Financial Traps, Imbalance of Power, Vague Definitions, Toxic Clauses
- Be thorough - flag even minor risks
- Assign severity based on actual impact: High = money loss, IP loss, court, can't cancel
- Translate everything EXCEPT original_clause quotes
- Mark Constitution violations as High severity
- Clearly state detected jurisdiction in summary
- Organize risks by category in your analysis (you can mention category in risk title)
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
async def upload(files: List[UploadFile] = File(...), user_id: Optional[str] = Form(None)):
    """
    Mass Upload: Принимает до 30 файлов, извлекает текст из каждого и объединяет в один документ.
    """
    # Лимит: максимум 30 файлов
    if len(files) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 files allowed per upload")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    logger.info(f"📤 Mass upload started: {len(files)} files for user {user_id}")
    
    # Список для хранения всех текстов с метаданными
    extracted_texts = []
    saved_paths = []  # Для очистки временных файлов
    
    try:
        # Обрабатываем каждый файл
        for idx, file in enumerate(files, 1):
            try:
                # Сохраняем файл во временную директорию
                safe_filename = f"{int(time.time() * 1000)}_{idx}_{file.filename}"
                temp_path = os.path.join(UPLOAD_DIR, safe_filename)
                
                with open(temp_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                saved_paths.append(temp_path)
                
                # Извлекаем текст
                logger.info(f"📄 Processing file {idx}/{len(files)}: {file.filename}")
                text = extract_text_from_file(temp_path, file.filename, content_type=file.content_type)
                
                if text:
                    extracted_texts.append({
                        "filename": file.filename,
                        "page_num": idx,
                        "text": text
                    })
                    logger.info(f"✅ Extracted {len(text)} characters from {file.filename}")
                else:
                    logger.warning(f"⚠️ No text extracted from {file.filename}")
                    extracted_texts.append({
                        "filename": file.filename,
                        "page_num": idx,
                        "text": ""
                    })
            
            except Exception as e:
                logger.error(f"❌ Error processing file {file.filename}: {e}")
                extracted_texts.append({
                    "filename": file.filename,
                    "page_num": idx,
                    "text": f"[Error extracting text from {file.filename}: {str(e)}]"
                })
        
        # Объединяем все тексты с разделителями
        full_text_parts = []
        for item in extracted_texts:
            if item["text"]:
                full_text_parts.append(f"\n\n--- Page {item['page_num']} ({item['filename']}) ---\n\n")
                full_text_parts.append(item["text"])
        
        full_text = "".join(full_text_parts).strip()
        
        # Создаем составное имя файла (первые несколько имен + ...)
        if len(files) == 1:
            composite_filename = files[0].filename
        elif len(files) <= 3:
            composite_filename = " + ".join(f.filename for f in files)
        else:
            composite_filename = f"{files[0].filename} + ... + {files[-1].filename} ({len(files)} files)"
        
        # Генерируем doc_id на основе объединенного содержимого
        # Используем комбинацию всех файлов для создания уникального ID
        combined_hash_input = "|".join(item["filename"] + ":" + item["text"][:1000] for item in extracted_texts)
        doc_id = hashlib.sha256(combined_hash_input.encode('utf-8')).hexdigest()
        
        # Сохраняем объединенный текст в базу данных как один документ
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
            
            cur.execute(
                q.replace("%s", "?") if db_type == "SQLITE" else q, 
                (doc_id, user_id, composite_filename, full_text, created_at)
            )
            conn.commit()
            logger.info(f"✅ Saved merged document: {doc_id} ({len(full_text)} total characters)")
        except Exception as e:
            logger.error(f"❌ DB Error: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
        
        # Очищаем временные файлы
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete temp file {path}: {e}")
        
        # Возвращаем результат
        is_valid = len(full_text.strip()) > 1
        preview = full_text[:500] if is_valid else "Could not read text from any file."
        
        return {
            "doc_id": doc_id,
            "valid": is_valid,
            "preview": preview,
            "files_processed": len(files),
            "files_with_text": len([item for item in extracted_texts if item["text"]]),
            "total_characters": len(full_text),
            "composite_filename": composite_filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        # Очищаем временные файлы при ошибке
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

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