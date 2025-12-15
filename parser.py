# parser.py
import os, re
from typing import List, Dict, Any

import pdfplumber
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

CLAUSE_RE = re.compile(
    r"""(?P<num>(?:^|\n)(?:\d+\.){1,3}|(?:^|\n)\d+\)|(?:^|\n)[IVXLCM]+\.\s)""",
    re.VERBOSE | re.IGNORECASE
)

def _normalize(s: str) -> str:
    s = s.replace('\r','\n')
    s = re.sub(r'[ \t]+',' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def _split_clauses(text: str) -> List[Dict[str, Any]]:
    text = _normalize(text)
    ms = list(CLAUSE_RE.finditer(text))
    if not ms:
        return [{"id":"clause_1","title":"Весь текст","text":text}]
    out=[]
    for i,m in enumerate(ms):
        start=m.start()
        end=ms[i+1].start() if i+1<len(ms) else len(text)
        chunk=text[start:end].strip()
        title=chunk.split('\n',1)[0][:120]
        out.append({"id":f"clause_{i+1}","title":title,"text":chunk})
    return out

def _detect_lang(text:str)->str:
    try:
        sample=text[:5000] if len(text)>5000 else text
        return detect(sample)
    except Exception:
        return "unknown"

def parse_pdf(path:str)->Dict[str,Any]:
    pages=[]; all_txt=[]
    with pdfplumber.open(path) as pdf:
        for i,pg in enumerate(pdf.pages,1):
            t=pg.extract_text() or ""
            t=_normalize(t)
            pages.append({"page":i,"text":t}); all_txt.append(t)
    merged="\n\n".join(p["text"] for p in pages).strip()

    # если PDF — скан (почти нет текста), делаем OCR
    if len(merged)<200:
        images=convert_from_path(path)
        pages=[]; all_txt=[]
        for i,img in enumerate(images,1):
            ocr=pytesseract.image_to_string(img, lang="rus+ukr+eng+tur")
            ocr=_normalize(ocr)
            pages.append({"page":i,"text":ocr}); all_txt.append(ocr)
        merged="\n\n".join(all_txt).strip()

    lang=_detect_lang(merged)
    return {
        "mime":"application/pdf",
        "pages":pages,
        "plain_text":merged,
        "sections":_split_clauses(merged),
        "metadata":{"detected_language":lang}
    }

def parse_docx(path:str)->Dict[str,Any]:
    doc=Document(path)
    paras=[p.text.strip() for p in doc.paragraphs if p.text.strip()]
    merged=_normalize("\n".join(paras))
    lang=_detect_lang(merged)
    return {
        "mime":"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pages":[{"page":1,"text":merged}],
        "plain_text":merged,
        "sections":_split_clauses(merged),
        "metadata":{"detected_language":lang}
    }

def parse_image(path:str)->Dict[str,Any]:
    img=Image.open(path)
    ocr=pytesseract.image_to_string(img, lang="rus+ukr+eng+tur")
    merged=_normalize(ocr)
    lang=_detect_lang(merged)
    return {
        "mime":"image",
        "pages":[{"page":1,"text":merged}],
        "plain_text":merged,
        "sections":_split_clauses(merged),
        "metadata":{"detected_language":lang}
    }

def parse_any(path:str)->Dict[str,Any]:
    ext=os.path.splitext(path.lower())[1]
    if ext==".pdf": return parse_pdf(path)
    if ext==".docx": return parse_docx(path)
    if ext in {".jpg",".jpeg",".png",".tif",".tiff"}: return parse_image(path)
    raise ValueError(f"Unsupported file type: {ext}")
