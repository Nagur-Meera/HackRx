# Minimal Pinecone upsert test endpoint
@app.get("/hackrx/test-upsert")
def test_upsert():
    import requests
    import os
    PINECONE_HOST = os.getenv("PINECONE_HOST")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    upsert_url = f"{PINECONE_HOST}/vectors/upsert"
    headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
    pinecone_records = [
        {"id": "doc1#chunk1", "text": "the quick brown fox jumped over the lazy dog"}
    ]
    upsert_body = {"vectors": pinecone_records}
    print("[DEBUG] Pinecone upsert URL:", upsert_url)
    print("[DEBUG] Pinecone upsert payload:", upsert_body)
    upsert_resp = requests.post(upsert_url, headers=headers, json=upsert_body)
    print("[DEBUG] Pinecone upsert response status:", upsert_resp.status_code)
    print("[DEBUG] Pinecone upsert response text:", upsert_resp.text)
    return {
        "url": upsert_url,
        "payload": upsert_body,
        "status_code": upsert_resp.status_code,
        "response": upsert_resp.text
    }
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Pinecone API Key (for REST API only)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment variables.")


from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Allow CORS for all origins (optional, but useful for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

class QueryRequest(BaseModel):
    documents: str
    questions: list[str]



import tempfile
import requests as httpx
import requests
from PyPDF2 import PdfReader
import docx

# Gemini API integration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def ask_gemini(question, context, api_key=GEMINI_API_KEY):
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    data = {
        "contents": [
            {"parts": [{"text": f"Context: {context}\nQuestion: {question}"}]}
        ]
    }
    response = httpx.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return "Gemini response parsing error."
    else:
        return f"Gemini API error: {response.status_code}"

from urllib.parse import urlparse

def download_file(url):
    response = httpx.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download document.")
    # Parse the file extension from the URL path (ignore query params)
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith('.pdf'):
        suffix = '.pdf'
    elif path.endswith('.docx'):
        suffix = '.docx'
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and DOCX are supported.")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(response.content)
    temp.close()
    return temp.name

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks




@app.post("/hackrx/run")
async def hackrx_run(request: QueryRequest):
    try:
        # 1. Download and parse the document (PDF/DOCX)
        file_path = download_file(request.documents)
        try:
            text = extract_text(file_path)
        finally:
            os.remove(file_path)
        # 2. Chunk the text
        chunks = chunk_text(text)
        # 3. Upsert chunks to Pinecone (embedding generated automatically by integrated model)
        PINECONE_HOST = os.getenv("PINECONE_HOST")
        upsert_url = f"{PINECONE_HOST}/vectors/upsert"
        headers = {"Api-Key": PINECONE_API_KEY, "Content-Type": "application/json"}
        pinecone_records = []
        for idx, chunk in enumerate(chunks):
            pinecone_records.append({"id": f"chunk-{idx}", "text": chunk})
        upsert_body = {"vectors": pinecone_records}
        print("[DEBUG] Pinecone upsert URL:", upsert_url)
        print("[DEBUG] Pinecone upsert payload:", upsert_body)
        upsert_resp = requests.post(upsert_url, headers=headers, json=upsert_body)
        print("[DEBUG] Pinecone upsert response status:", upsert_resp.status_code)
        print("[DEBUG] Pinecone upsert response text:", upsert_resp.text)
        if not upsert_resp.ok:
            raise Exception(f"Pinecone upsert failed: {upsert_resp.text}")

        # 4. For each question: search Pinecone, get context, call Gemini LLM
        answers = []

        for question in request.questions:
            # Use Pinecone REST API for query with 'text' field (integrated model)
            query_url = f"{PINECONE_HOST}/query"
            query_body = {
                "topK": 3,
                "includeMetadata": True,
                "text": question
            }
            query_resp = requests.post(query_url, headers=headers, json=query_body)
            if not query_resp.ok:
                raise Exception(f"Pinecone query failed: {query_resp.text}")
            search_results = query_resp.json()
            # Extract top chunks' text as context
            top_chunks = []
            for match in search_results.get("matches", []):
                chunk_val = match.get("metadata", {}).get("text", "")
                if chunk_val:
                    top_chunks.append(chunk_val)
            context = "\n".join(top_chunks)
            # Call Gemini LLM
            answer = ask_gemini(question, context)
            answers.append(answer)

        return JSONResponse({"answers": answers})
    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

# To run: uvicorn main:app --reload
