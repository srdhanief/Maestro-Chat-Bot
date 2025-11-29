# maestro_chat_improved.py
import os
import pickle
import time
import hashlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from gpt4all import GPT4All

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.dirname(__file__)
FAISS_INDEX_PATH = r'C:\Users\dsheikdawood\Documents\Hackathon\Maestro Chat Bot with RAG\faiss_index\index.faiss'
DOCS_PATH = r'C:\Users\dsheikdawood\Documents\Hackathon\Maestro Chat Bot with RAG\faiss_index\docs.pkl'
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

QUERY_EMB_CACHE = os.path.join(CACHE_DIR, "query_embeds.pkl")
RETRIEVAL_CACHE = os.path.join(CACHE_DIR, "retrievals.pkl")
ANSWER_CACHE = os.path.join(CACHE_DIR, "answers.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GPT4ALL_MODEL_NAME = "gpt4all-falcon-newbpe-q4_0.gguf"
GPT4ALL_MODEL_PATH = r'C:\Users\dsheikdawood\Documents\Hackathon\Maestro Chat Bot with RAG\models'

TOP_K = 8
MAX_CONTEXT_CHARS = 3000
MAX_NEW_TOKENS = 400

# -------------------------
# API init
# -------------------------
app = FastAPI(title="Maestro Chat Bot (High Accuracy RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load models
# -------------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
index = faiss.read_index(FAISS_INDEX_PATH)
docs = pickle.load(open(DOCS_PATH, "rb"))
llm = GPT4All(GPT4ALL_MODEL_NAME, model_path=GPT4ALL_MODEL_PATH, device="cpu")

# -------------------------
# Load caches
# -------------------------
def load_cache(path):
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except:
            return {}
    return {}

def save_cache(path, obj):
    pickle.dump(obj, open(path, "wb"))

query_emb_cache = load_cache(QUERY_EMB_CACHE)
retrieval_cache = load_cache(RETRIEVAL_CACHE)
answer_cache = load_cache(ANSWER_CACHE)

# -------------------------
# Hash helpers
# -------------------------
def h(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -------------------------
# Embedding
# -------------------------
def get_query_embedding(query: str):
    key = query.strip()
    if key in query_emb_cache:
        return np.array(query_emb_cache[key], dtype="float32")

    emb = embedder.encode([query], convert_to_numpy=True)[0].astype("float32")
    emb = emb / (np.linalg.norm(emb) + 1e-12)

    query_emb_cache[key] = emb.tolist()
    save_cache(QUERY_EMB_CACHE, query_emb_cache)
    return emb

# -------------------------
# Retrieval (Normalized Cosine Search)
# -------------------------
def retrieve_raw_context(query: str):
    retrieval_key = h(query)
    if retrieval_key in retrieval_cache:
        return retrieval_cache[retrieval_key]["context"], retrieval_cache[retrieval_key]["sources"]

    q_emb = get_query_embedding(query).reshape(1, -1)
    D, I = index.search(q_emb, TOP_K)
    idxs = [int(i) for i in I[0] if 0 <= i < len(docs)]
    if not idxs:
        return "", []

    # Build context by relevance order already returned by index
    context = "\n\n".join([docs[i] for i in idxs])
    sources = [f"chunk_{i}" for i in idxs]

    # Trim context but keep factual info intact
    context = context[:MAX_CONTEXT_CHARS]

    retrieval_cache[retrieval_key] = {"context": context, "sources": sources, "timestamp": time.time()}
    save_cache(RETRIEVAL_CACHE, retrieval_cache)
    return context, sources

# -------------------------
# Streaming answer generation
# -------------------------
def stream_answer(prompt: str):
    answer_key = h(prompt)
    if answer_key in answer_cache:
        yield answer_cache[answer_key]
        return

    answer_text = ""
    for token in llm.generate(prompt, max_tokens=MAX_NEW_TOKENS, streaming=True):
        answer_text += token
        yield token

    answer_cache[answer_key] = answer_text
    save_cache(ANSWER_CACHE, answer_cache)

# -------------------------
# Request model
# -------------------------
class ChatRequest(BaseModel):
    query: str

# -------------------------
# Chat endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query.strip()
    if not query:
        return {"error": "Empty query"}

    context, sources = retrieve_raw_context(query)

    # Improved prompt template for nice answers
    if not context:
        prompt = (
            "You are Maestro Chat Bot (offline). "
            "The corpus does not contain relevant information. "
            "Kindly answer: I don't know."
        )
        return StreamingResponse(stream_answer(prompt), media_type="text/plain")

    prompt = (
        "You are Maestro Chat Bot (offline). Use ONLY the context below to answer.\n"
        "Rules:\n"
        "- If answer is missing from context, say: I don't know.\n"
        "- Explain clearly using 3 to 8 sentences.\n"
        "- Add short example if applicable.\n"
        "- Don't hallucinate.\n"
        "- Be helpful and professional.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "ANSWER:\n"
    )

    return StreamingResponse(stream_answer(prompt), media_type="text/plain")
