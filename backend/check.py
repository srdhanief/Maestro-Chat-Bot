import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

FAISS_INDEX_PATH = r'C:\Users\dsheikdawood\Documents\Hackathon\Maestro Chat Bot with RAG\faiss_index\index.faiss'   # your path
DOCS_PATH = r'C:\Users\dsheikdawood\Documents\Hackathon\Maestro Chat Bot with RAG\faiss_index\docs.pkl'
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# load resources
index = faiss.read_index(FAISS_INDEX_PATH)
docs = pickle.load(open(DOCS_PATH, "rb"))
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("FAISS index info:")
print("  ntotal:", index.ntotal)
try:
    print("  d (dim):", index.d)
except:
    print("  index has no attribute d (older FAISS). Try index.get_dimension()")
    print("  dimension:", index.get_dimension())

# embed a sample doc and the same doc from docs list to verify shape
sample_doc = docs[0]
doc_emb = embedder.encode([sample_doc], convert_to_numpy=True)
print("sample_doc emb shape:", doc_emb.shape, doc_emb.dtype)

# run a quick search using the raw embedding (NOT normalized)
q = doc_emb.astype("float32")
D, I = index.search(q, 3)
print("raw search distances:", D)
print("raw search indices:", I)
