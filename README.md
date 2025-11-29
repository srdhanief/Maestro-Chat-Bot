# 🤖 Maestro Chat Bot — Offline Custom LLM with RAG

Maestro Chat Bot is a **completely offline, private, and domain-aware custom LLM system** built for exploring and answering questions about **Maestro workflows, configuration, and supply chain validation** using **Retrieval-Augmented Generation (RAG).**

It provides accurate answers by retrieving relevant documentation from a FAISS index and generating responses through GPT4All running on CPU.

---

## 🚀 Features

- 🧠 **Custom Local LLM** → GPT4All (Falcon Q4 quantized)
- 📚 **RAG Pipeline** → FAISS vector search with Sentence-Transformers embeddings (384d)
- 🔄 **Fully Offline Inference** → No internet required once running
- 🚄 **Streaming Responses** → Real-time token streaming from backend to UI
- 🎨 **Modern UI** → React.JS chat interface
- ⚡ **Fast Backend APIs** → Built with FastAPI
- 💾 **Persistent Caching** → Embeddings, retrievals, and answers cached locally for performance

---

## 🏗 Tech Stack

| Component | Technology |
|---------|-----------|
| **Frontend** | React.JS |
| **Backend API** | FastAPI |
| **Vector Store** | FAISS index search |
| **Embeddings** | Sentence-Transformers `all-MiniLM-L6-v2` (384 dimensions) |
| **LLM Model** | GPT4All Falcon Q4 (`.gguf`) |
| **Device** | CPU-only inference |

---

## 🛠 Installation & Setup

### 1. Clone the Repository
```sh
git clone <repo-link>
cd Maestro-Chat-Bot
2. Install backend dependencies
pip install fastapi uvicorn sentence-transformers gpt4all numpy faiss-cpu

3. Run the API server
uvicorn maestro_chat_improved:app --reload

4. Install and start the frontend
cd frontend
npm install
npm start2. Install backend dependencies
pip install fastapi uvicorn sentence-transformers gpt4all numpy faiss-cpu

3. Run the API server
uvicorn maestro_chat_improved:app --reload

4. Install and start the frontend
cd frontend
npm install
npm start
```

📖 How It Works (RAG Flow)

User asks a question from the chat UI

Backend generates a cached normalized query embedding

FAISS retrieves the most relevant docs/chunks

Context is injected into a strict grounded prompt

GPT4All generates answer (streamed + cached)

Frontend renders the answer in real-time

📌 Future Enhancements (Optional)

🔎 Cross-Encoder re-ranking layer

🧬 Replace Falcon with Llama or Mistral local model

🔐 Add user authentication

☁ Deploy using Docker or Cloud Run

🧾 Show retrieved sources in UI
