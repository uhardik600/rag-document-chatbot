# 🤖 RAG-Based Intelligent Document Chatbot

An AI-powered document chatbot that lets you upload any PDF or TXT file and ask questions about it. Built with **Flask**, **LangChain**, **FAISS**, and **OpenAI GPT**, using a full Retrieval-Augmented Generation (RAG) pipeline.

---

## ✨ Features

- 📄 **Upload PDF or TXT** documents for instant indexing
- 🔍 **Semantic search** via FAISS vector similarity
- 🧠 **Context-aware responses** using GPT-3.5-turbo
- 💬 **Multi-turn chat** with Redis-backed session history
- ⚡ **Chunked ingestion** with overlap for better retrieval accuracy
- 🔄 **Graceful Redis fallback** — works even without Redis

---

## 🏗️ Architecture

```
User uploads PDF/TXT
        │
        ▼
  Document Loader (PyPDF / TextLoader)
        │
        ▼
  Text Splitter (RecursiveCharacterTextSplitter)
        │
        ▼
  OpenAI Embeddings (text-embedding-3-small)
        │
        ▼
  FAISS Vector Index (stored per session)
        │
        ▼
  User asks a question
        │
        ├──► FAISS Similarity Search (Top-K chunks)
        │
        ├──► Redis Chat History
        │
        └──► GPT-3.5-turbo → Answer
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/uhardik600/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Create virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 5. Run the server

```bash
python main.py
```

Server runs at `http://localhost:5000`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/upload` | Upload a PDF or TXT document |
| `POST` | `/chat` | Ask a question about the document |
| `GET` | `/history/<session_id>` | Get full chat history |
| `DELETE` | `/session/<session_id>` | Clear session and document |

### Example: Upload a document

```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "session_id": "abc123-...",
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "message": "Document uploaded and indexed successfully."
}
```

### Example: Ask a question

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123-...", "question": "What is the main topic of this document?"}'
```

**Response:**
```json
{
  "session_id": "abc123-...",
  "question": "What is the main topic of this document?",
  "answer": "The document covers...",
  "context_chunks_used": 4
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Flask |
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS |
| Orchestration | LangChain |
| Session Cache | Redis |
| PDF Parsing | PyPDF |

---

## 📁 Project Structure

```
rag-document-chatbot/
├── main.py          # Flask app & API routes
├── ingestor.py      # Document loading & text chunking
├── embeddings.py    # Embedding generation & FAISS storage
├── retriever.py     # Vector similarity search
├── chatbot.py       # LLM response generation
├── cache.py         # Redis session management
├── requirements.txt
└── .env.example
```

---

## 📜 License

MIT License — feel free to use and modify.
