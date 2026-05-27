from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid

from ingestor import ingest_document
from retriever import search_similar_chunks
from chatbot import generate_response
from cache import get_chat_history, save_chat_history, clear_chat_history

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB max upload
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return jsonify({"status": "ok", "service": "RAG Document Chatbot"})


@app.post("/upload")
def upload_document():
    """
    Upload a PDF or TXT document. Returns a session_id for follow-up queries.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Only PDF and TXT are supported."}), 400

    session_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(filepath)

    try:
        chunk_count = ingest_document(filepath, session_id)
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

    return jsonify({
        "session_id": session_id,
        "filename": filename,
        "chunks_indexed": chunk_count,
        "message": "Document uploaded and indexed successfully. Use session_id to chat."
    }), 201


@app.post("/chat")
def chat():
    """
    Ask a question about the uploaded document.
    Requires session_id from /upload response.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or not question:
        return jsonify({"error": "Both session_id and question are required."}), 400

    # Retrieve relevant chunks from FAISS
    relevant_chunks = search_similar_chunks(question, session_id)
    if not relevant_chunks:
        return jsonify({"error": "No document found for this session. Please upload first."}), 404

    # Get chat history from Redis
    history = get_chat_history(session_id)

    # Generate LLM response
    answer = generate_response(question, relevant_chunks, history)

    # Save updated history to Redis
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    save_chat_history(session_id, history)

    return jsonify({
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "context_chunks_used": len(relevant_chunks)
    })


@app.get("/history/<session_id>")
def get_history(session_id):
    """
    Retrieve full chat history for a session.
    """
    history = get_chat_history(session_id)
    if not history:
        return jsonify({"error": "No history found for this session."}), 404
    return jsonify({"session_id": session_id, "history": history})


@app.delete("/session/<session_id>")
def clear_session(session_id):
    """
    Clear chat history and remove indexed document for a session.
    """
    clear_chat_history(session_id)
    # Remove uploaded files for this session
    for f in os.listdir(UPLOAD_FOLDER):
        if f.startswith(session_id):
            os.remove(os.path.join(UPLOAD_FOLDER, f))
    return jsonify({"message": f"Session {session_id} cleared successfully."})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
