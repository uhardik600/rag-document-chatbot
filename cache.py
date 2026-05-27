import redis
import json
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 3600))  # 1 hour default

HISTORY_PREFIX = "chat_history:"

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=2
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False
    print("[WARNING] Redis unavailable. Chat history will not persist.")

# In-memory fallback when Redis is unavailable
_memory_store = {}


def get_chat_history(session_id: str) -> list:
    key = f"{HISTORY_PREFIX}{session_id}"
    if REDIS_AVAILABLE:
        try:
            data = redis_client.get(key)
            return json.loads(data) if data else []
        except Exception:
            pass
    return _memory_store.get(key, [])


def save_chat_history(session_id: str, history: list):
    key = f"{HISTORY_PREFIX}{session_id}"
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(key, SESSION_TTL, json.dumps(history))
            return
        except Exception:
            pass
    _memory_store[key] = history


def clear_chat_history(session_id: str):
    key = f"{HISTORY_PREFIX}{session_id}"
    if REDIS_AVAILABLE:
        try:
            redis_client.delete(key)
        except Exception:
            pass
    _memory_store.pop(key, None)
