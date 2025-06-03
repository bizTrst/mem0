import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from mem0.memory.main import Memory

# Setup logging and environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

# Environment configuration
def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

DB_CONFIG = {
    "host": get_env("POSTGRES_HOST", "postgres"),
    "port": int(get_env("POSTGRES_PORT", "5432")),
    "dbname": get_env("POSTGRES_DB", "postgres"),
    "user": get_env("POSTGRES_USER", "postgres"),
    "password": get_env("POSTGRES_PASSWORD", "postgres"),
    "collection_name": get_env("POSTGRES_COLLECTION_NAME", "memories"),
}

graph_config = {
    "url": get_env("NEO4J_URI", "bolt://neo4j:7687"),
    "username": get_env("NEO4J_USERNAME", "neo4j"),
    "password": get_env("NEO4J_PASSWORD", "mem0graph"),
}

llm_config = {
    "provider": "openai",
    "config": {
        "api_key": get_env("OPENAI_API_KEY"),
        "temperature": 0.2,
        "model": "gpt-4o",
    },
}

embedder_config = {
    "provider": "openai",
    "config": {
        "api_key": get_env("OPENAI_API_KEY"),
        "model": "text-embedding-3-small",
    },
}

DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {"provider": "pgvector", "config": DB_CONFIG},
    "graph_store": {"provider": "neo4j", "config": graph_config},
    "llm": llm_config,
    "embedder": embedder_config,
    "history_db_path": get_env("HISTORY_DB_PATH", "/app/history/history.db"),
}

MEMORY_INSTANCE = Memory.from_config(DEFAULT_CONFIG)

# FastAPI App
app = FastAPI(title="Mem0 REST APIs", description="Memory APIs for AI Agents.", version="1.0.0")

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class MemoryCreate(BaseModel):
    messages: List[Message]
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

# API Endpoints
@app.post("/configure")
def set_config(config: Dict[str, Any]):
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    return {"message": "Configuration set successfully"}

@app.post("/memories")
def add_memory(memory_create: MemoryCreate):
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {k: v for k, v in memory_create.model_dump().items() if k != "messages" and v is not None}
        messages = [m.model_dump() for m in memory_create.messages]
        return JSONResponse(content=MEMORY_INSTANCE.add(messages=messages, **params))
    except Exception as e:
        logging.exception("Add memory failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories")
def get_all_memories(user_id: Optional[str] = None, run_id: Optional[str] = None, agent_id: Optional[str] = None):
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        return MEMORY_INSTANCE.get_all(**{k: v for k, v in locals().items() if v is not None})
    except Exception as e:
        logging.exception("Get all memories failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{memory_id}")
def get_memory(memory_id: str):
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Get memory failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_memories(search_req: SearchRequest):
    try:
        kwargs = {k: v for k, v in search_req.model_dump().items() if k != "query" and v is not None}
        return MEMORY_INSTANCE.search(query=search_req.query, **kwargs)
    except Exception as e:
        logging.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memories/{memory_id}")
def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Update memory failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{memory_id}/history")
def memory_history(memory_id: str):
    try:
        return MEMORY_INSTANCE.history(memory_id)
    except Exception as e:
        logging.exception("Memory history failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}")
def delete_memory(memory_id: str):
    try:
        MEMORY_INSTANCE.delete(memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Delete memory failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories")
def delete_all_memories(user_id: Optional[str] = None, run_id: Optional[str] = None, agent_id: Optional[str] = None):
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        MEMORY_INSTANCE.delete_all(**{k: v for k, v in locals().items() if v is not None})
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Delete all memories failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_memory():
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return RedirectResponse(url="/docs")
