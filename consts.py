import os
from dotenv import load_dotenv

load_dotenv()

# LLM related variables
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME") or "qwen2"
LLM_MODEL_TEMP = int(os.environ.get("LLM_MODEL_TEMP") or 0.7)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or "http://10.0.0.4:11434"

# Embeddings related variables
EMBEDDING_MODEL_NAME = (
    os.environ.get("EMBEDDING_MODEL_NAME") or "sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB related variables
VECTOR_DB_FILE_PATH = "faiss_index"
VECTOR_RETRIVAL_TEMP = 0.7

# DB related variable
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")
DB_PORT = os.environ.get("DB_PORT") or 3306
