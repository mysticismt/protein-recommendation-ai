from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = PROJECT_ROOT / "Protein_RAG"
CORPUS_DIR = PROJECT_ROOT / "data" / "rag_corpus"
ARTIFACT_DIR = RAG_ROOT / "artifacts" / "rag"
CHROMA_DIR = ARTIFACT_DIR / "chroma"
BM25_PATH = ARTIFACT_DIR / "bm25.joblib"
CATALOG_PATH = CORPUS_DIR / "MANIFEST.json"
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", os.getenv("RAG_LLM_MODEL", "qwen3.5"))
CHROMA_COLLECTION = "protein_rag_v1"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
DENSE_TOP_K = int(os.getenv("RAG_DENSE_TOP_K", "8"))
LEXICAL_TOP_K = int(os.getenv("RAG_LEXICAL_TOP_K", "8"))
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "6"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "1"))
