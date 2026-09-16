"""Build the Protein RAG vector and lexical indexes."""
from Protein_RAG.rag.index import build_index

if __name__ == "__main__":
    build_index(include_optional=False)
