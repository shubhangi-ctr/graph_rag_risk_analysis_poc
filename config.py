"""
Configuration for Combined RAG (Graph RAG + Traditional RAG)
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Neo4j Configuration (Graph RAG)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ChromaDB Configuration (Traditional RAG)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "risk_rag_chunks")
GRAPH_CHROMA_COLLECTION = os.getenv("GRAPH_CHROMA_COLLECTION", "risk_graph_kg_chunks")
TOP_K = int(os.getenv("TOP_K", "6"))
GRAPH_CHROMA_TOP_K = int(os.getenv("GRAPH_CHROMA_TOP_K", "6"))

# Application Settings
APP_TITLE = "RAG Comparison - Graph vs Traditional"
APP_DESCRIPTION = "Compare Graph RAG (Neo4j) vs Traditional RAG (Vector DB) side-by-side"
