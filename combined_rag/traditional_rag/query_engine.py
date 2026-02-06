"""query_engine.py
Traditional RAG Query Engine

- Retrieves top-k documents from ChromaDB
- Produces a grounded answer using the retrieved context
- Keeps the same UI-facing contract as the Graph RAG POC:
  query(question) -> (debug_query, results, answer)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .vector_store import ChromaVectorStore


class QueryEngine:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        openai_api_key: str | None = None,
    ):
        self.vector_store = vector_store
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.client: OpenAI | None = None
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "rag_prompt.txt")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return """Use ONLY the provided context to answer.\nQUESTION: {question}\nCONTEXT: {context}"""

    def connect(self) -> bool:
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = OpenAI(api_key=self.openai_api_key)
        return True

    def close(self):
        # Nothing to close for Chroma PersistentClient in this simple setup
        pass

    def get_sample_queries(self) -> List[Dict[str, str]]:
        """Return sample queries for the UI - mirrors the Graph RAG POC samples."""
        return [
            {
                "question": "Which controls mitigate multiple high-severity hazards?",
                "description": "Find shared controls for severe hazards (Control reuse)",
            },
            {
                "question": "What hazards remain above ALAP after all controls are applied?",
                "description": "Find residual risks still at Medium or High level",
            },
            {
                "question": "Which hazards occur during maintenance or service phases?",
                "description": "Lifecycle-specific hazard analysis",
            },
            {
                "question": "Show me the hazard-cause-consequence chain for thermal hazards",
                "description": "Full risk chain exploration",
            },
            {
                "question": "What controls reference the IEC 61010-1 safety standard?",
                "description": "Standard-specific control search",
            },
            {
                "question": "Which hazards affect the User actor?",
                "description": "Actor-based hazard filtering",
            },
            {
                "question": "What is the average risk reduction achieved by each control?",
                "description": "Control effectiveness analysis (may require multiple evidence rows)",
            },
        ]

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for h in hits:
            m = h.get("meta") or {}
            blocks.append(
                "\n".join(
                    [
                        f"[source={m.get('source','')} doc_type={m.get('doc_type','')} id={m.get('id','')}]" ,
                        h.get("text", ""),
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]], str]:
        """Run vector retrieval + grounded answering."""
        if not self.client:
            raise ValueError("QueryEngine not connected. Call connect() first.")

        hits = self.vector_store.query(
            oai=self.client,
            query_text=question,
            embedding_model=config.OPENAI_EMBEDDING_MODEL,
            top_k=config.TOP_K,
        )

        debug_query = f"VECTOR_SEARCH(collection={config.CHROMA_COLLECTION}, top_k={config.TOP_K})"
        if not hits:
            return debug_query, [], "No relevant evidence found in the vector store. Try ingesting data or rephrasing the question."

        context = self._format_context(hits)
        prompt = self.prompt_template.format(question=question, context=context)

        resp = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a risk analysis assistant that answers ONLY from provided context and always cites sources.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )

        answer = (resp.choices[0].message.content or "").strip()
        return debug_query, hits, answer
