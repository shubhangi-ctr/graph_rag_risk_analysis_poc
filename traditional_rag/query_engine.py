"""query_engine.py
Traditional RAG Query Engine

- Retrieves top-k documents from ChromaDB
- Produces a grounded answer using the retrieved context
- Keeps the same UI-facing contract as the Graph RAG POC:
  query(question) -> (debug_query, results, answer)
"""
from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from openai import OpenAI

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .vector_store import ChromaVectorStore, normalize_product_key


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
        self.indexed_products: List[Dict[str, str]] = []

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
        self._refresh_product_catalog()
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
            {
                "question": "For Incubators, what are the highest residual risk hazards?",
                "description": "Product-specific retrieval from the matching product file",
            },
        ]

    def _refresh_product_catalog(self) -> None:
        try:
            self.indexed_products = self.vector_store.list_products()
        except Exception:
            self.indexed_products = []

    def _detect_product_filter(self, question: str) -> Dict[str, str] | None:
        """Infer product filter from question using indexed product metadata."""
        if not self.indexed_products:
            self._refresh_product_catalog()
        if not self.indexed_products:
            return None

        normalized_question = f" {normalize_product_key(question)} "
        if normalized_question.strip() == "":
            return None

        matches: List[Tuple[int, Dict[str, str]]] = []
        for product in self.indexed_products:
            product_key = (product.get("product_key") or "").strip()
            product_name = (product.get("product_name") or "").strip()
            aliases = {product_key, normalize_product_key(product_name)}

            for alias in aliases:
                if not alias:
                    continue
                pattern = rf"(^|\s){re.escape(alias)}(\s|$)"
                if re.search(pattern, normalized_question):
                    matches.append((len(alias), product))
                    break

        if not matches:
            # Fuzzy fallback for minor spelling differences in product names.
            q_tokens = [t for t in normalize_product_key(question).split(" ") if t]
            fuzzy_matches: List[Tuple[float, Dict[str, str]]] = []
            for product in self.indexed_products:
                alias = (product.get("product_key") or normalize_product_key(product.get("product_name", ""))).strip()
                p_tokens = [t for t in alias.split(" ") if t]
                if not p_tokens:
                    continue

                matched = 0
                for p_tok in p_tokens:
                    if p_tok in q_tokens:
                        matched += 1
                        continue
                    if any(SequenceMatcher(None, p_tok, q_tok).ratio() >= 0.84 for q_tok in q_tokens):
                        matched += 1

                coverage = matched / len(p_tokens)
                if coverage >= 0.75 and matched >= max(1, len(p_tokens) - 1):
                    fuzzy_matches.append((coverage, product))

            if not fuzzy_matches:
                return None

            fuzzy_matches.sort(key=lambda item: item[0], reverse=True)
            best_score = fuzzy_matches[0][0]
            top_fuzzy = [product for score, product in fuzzy_matches if score == best_score]
            top_keys = {p.get("product_key", "") for p in top_fuzzy}
            if len(top_keys) > 1:
                return None
            return top_fuzzy[0]

        matches.sort(key=lambda item: item[0], reverse=True)
        longest = matches[0][0]
        top_matches = [product for size, product in matches if size == longest]

        top_keys = {p.get("product_key", "") for p in top_matches}
        if len(top_keys) > 1:
            return None
        return top_matches[0]

    def _is_cross_product_query(self, question: str) -> bool:
        """Detect when user explicitly asks for results across products."""
        q = normalize_product_key(question)
        patterns = [
            r"\ball products\b",
            r"\bacross products\b",
            r"\bmultiple products\b",
            r"\beach product\b",
            r"\bper product\b",
            r"\bby product\b",
        ]
        return any(re.search(p, q) for p in patterns)

    def _retrieve_diverse_hits(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve hits with product diversity.
        This avoids generic questions collapsing into only one product.
        """
        expanded_k = max(top_k * 8, 48)
        raw_hits = self.vector_store.query(
            oai=self.client,
            query_text=question,
            embedding_model=config.OPENAI_EMBEDDING_MODEL,
            top_k=expanded_k,
            metadata_filter=None,
        )
        if not raw_hits:
            return []

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for hit in raw_hits:
            meta = hit.get("meta") or {}
            key = str(meta.get("product_key") or "__unknown__")
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(hit)

        order.sort(key=lambda k: (buckets[k][0].get("distance") if buckets[k] else 10**9))
        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        idx = 0
        while len(selected) < top_k and any(idx < len(buckets[k]) for k in order):
            for k in order:
                if idx < len(buckets[k]) and len(selected) < top_k:
                    candidate = buckets[k][idx]
                    cid = candidate.get("id")
                    if cid in selected_ids:
                        continue
                    selected.append(candidate)
                    selected_ids.add(cid)
            idx += 1

        # Backfill missing products with their best-matching chunk.
        seen_product_keys = {
            str((hit.get("meta") or {}).get("product_key") or "")
            for hit in selected
        }
        for product in self.indexed_products:
            if len(selected) >= top_k:
                break
            product_key = (product.get("product_key") or "").strip()
            if not product_key or product_key in seen_product_keys:
                continue
            product_hits = self.vector_store.query(
                oai=self.client,
                query_text=question,
                embedding_model=config.OPENAI_EMBEDDING_MODEL,
                top_k=1,
                metadata_filter={"product_key": product_key},
            )
            if not product_hits:
                continue
            candidate = product_hits[0]
            cid = candidate.get("id")
            if cid in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(cid)
            seen_product_keys.add(product_key)

        return selected[:top_k]

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for h in hits:
            m = h.get("meta") or {}
            blocks.append(
                "\n".join(
                    [
                        (
                            f"[source={m.get('source','')} product={m.get('product_name','')} "
                            f"doc_type={m.get('doc_type','')} id={m.get('id','')}]"
                        ),
                        h.get("text", ""),
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]], str]:
        """Run vector retrieval + grounded answering."""
        if not self.client:
            raise ValueError("QueryEngine not connected. Call connect() first.")

        cross_product = self._is_cross_product_query(question)
        product = None if cross_product else self._detect_product_filter(question)
        metadata_filter = None
        if product and product.get("product_key"):
            metadata_filter = {"product_key": product["product_key"]}

        if metadata_filter:
            hits = self.vector_store.query(
                oai=self.client,
                query_text=question,
                embedding_model=config.OPENAI_EMBEDDING_MODEL,
                top_k=config.TOP_K,
                metadata_filter=metadata_filter,
            )
            retrieval_mode = "filtered"
        elif len(self.indexed_products) > 1:
            hits = self._retrieve_diverse_hits(question, top_k=config.TOP_K)
            retrieval_mode = "diverse"
        else:
            hits = self.vector_store.query(
                oai=self.client,
                query_text=question,
                embedding_model=config.OPENAI_EMBEDDING_MODEL,
                top_k=config.TOP_K,
                metadata_filter=None,
            )
            retrieval_mode = "plain"

        debug_query = (
            f"VECTOR_SEARCH(collection={config.CHROMA_COLLECTION}, top_k={config.TOP_K}, mode={retrieval_mode}"
        )
        if metadata_filter:
            debug_query += f", filter={metadata_filter}"
        debug_query += ")"
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
