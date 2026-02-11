"""graph_rag/chroma_store.py
ChromaDB-backed storage for graph knowledge chunks sourced from Neo4j.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from neo4j import GraphDatabase
from openai import OpenAI


def _sha_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1((prefix + "|" + text).encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}:{digest[:12]}"


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma metadata values must be scalar bool/int/float/str.
    Remove None values and stringify unsupported types.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


class GraphChromaStore:
    """Simple Chroma wrapper for graph chunks."""

    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)

    @staticmethod
    def embed_texts(oai: OpenAI, texts: List[str], model: str) -> List[List[float]]:
        resp = oai.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def add_documents(
        self,
        oai: OpenAI,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embedding_model: str,
        batch_size: int = 64,
    ) -> int:
        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length")

        added = 0
        for i in range(0, len(documents), batch_size):
            docs = documents[i : i + batch_size]
            metas = [_sanitize_metadata(m) for m in metadatas[i : i + batch_size]]
            ids: List[str] = []

            for d, m in zip(docs, metas):
                prefix = f"{m.get('source','')}|{m.get('doc_type','')}|{m.get('id','')}"
                ids.append(_sha_id(prefix, d))

            embs = self.embed_texts(oai, docs, model=embedding_model)
            self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            added += len(docs)

        return added

    def query(
        self,
        oai: OpenAI,
        query_text: str,
        embedding_model: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        q_emb = self.embed_texts(oai, [query_text], model=embedding_model)[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        for i in range(len(res["ids"][0])):
            hits.append(
                {
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "meta": res["metadatas"][0][i],
                    "distance": res["distances"][0][i],
                }
            )
        return hits


def _build_hazard_documents(session, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    query = """
    MATCH (h:Hazard)
    OPTIONAL MATCH (h)-[:HAS_CAUSE]->(ca:Cause)
    WITH h, collect(DISTINCT ca.description) AS causes
    OPTIONAL MATCH (h)-[:HAS_CONSEQUENCE]->(co:Consequence)
    WITH h, causes, collect(DISTINCT co.description) AS consequences
    OPTIONAL MATCH (h)-[m:MITIGATED_BY]->(ctrl:Control)
    WITH h, causes, consequences,
         collect(DISTINCT {
             control_id: ctrl.id,
             control_description: ctrl.description,
             p_reduction: m.p_reduction,
             s_reduction: m.s_reduction
         }) AS mitigations
    OPTIONAL MATCH (h)-[:AFFECTS]->(a:Actor)
    WITH h, causes, consequences, mitigations, collect(DISTINCT a.name) AS actors
    OPTIONAL MATCH (h)-[:OCCURS_DURING]->(l:LifecyclePhase)
    WITH h, causes, consequences, mitigations, actors, collect(DISTINCT l.name) AS lifecycle_phases
    OPTIONAL MATCH (hc:HazardCategory)-[:CONTAINS]->(h)
    RETURN
        h.id AS hazard_id,
        h.name AS hazard_name,
        h.h_type AS hazard_type,
        h.q_source AS source_section,
        h.p_init AS p_init,
        h.s_init AS s_init,
        h.r_init AS r_init,
        h.p_final AS p_final,
        h.s_final AS s_final,
        h.r_final AS r_final,
        [x IN causes WHERE x IS NOT NULL] AS causes,
        [x IN consequences WHERE x IS NOT NULL] AS consequences,
        [x IN actors WHERE x IS NOT NULL] AS actors,
        [x IN lifecycle_phases WHERE x IS NOT NULL] AS lifecycle_phases,
        [x IN collect(DISTINCT hc.name) WHERE x IS NOT NULL] AS categories,
        [x IN mitigations WHERE x.control_id IS NOT NULL] AS mitigations
    ORDER BY hazard_id
    """

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for row in session.run(query):
        mitigations = row["mitigations"] or []
        control_lines: List[str] = []
        for m in mitigations:
            reductions: List[str] = []
            if m.get("p_reduction") is not None:
                reductions.append(f"Dp={m['p_reduction']}")
            if m.get("s_reduction") is not None:
                reductions.append(f"Ds={m['s_reduction']}")
            reduction_txt = f" ({', '.join(reductions)})" if reductions else ""
            control_lines.append(
                f"- {m.get('control_id')}: {m.get('control_description', '')}{reduction_txt}"
            )

        lines = [
            f"Hazard {row.get('hazard_id')}: {row.get('hazard_name') or 'Unknown'}",
            f"Type: {row.get('hazard_type')}" if row.get("hazard_type") else "",
            f"Source/Section: {row.get('source_section')}" if row.get("source_section") else "",
            (
                f"Initial Risk: P={row.get('p_init')} S={row.get('s_init')} R={row.get('r_init')}"
                if (row.get("p_init") is not None or row.get("s_init") is not None or row.get("r_init"))
                else ""
            ),
            (
                f"Final Risk: P={row.get('p_final')} S={row.get('s_final')} R={row.get('r_final')}"
                if (row.get("p_final") is not None or row.get("s_final") is not None or row.get("r_final"))
                else ""
            ),
            f"Categories: {', '.join(row.get('categories') or [])}" if row.get("categories") else "",
            f"Actors: {', '.join(row.get('actors') or [])}" if row.get("actors") else "",
            f"Lifecycle phases: {', '.join(row.get('lifecycle_phases') or [])}" if row.get("lifecycle_phases") else "",
            "Causes:\n" + "\n".join([f"- {x}" for x in (row.get("causes") or [])]) if row.get("causes") else "",
            "Consequences:\n" + "\n".join([f"- {x}" for x in (row.get("consequences") or [])]) if row.get("consequences") else "",
            "Controls (mitigations):\n" + "\n".join(control_lines) if control_lines else "",
        ]
        text = "\n".join([line for line in lines if line.strip()])
        documents.append(text)
        metadatas.append(
            {
                "source": source_name,
                "doc_type": "graph_hazard_chunk",
                "id": row.get("hazard_id"),
                "hazard_id": row.get("hazard_id"),
                "hazard_name": row.get("hazard_name"),
            }
        )

    return documents, metadatas


def _build_control_documents(session, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    query = """
    MATCH (c:Control)
    OPTIONAL MATCH (h:Hazard)-[:MITIGATED_BY]->(c)
    WITH c, collect(DISTINCT h.id) AS hazard_ids
    OPTIONAL MATCH (c)-[:REFERENCES]->(s:Standard)
    WITH c, hazard_ids, collect(DISTINCT s.id) AS standard_ids
    OPTIONAL MATCH (c)-[:DOCUMENTED_IN]->(d:DocumentSection)
    RETURN
        c.id AS control_id,
        c.description AS control_description,
        c.implementation_reference AS implementation_reference,
        [x IN hazard_ids WHERE x IS NOT NULL] AS hazard_ids,
        [x IN standard_ids WHERE x IS NOT NULL] AS standard_ids,
        [x IN collect(DISTINCT d.id) WHERE x IS NOT NULL] AS document_ids
    ORDER BY control_id
    """

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for row in session.run(query):
        lines = [
            f"Control {row.get('control_id')}: {row.get('control_description') or ''}",
            (
                f"Implementation reference: {row.get('implementation_reference')}"
                if row.get("implementation_reference")
                else ""
            ),
            f"Mitigates hazards: {', '.join(row.get('hazard_ids') or [])}" if row.get("hazard_ids") else "",
            f"References standards: {', '.join(row.get('standard_ids') or [])}" if row.get("standard_ids") else "",
            f"Document sections: {', '.join(row.get('document_ids') or [])}" if row.get("document_ids") else "",
        ]
        text = "\n".join([line for line in lines if line.strip()])
        documents.append(text)
        metadatas.append(
            {
                "source": source_name,
                "doc_type": "graph_control_chunk",
                "id": row.get("control_id"),
                "control_id": row.get("control_id"),
            }
        )

    return documents, metadatas


def build_graph_documents_from_neo4j(uri: str, user: str, password: str, source_name: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """Read graph knowledge from Neo4j and convert it to Chroma-ready text chunks."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            hazard_docs, hazard_meta = _build_hazard_documents(session, source_name=source_name)
            control_docs, control_meta = _build_control_documents(session, source_name=source_name)

        documents = hazard_docs + control_docs
        metadatas = hazard_meta + control_meta
        stats = {
            "graph_hazard_chunks": len(hazard_docs),
            "graph_control_chunks": len(control_docs),
            "graph_total_chunks": len(documents),
        }
        return documents, metadatas, stats
    finally:
        driver.close()


def sync_graph_knowledge_from_neo4j(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    openai_api_key: str,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    source_name: str = "neo4j_graph",
    reset: bool = True,
) -> Dict[str, int]:
    """Index Neo4j graph knowledge into a Chroma collection."""
    if not openai_api_key:
        raise ValueError("OpenAI API key is required to index graph chunks in Chroma.")

    documents, metadatas, stats = build_graph_documents_from_neo4j(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        source_name=source_name,
    )

    store = GraphChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    if reset:
        store.reset()

    if not documents:
        return {**stats, "graph_indexed_chunks": 0}

    oai = OpenAI(api_key=openai_api_key)
    added = store.add_documents(
        oai=oai,
        documents=documents,
        metadatas=metadatas,
        embedding_model=embedding_model,
    )

    return {**stats, "graph_indexed_chunks": added}
