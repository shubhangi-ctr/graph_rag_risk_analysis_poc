"""vector_store.py
ChromaDB-backed vector store for Traditional RAG.

We store *text documents* representing:
- Hazards (with causes, consequences, controls, actors, lifecycle, category)
- Controls (with linked standards + document sections + hazards mitigated)
- Standards / DocumentSections / Lifecycle phases (lightweight summaries)

All docs have metadata for UI citations.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

from excel_parser import ParsedData


def _sha_id(prefix: str, text: str) -> str:
    h = hashlib.sha1((prefix + "|" + text).encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}:{h[:12]}"


class ChromaVectorStore:
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
        assert len(documents) == len(metadatas), "documents and metadatas must have same length"

        added = 0
        for i in range(0, len(documents), batch_size):
            docs = documents[i : i + batch_size]
            metas = metadatas[i : i + batch_size]

            ids = []
            for d, m in zip(docs, metas):
                # Deterministic ID to avoid duplicates across re-ingests
                prefix = f"{m.get('source','')}|{m.get('doc_type','')}|{m.get('id','')}|{m.get('hazard_id','')}|{m.get('control_id','')}"
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


def build_rag_documents(data: ParsedData, source_name: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, int]]:
    """Convert ParsedData graph objects to textual documents for vector RAG."""
    # Index controls by id for easy lookup
    controls_by_id = {c.id: c for c in data.controls}
    standards_by_id = {s.id: s for s in data.standards}
    docs_by_id = {d.id: d for d in data.document_sections}
    hazard_by_id = {h.id: h for h in data.hazards}

    # Relationships
    hazard_controls = {}
    for link in data.hazard_control_links:
        hazard_controls.setdefault(link.hazard_id, []).append(link)

    hazard_causes = {}
    for ca in data.causes:
        hazard_causes.setdefault(ca.hazard_id, []).append(ca)

    hazard_conseq = {}
    for co in data.consequences:
        hazard_conseq.setdefault(co.hazard_id, []).append(co)

    hazard_actors = {}
    for ha in data.hazard_actor_links:
        hazard_actors.setdefault(ha.hazard_id, []).append(ha.actor_name)

    hazard_categories = {}
    for hc in data.hazard_category_links:
        hazard_categories.setdefault(hc.hazard_id, []).append(hc.category_id)

    hazard_lifecycle = {}
    for hl in data.hazard_lifecycle_links:
        hazard_lifecycle.setdefault(hl.hazard_id, []).append(hl.lifecycle_phase)

    control_standards = {}
    for cs in data.control_standard_links:
        control_standards.setdefault(cs.control_id, []).append(cs.standard_id)

    control_docs = {}
    for cd in data.control_document_links:
        control_docs.setdefault(cd.control_id, []).append(cd.document_id)

    # Reverse mapping: which hazards a control mitigates
    control_hazards = {}
    for hid, links in hazard_controls.items():
        for link in links:
            control_hazards.setdefault(link.control_id, set()).add(hid)

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    # --- Hazard documents (primary evidence unit) ---
    for h in data.hazards:
        causes = hazard_causes.get(h.id, [])
        consequences = hazard_conseq.get(h.id, [])
        links = hazard_controls.get(h.id, [])

        cat_ids = hazard_categories.get(h.id, [])
        lifecycle = hazard_lifecycle.get(h.id, [])
        actors = hazard_actors.get(h.id, [])

        # Controls expanded
        control_lines = []
        for l in links:
            c = controls_by_id.get(l.control_id)
            if not c:
                continue
            reductions = []
            if l.p_reduction is not None:
                reductions.append(f"ΔP={l.p_reduction}")
            if l.s_reduction is not None:
                reductions.append(f"ΔS={l.s_reduction}")
            red = (" (" + ", ".join(reductions) + ")") if reductions else ""
            control_lines.append(f"- {c.id}: {c.description}{red}")

        text = "\n".join(
            [
                f"Hazard {h.id}: {h.name}",
                f"Type: {h.h_type}" if h.h_type else "",
                f"Source/Section: {h.q_source}" if h.q_source else "",
                f"Initial Risk: P={h.p_init} S={h.s_init} R={h.r_init}" if (h.p_init or h.s_init or h.r_init) else "",
                f"Final Risk:   P={h.p_final} S={h.s_final} R={h.r_final}" if (h.p_final or h.s_final or h.r_final) else "",
                f"Actors affected: {', '.join(sorted(set(actors)))}" if actors else "",
                f"Lifecycle phases: {', '.join(sorted(set(lifecycle)))}" if lifecycle else "",
                f"Categories: {', '.join(cat_ids)}" if cat_ids else "",
                "Causes:\n" + "\n".join([f"- {c.id}: {c.description}" for c in causes]) if causes else "",
                "Consequences:\n" + "\n".join([f"- {c.id}: {c.description}" for c in consequences]) if consequences else "",
                "Controls (mitigations):\n" + "\n".join(control_lines) if control_lines else "",
            ]
        )
        text = "\n".join([line for line in text.splitlines() if line.strip()])

        documents.append(text)
        metadatas.append(
            {
                "source": source_name,
                "doc_type": "hazard",
                "id": h.id,
                "hazard_id": h.id,
                "hazard_name": h.name,
            }
        )

    # --- Control documents (helpful for control-centric queries) ---
    for c in data.controls:
        stds = [standards_by_id[sid].name if sid in standards_by_id else sid for sid in control_standards.get(c.id, [])]
        docsecs = [docs_by_id[did].name if did in docs_by_id else did for did in control_docs.get(c.id, [])]
        mitigates = sorted(list(control_hazards.get(c.id, set())))
        hazards_preview = []
        for hid in mitigates[:10]:
            hh = hazard_by_id.get(hid)
            hazards_preview.append(f"- {hid}: {hh.name}" if hh else f"- {hid}")
        if len(mitigates) > 10:
            hazards_preview.append(f"... and {len(mitigates)-10} more")

        text = "\n".join(
            [
                f"Control {c.id}: {c.description}",
                f"Implementation reference: {c.implementation_reference}" if c.implementation_reference else "",
                f"Standards referenced: {', '.join(stds)}" if stds else "",
                f"Document sections: {', '.join(docsecs)}" if docsecs else "",
                f"Mitigates hazards ({len(mitigates)}):" if mitigates else "",
                "\n".join(hazards_preview) if hazards_preview else "",
            ]
        )
        text = "\n".join([line for line in text.splitlines() if line.strip()])

        documents.append(text)
        metadatas.append(
            {
                "source": source_name,
                "doc_type": "control",
                "id": c.id,
                "control_id": c.id,
            }
        )

    # --- Standards (light) ---
    for s in data.standards:
        text = f"Standard {s.id}: {s.name}"
        documents.append(text)
        metadatas.append({"source": source_name, "doc_type": "standard", "id": s.id})

    # --- Document sections (light) ---
    for d in data.document_sections:
        text = f"DocumentSection {d.id}: {d.name} (type: {d.document_type})" if d.document_type else f"DocumentSection {d.id}: {d.name}"
        documents.append(text)
        metadatas.append({"source": source_name, "doc_type": "document_section", "id": d.id})

    # --- Lifecycle phases (light) ---
    for l in data.lifecycle_phases:
        text = f"LifecyclePhase: {l.name}"
        documents.append(text)
        metadatas.append({"source": source_name, "doc_type": "lifecycle_phase", "id": l.name})

    stats = {
        "hazards": len(data.hazards),
        "controls": len(data.controls),
        "causes": len(data.causes),
        "consequences": len(data.consequences),
        "standards": len(data.standards),
        "document_sections": len(data.document_sections),
        "lifecycle_phases": len(data.lifecycle_phases),
        "vector_docs": len(documents),
    }
    return documents, metadatas, stats
