"""
Query Engine Module
Converts natural language questions to Cypher queries using LLM and executes them.
Supports full scope requirements including all node types and relationships.
"""
import os
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from openai import OpenAI
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from graph_rag.chroma_store import GraphChromaStore


class QueryEngine:
    """Handles natural language to Cypher conversion and query execution."""
    
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        openai_api_key: str = None,
        chroma_persist_dir: str = None,
        graph_collection_name: str = None,
        graph_top_k: int = None,
    ):
        self.neo4j_uri = neo4j_uri or config.NEO4J_URI
        self.neo4j_user = neo4j_user or config.NEO4J_USER
        self.neo4j_password = neo4j_password or config.NEO4J_PASSWORD
        self.openai_api_key = openai_api_key or config.OPENAI_API_KEY
        self.chroma_persist_dir = chroma_persist_dir or config.CHROMA_PERSIST_DIR
        self.graph_collection_name = graph_collection_name or config.GRAPH_CHROMA_COLLECTION
        self.graph_top_k = graph_top_k or config.GRAPH_CHROMA_TOP_K
        
        self.driver = None
        self.client = None
        self.graph_store = None
        self.prompt_template = self._load_prompt_template()
        self.indexed_products: List[Dict[str, str]] = []

    @staticmethod
    def _normalize_product_key(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
        return re.sub(r"\s+", " ", cleaned)
    
    def _load_prompt_template(self) -> str:
        """Load the Cypher prompt template."""
        prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "cypher_prompt.txt")
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return "You are a Cypher expert. Question: {question}. Return ONLY the Cypher query."
    
    def connect(self):
        """Establish connections to Neo4j, OpenAI, and Graph Chroma store."""
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )

        self.graph_store = GraphChromaStore(
            persist_dir=self.chroma_persist_dir,
            collection_name=self.graph_collection_name,
        )

        if self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        self._refresh_product_catalog()
        
        return True
    
    def close(self):
        """Close connections."""
        if self.driver:
            self.driver.close()
    
    def _refresh_product_catalog(self) -> None:
        try:
            self.indexed_products = self.graph_store.list_products() if self.graph_store else []
        except Exception:
            self.indexed_products = []

    def _detect_product_filter(self, question: str) -> Dict[str, str] | None:
        if not self.indexed_products:
            self._refresh_product_catalog()
        if not self.indexed_products:
            return None

        normalized_question = f" {self._normalize_product_key(question)} "
        if normalized_question.strip() == "":
            return None

        matches: List[Tuple[int, Dict[str, str]]] = []
        for product in self.indexed_products:
            product_key = (product.get("product_key") or "").strip()
            product_name = (product.get("product_name") or "").strip()
            aliases = {product_key, self._normalize_product_key(product_name)}

            for alias in aliases:
                if not alias:
                    continue
                pattern = rf"(^|\s){re.escape(alias)}(\s|$)"
                if re.search(pattern, normalized_question):
                    matches.append((len(alias), product))
                    break

        if not matches:
            q_tokens = [t for t in self._normalize_product_key(question).split(" ") if t]
            fuzzy_matches: List[Tuple[float, Dict[str, str]]] = []
            for product in self.indexed_products:
                alias = (product.get("product_key") or self._normalize_product_key(product.get("product_name", ""))).strip()
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
        q = self._normalize_product_key(question)
        patterns = [
            r"\ball products\b",
            r"\bacross products\b",
            r"\bmultiple products\b",
            r"\beach product\b",
            r"\bper product\b",
            r"\bby product\b",
        ]
        return any(re.search(p, q) for p in patterns)

    def _retrieve_diverse_graph_chunks(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve chunks with product diversity for cross-product questions.
        This avoids top-k collapsing onto only one product.
        """
        expanded_k = max(top_k * 4, 24)
        raw_hits = self.graph_store.query(
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

        # Sort products by best distance first, then round-robin selection.
        order.sort(key=lambda k: (buckets[k][0].get("distance") if buckets[k] else 10**9))
        selected: List[Dict[str, Any]] = []
        idx = 0
        while len(selected) < top_k and any(idx < len(buckets[k]) for k in order):
            for k in order:
                if idx < len(buckets[k]) and len(selected) < top_k:
                    selected.append(buckets[k][idx])
            idx += 1
        return selected

    def retrieve_graph_chunks(
        self,
        question: str,
        top_k: int = None,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant graph knowledge chunks from Chroma."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        if not self.graph_store:
            raise ValueError("Graph Chroma store not initialized. Call connect() first.")

        return self.graph_store.query(
            oai=self.client,
            query_text=question,
            embedding_model=config.OPENAI_EMBEDDING_MODEL,
            top_k=top_k or self.graph_top_k,
            metadata_filter=metadata_filter,
        )

    def _format_graph_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Serialize retrieved graph chunks for Cypher prompt conditioning."""
        if not chunks:
            return ""

        blocks: List[str] = []
        for i, hit in enumerate(chunks, start=1):
            meta = hit.get("meta") or {}
            text = (hit.get("text") or "").strip()
            if len(text) > 900:
                text = text[:900] + "..."
            blocks.append(
                "\n".join(
                    [
                        f"[chunk={i}] source={meta.get('source','')} doc_type={meta.get('doc_type','')} id={meta.get('id','')} distance={hit.get('distance')}",
                        f"product={meta.get('product_name','')}",
                        text,
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def generate_cypher(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]] = None,
        product_filter: Dict[str, str] | None = None,
        cross_product: bool = False,
    ) -> str:
        """Generate Cypher query from natural language + retrieved graph context."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        prompt = self.prompt_template.format(question=question)
        if context_chunks:
            graph_context = self._format_graph_context(context_chunks)
            prompt += (
                "\n\n## Retrieved Knowledge Graph Context\n"
                "Use this evidence to choose correct labels/properties and narrow filters.\n"
                f"{graph_context}"
            )
        if product_filter and product_filter.get("product_key"):
            prompt += (
                "\n\n## Mandatory Product Scope\n"
                f"Restrict query to product_key = '{product_filter['product_key']}'. "
                "Apply this filter to all Hazard/Control/Cause/Consequence matches."
            )
        if cross_product:
            prompt += (
                "\n\n## Mandatory Cross-Product Scope\n"
                "Do NOT filter to a single product_key. Return results across all products. "
                "Include product_name or product_key in returned fields, grouped per product when relevant."
            )
        
        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a Cypher query expert. Return ONLY valid Cypher queries, no explanations or markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        cypher = response.choices[0].message.content.strip()
        
        # Clean up the response
        if cypher.startswith("```"):
            lines = cypher.split('\n')
            cypher = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        
        return cypher

    def _apply_product_scope_to_cypher(self, cypher: str, product_key: str) -> str:
        """Best-effort guardrail: inject product filter into relevant MATCH clauses."""
        if not cypher or not product_key:
            return cypher
        if "product_key" in cypher:
            return cypher

        pattern = re.compile(r"\((\w+)\s*:\s*(Hazard|Control|Cause|Consequence)\b")
        scoped_lines = cypher.splitlines()
        for i, line in enumerate(scoped_lines):
            aliases = [m.group(1) for m in pattern.finditer(line)]
            if aliases and re.search(r"\bMATCH\b", line, flags=re.IGNORECASE):
                clause = " AND ".join([f"{alias}.product_key = '{product_key}'" for alias in aliases])
                if re.search(r"\bWHERE\b", line, flags=re.IGNORECASE):
                    scoped_lines[i] = f"{line} AND {clause}"
                else:
                    # Handle multiline style: MATCH (...) \n WHERE ...
                    next_where_idx = None
                    for j in range(i + 1, len(scoped_lines)):
                        if scoped_lines[j].strip() == "":
                            continue
                        if re.search(r"^\s*WHERE\b", scoped_lines[j], flags=re.IGNORECASE):
                            next_where_idx = j
                        break

                    if next_where_idx is not None:
                        scoped_lines[next_where_idx] = f"{scoped_lines[next_where_idx]} AND {clause}"
                    else:
                        scoped_lines[i] = f"{line} WHERE {clause}"
        return "\n".join(scoped_lines)
    
    def execute_cypher(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self.driver:
            raise ValueError("Neo4j driver not initialized. Call connect() first.")
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = []
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j Node objects
                        if hasattr(value, 'labels'):
                            # It's a Node - extract properties as dict
                            node_props = {}
                            for prop_key, prop_val in value.items():
                                node_props[prop_key] = prop_val
                            record_dict[key] = node_props
                        # Handle Neo4j Relationship objects
                        elif hasattr(value, 'type') and hasattr(value, 'start_node'):
                            # It's a Relationship - extract properties
                            rel_props = {'_type': value.type}
                            for prop_key, prop_val in value.items():
                                rel_props[prop_key] = prop_val
                            record_dict[key] = rel_props
                        # Handle lists (could contain nodes)
                        elif isinstance(value, list):
                            processed_list = []
                            for item in value:
                                if hasattr(item, 'labels'):
                                    node_props = {}
                                    for prop_key, prop_val in item.items():
                                        node_props[prop_key] = prop_val
                                    processed_list.append(node_props)
                                elif hasattr(item, 'type') and hasattr(item, 'start_node'):
                                    rel_props = {'_type': item.type}
                                    for prop_key, prop_val in item.items():
                                        rel_props[prop_key] = prop_val
                                    processed_list.append(rel_props)
                                else:
                                    processed_list.append(item)
                            record_dict[key] = processed_list
                        else:
                            # Regular value (string, int, None, etc.)
                            record_dict[key] = value
                    records.append(record_dict)
                return records
        except Exception as e:
            raise Exception(f"Cypher execution error: {str(e)}. Query: {cypher}")
    
    def format_results(self, question: str, cypher: str, results: List[Dict[str, Any]]) -> str:
        """Format query results into natural language using LLM."""
        if not self.client:
            return self._simple_format(results)
        
        if not results:
            return "No results found for your query."
        
        results_text = str(results[:20])
        if len(results) > 20:
            results_text += f"\n... and {len(results) - 20} more results"
        
        prompt = f"""Based on the following query results from a risk analysis database, provide a clear and concise answer to the user's question.

User Question: {question}

Cypher Query Used: {cypher}

Query Results: {results_text}

Provide a helpful, conversational answer that summarizes the key findings. Include specific data points where relevant."""

        response = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant explaining risk analysis data. Be concise but thorough."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    def _simple_format(self, results: List[Dict[str, Any]]) -> str:
        """Simple formatting without LLM."""
        if not results:
            return "No results found."
        
        output = []
        for i, record in enumerate(results[:10], 1):
            items = [f"{k}: {v}" for k, v in record.items()]
            output.append(f"{i}. " + ", ".join(items))
        
        if len(results) > 10:
            output.append(f"... and {len(results) - 10} more results")
        
        return "\n".join(output)
    
    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """
        Full query pipeline: question -> graph chunk retrieval -> cypher -> execute -> format
        Returns: (cypher_query, raw_results, formatted_answer, retrieved_chunks)
        """
        chunks: List[Dict[str, Any]] = []
        cross_product = self._is_cross_product_query(question)
        product = None if cross_product else self._detect_product_filter(question)
        metadata_filter = None
        if product and product.get("product_key"):
            metadata_filter = {"product_key": product["product_key"]}
        try:
            if cross_product:
                chunks = self._retrieve_diverse_graph_chunks(question, top_k=self.graph_top_k)
            else:
                chunks = self.retrieve_graph_chunks(
                    question,
                    top_k=self.graph_top_k,
                    metadata_filter=metadata_filter,
                )
        except Exception:
            # Retrieval failure should not block query execution.
            chunks = []

        cypher = self.generate_cypher(
            question,
            context_chunks=chunks,
            product_filter=product,
            cross_product=cross_product,
        )
        if metadata_filter:
            cypher = self._apply_product_scope_to_cypher(cypher, metadata_filter["product_key"])
        elif cross_product and "product_key" in cypher:
            # Retry once with a stronger no-filter hint if model over-scopes.
            retry_question = (
                question
                + " IMPORTANT: return results for all products, do not use product_key='...'."
            )
            cypher_retry = self.generate_cypher(
                retry_question,
                context_chunks=chunks,
                product_filter=None,
                cross_product=True,
            )
            if "product_key" not in cypher_retry:
                cypher = cypher_retry
        
        if cypher.startswith("// Cannot answer"):
            return cypher, [], cypher.replace("// ", ""), chunks
        
        try:
            results = self.execute_cypher(cypher)
            answer = self.format_results(question, cypher, results)
            return cypher, results, answer, chunks
        except Exception as e:
            error_msg = f"Query execution error: {str(e)}"
            return cypher, [], error_msg, chunks
    
    def get_sample_queries(self) -> List[Dict[str, str]]:
        """Return sample queries for the UI - covers all scope requirements."""
        return [
            {
                "question": "Which controls mitigate multiple high-severity hazards?",
                "description": "Find shared controls for severe hazards (Control reuse)"
            },
            {
                "question": "What hazards remain above ALAP after all controls are applied?",
                "description": "Find residual risks still at Medium or High level"
            },
            {
                "question": "Which hazards occur during maintenance or service phases?",
                "description": "Lifecycle-specific hazard analysis"
            },
            {
                "question": "Show me the hazard-cause-consequence chain for thermal hazards",
                "description": "Full risk chain exploration"
            },
            {
                "question": "What controls reference the IEC 61010-1 safety standard?",
                "description": "Standard-specific control search"
            },
            {
                "question": "Which hazards affect the User actor?",
                "description": "Actor-based hazard filtering"
            },
            {
                "question": "What is the average risk reduction achieved by each control?",
                "description": "Control effectiveness analysis"
            },
            {
                "question": "For Incubators, list high residual risk hazards only for that product",
                "description": "Product-scoped Graph RAG query"
            },
            {
                "question": "Which controls are documented in the Operating Instructions?",
                "description": "Documentation-based control search"
            },
            {
                "question": "Show hazards with their lifecycle phases, actors, and controls",
                "description": "Complete hazard context view"
            },
            {
                "question": "Which hazard categories have the most high-severity hazards?",
                "description": "Category-level risk analysis"
            },
            {
                "question": "Find controls that achieve both probability and severity reduction",
                "description": "Dual-effect mitigation measures"
            },
            {
                "question": "What hazards are related to electrical or power supply issues?",
                "description": "Topic-based hazard search"
            }
        ]


# Predefined Cypher queries for common questions
PREDEFINED_QUERIES = {
    "controls_multiple_hazards": """
        MATCH (c:Control)<-[:MITIGATED_BY]-(h:Hazard)
        WITH c, count(h) as hazard_count, collect(h.id) as hazard_ids
        WHERE hazard_count > 1
        RETURN c.id as control_id, c.description as description, 
               hazard_count, hazard_ids
        ORDER BY hazard_count DESC
    """,
    
    "high_severity_hazards": """
        MATCH (h:Hazard)
        WHERE h.s_init >= 4
        RETURN h.id as hazard_id, h.name as hazard_name, 
               h.s_init as initial_severity, h.s_final as final_severity
        ORDER BY h.s_init DESC
    """,
    
    "residual_risk_above_alap": """
        MATCH (h:Hazard)
        WHERE h.r_final IN ['M', 'H']
        RETURN h.id as hazard_id, h.name as hazard_name,
               h.r_init as initial_risk, h.r_final as final_risk
        ORDER BY h.r_final DESC
    """,
    
    "hazards_by_lifecycle": """
        MATCH (h:Hazard)-[:OCCURS_DURING]->(l:LifecyclePhase)
        RETURN l.name as lifecycle_phase, collect(h.id) as hazard_ids, count(h) as count
        ORDER BY count DESC
    """,
    
    "controls_by_standard": """
        MATCH (c:Control)-[:REFERENCES]->(s:Standard)
        RETURN s.id as standard, s.name as description, 
               collect(c.id) as controls, count(c) as count
        ORDER BY count DESC
    """,
    
    "hazards_by_actor": """
        MATCH (h:Hazard)-[:AFFECTS]->(a:Actor)
        RETURN a.name as actor, collect(h.id) as hazard_ids, count(h) as count
        ORDER BY count DESC
    """,
    
    "risk_reduction_effectiveness": """
        MATCH (h:Hazard)-[r:MITIGATED_BY]->(c:Control)
        WHERE r.p_reduction IS NOT NULL OR r.s_reduction IS NOT NULL
        RETURN c.id as control_id, c.description,
               avg(r.p_reduction) as avg_prob_reduction,
               avg(r.s_reduction) as avg_severity_reduction,
               count(h) as hazards_mitigated
        ORDER BY avg_severity_reduction DESC
    """,
    
    "full_hazard_chain": """
        MATCH (hc:HazardCategory)-[:CONTAINS]->(h:Hazard)
        OPTIONAL MATCH (h)-[:HAS_CAUSE]->(ca:Cause)
        OPTIONAL MATCH (h)-[:HAS_CONSEQUENCE]->(co:Consequence)
        OPTIONAL MATCH (h)-[:MITIGATED_BY]->(c:Control)
        RETURN hc.name as category, h.id as hazard_id, h.name as hazard,
               ca.description as cause, co.description as consequence,
               collect(DISTINCT c.id) as controls
        LIMIT 20
    """
}


def create_query_engine(neo4j_uri: str = None, neo4j_user: str = None,
                        neo4j_password: str = None, openai_api_key: str = None) -> QueryEngine:
    """Factory function to create and connect a QueryEngine."""
    engine = QueryEngine(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
    engine.connect()
    return engine


if __name__ == "__main__":
    engine = QueryEngine()
    
    try:
        engine.connect()
        question = "Which controls mitigate multiple hazards?"
        print(f"Question: {question}")
        
        cypher, results, answer, chunks = engine.query(question)
        print(f"\nGenerated Cypher:\n{cypher}")
        print(f"\nResults count: {len(results)}")
        print(f"\nRetrieved chunks: {len(chunks)}")
        print(f"\nFormatted Answer:\n{answer}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        engine.close()
