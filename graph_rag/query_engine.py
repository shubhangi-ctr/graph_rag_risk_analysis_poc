"""
Query Engine Module
Converts natural language questions to Cypher queries using LLM and executes them.
Supports full scope requirements including all node types and relationships.
"""
import os
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
        
        return True
    
    def close(self):
        """Close connections."""
        if self.driver:
            self.driver.close()
    
    def retrieve_graph_chunks(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
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
                        text,
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def generate_cypher(self, question: str, context_chunks: List[Dict[str, Any]] = None) -> str:
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
        try:
            chunks = self.retrieve_graph_chunks(question, top_k=self.graph_top_k)
        except Exception:
            # Retrieval failure should not block query execution.
            chunks = []

        cypher = self.generate_cypher(question, context_chunks=chunks)
        
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
