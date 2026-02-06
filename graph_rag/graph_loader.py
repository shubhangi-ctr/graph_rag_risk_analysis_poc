"""
Graph Loader Module
Loads parsed Excel data into Neo4j graph database.
Implements full scope requirements including:
- Hazard ↔ Cause ↔ Consequence chains
- Control reuse across hazards
- Risk reduction before and after mitigation (on relationships)
- Contextual links to standards, instructions, and lifecycle stages
- Shared controls tracking
"""
from typing import Optional
from neo4j import GraphDatabase
from excel_parser import ParsedData
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GraphLoader:
    """Loads structured data into Neo4j."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver = None
    
    def connect(self):
        """Establish connection to Neo4j."""
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # Test connection
        with self.driver.session() as session:
            session.run("RETURN 1")
        return True
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def create_constraints(self):
        """Create uniqueness constraints for node IDs."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hazard) REQUIRE h.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Control) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ca:Cause) REQUIRE ca.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (co:Consequence) REQUIRE co.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (hc:HazardCategory) REQUIRE hc.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Standard) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:DocumentSection) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:LifecyclePhase) REQUIRE l.name IS UNIQUE",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
    
    def load_data(self, data: ParsedData, progress_callback=None):
        """Load all parsed data into Neo4j."""
        total_steps = 14
        current_step = 0
        
        def update_progress(message: str):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step / total_steps, message)
        
        # Create constraints
        self.create_constraints()
        update_progress("Created constraints")
        
        # Load actors
        self._load_actors(data.actors)
        update_progress(f"Loaded {len(data.actors)} actors")
        
        # Load hazard categories
        self._load_hazard_categories(data.hazard_categories)
        update_progress(f"Loaded {len(data.hazard_categories)} hazard categories")
        
        # Load standards
        self._load_standards(data.standards)
        update_progress(f"Loaded {len(data.standards)} standards")
        
        # Load document sections
        self._load_document_sections(data.document_sections)
        update_progress(f"Loaded {len(data.document_sections)} document sections")
        
        # Load lifecycle phases
        self._load_lifecycle_phases(data.lifecycle_phases)
        update_progress(f"Loaded {len(data.lifecycle_phases)} lifecycle phases")
        
        # Load controls
        self._load_controls(data.controls)
        update_progress(f"Loaded {len(data.controls)} controls")
        
        # Load hazards
        self._load_hazards(data.hazards)
        update_progress(f"Loaded {len(data.hazards)} hazards")
        
        # Load causes and link to hazards
        self._load_causes(data.causes)
        update_progress(f"Loaded {len(data.causes)} causes")
        
        # Load consequences and link to hazards
        self._load_consequences(data.consequences)
        update_progress(f"Loaded {len(data.consequences)} consequences")
        
        # Create hazard-control relationships (with risk reduction properties)
        self._link_hazards_to_controls(data.hazard_control_links)
        update_progress(f"Created {len(data.hazard_control_links)} hazard-control links")
        
        # Create hazard-category relationships
        self._link_hazards_to_categories(data.hazard_category_links)
        update_progress(f"Created {len(data.hazard_category_links)} hazard-category links")
        
        # Create hazard-actor relationships
        self._link_hazards_to_actors(data.hazard_actor_links)
        update_progress(f"Created {len(data.hazard_actor_links)} hazard-actor links")
        
        # Create hazard-lifecycle relationships
        self._link_hazards_to_lifecycles(data.hazard_lifecycle_links)
        update_progress(f"Created {len(data.hazard_lifecycle_links)} hazard-lifecycle links")
        
        # Create control-standard relationships
        self._link_controls_to_standards(data.control_standard_links)
        
        # Create control-document relationships
        self._link_controls_to_documents(data.control_document_links)
    
    def _load_actors(self, actors):
        """Load Actor nodes."""
        query = """
        UNWIND $actors AS actor
        MERGE (a:Actor {name: actor.name})
        """
        with self.driver.session() as session:
            session.run(query, actors=[{"name": a.name} for a in actors])
    
    def _load_hazard_categories(self, categories):
        """Load HazardCategory nodes."""
        query = """
        UNWIND $categories AS cat
        MERGE (hc:HazardCategory {id: cat.id})
        SET hc.name = cat.name
        """
        with self.driver.session() as session:
            session.run(query, categories=[{"id": c.id, "name": c.name} for c in categories])
    
    def _load_standards(self, standards):
        """Load Standard nodes."""
        query = """
        UNWIND $standards AS std
        MERGE (s:Standard {id: std.id})
        SET s.name = std.name
        """
        with self.driver.session() as session:
            session.run(query, standards=[{"id": s.id, "name": s.name} for s in standards])
    
    def _load_document_sections(self, documents):
        """Load DocumentSection nodes."""
        query = """
        UNWIND $documents AS doc
        MERGE (d:DocumentSection {id: doc.id})
        SET d.name = doc.name,
            d.document_type = doc.document_type
        """
        with self.driver.session() as session:
            session.run(query, documents=[
                {"id": d.id, "name": d.name, "document_type": d.document_type} 
                for d in documents
            ])
    
    def _load_lifecycle_phases(self, phases):
        """Load LifecyclePhase nodes."""
        query = """
        UNWIND $phases AS phase
        MERGE (l:LifecyclePhase {name: phase.name})
        """
        with self.driver.session() as session:
            session.run(query, phases=[{"name": p.name} for p in phases])
    
    def _load_controls(self, controls):
        """Load Control nodes."""
        query = """
        UNWIND $controls AS ctrl
        MERGE (c:Control {id: ctrl.id})
        SET c.description = ctrl.description,
            c.implementation_reference = ctrl.implementation_reference
        """
        with self.driver.session() as session:
            session.run(query, controls=[
                {
                    "id": c.id,
                    "description": c.description,
                    "implementation_reference": c.implementation_reference
                } for c in controls
            ])
    
    def _load_hazards(self, hazards):
        """Load Hazard nodes with risk scores. Only update name if not empty."""
        # First, create/update hazards with non-empty names
        query_with_name = """
        UNWIND $hazards AS h
        MERGE (haz:Hazard {id: h.id})
        SET haz.name = CASE WHEN h.name IS NOT NULL AND h.name <> '' THEN h.name ELSE haz.name END,
            haz.h_type = CASE WHEN h.h_type IS NOT NULL AND h.h_type <> '' THEN h.h_type ELSE haz.h_type END,
            haz.q_source = CASE WHEN h.q_source IS NOT NULL AND h.q_source <> '' THEN h.q_source ELSE haz.q_source END,
            haz.p_init = COALESCE(h.p_init, haz.p_init),
            haz.s_init = COALESCE(h.s_init, haz.s_init),
            haz.r_init = CASE WHEN h.r_init IS NOT NULL AND h.r_init <> '' THEN h.r_init ELSE haz.r_init END,
            haz.p_final = COALESCE(h.p_final, haz.p_final),
            haz.s_final = COALESCE(h.s_final, haz.s_final),
            haz.r_final = CASE WHEN h.r_final IS NOT NULL AND h.r_final <> '' THEN h.r_final ELSE haz.r_final END
        """
        with self.driver.session() as session:
            session.run(query_with_name, hazards=[
                {
                    "id": h.id,
                    "name": h.name if h.name and h.name != 'nan' else None,
                    "h_type": h.h_type if h.h_type and h.h_type != 'nan' else None,
                    "q_source": h.q_source if h.q_source and h.q_source != 'nan' else None,
                    "p_init": h.p_init,
                    "s_init": h.s_init,
                    "r_init": h.r_init if h.r_init and h.r_init != 'nan' else None,
                    "p_final": h.p_final,
                    "s_final": h.s_final,
                    "r_final": h.r_final if h.r_final and h.r_final != 'nan' else None
                } for h in hazards
            ])
    
    def _load_causes(self, causes):
        """Load Cause nodes and link to hazards."""
        query = """
        UNWIND $causes AS c
        MERGE (ca:Cause {id: c.id})
        SET ca.description = c.description
        WITH ca, c
        MATCH (h:Hazard {id: c.hazard_id})
        MERGE (h)-[:HAS_CAUSE]->(ca)
        """
        with self.driver.session() as session:
            session.run(query, causes=[
                {
                    "id": c.id,
                    "description": c.description,
                    "hazard_id": c.hazard_id
                } for c in causes
            ])
    
    def _load_consequences(self, consequences):
        """Load Consequence nodes and link to hazards."""
        query = """
        UNWIND $consequences AS c
        MERGE (co:Consequence {id: c.id})
        SET co.description = c.description
        WITH co, c
        MATCH (h:Hazard {id: c.hazard_id})
        MERGE (h)-[:HAS_CONSEQUENCE]->(co)
        """
        with self.driver.session() as session:
            session.run(query, consequences=[
                {
                    "id": c.id,
                    "description": c.description,
                    "hazard_id": c.hazard_id
                } for c in consequences
            ])
    
    def _link_hazards_to_controls(self, links):
        """Create MITIGATED_BY relationships with risk reduction properties."""
        query = """
        UNWIND $links AS link
        MATCH (h:Hazard {id: link.hazard_id})
        MATCH (c:Control {id: link.control_id})
        MERGE (h)-[r:MITIGATED_BY]->(c)
        SET r.p_reduction = link.p_reduction,
            r.s_reduction = link.s_reduction
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {
                    "hazard_id": l.hazard_id, 
                    "control_id": l.control_id,
                    "p_reduction": l.p_reduction,
                    "s_reduction": l.s_reduction
                }
                for l in links
            ])
    
    def _link_hazards_to_categories(self, links):
        """Create CONTAINS relationships between categories and hazards."""
        query = """
        UNWIND $links AS link
        MATCH (h:Hazard {id: link.hazard_id})
        MATCH (hc:HazardCategory {id: link.category_id})
        MERGE (hc)-[:CONTAINS]->(h)
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {"hazard_id": l.hazard_id, "category_id": l.category_id}
                for l in links
            ])
    
    def _link_hazards_to_actors(self, links):
        """Create AFFECTS relationships between hazards and actors."""
        query = """
        UNWIND $links AS link
        MATCH (h:Hazard {id: link.hazard_id})
        MATCH (a:Actor {name: link.actor_name})
        MERGE (h)-[:AFFECTS]->(a)
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {"hazard_id": l.hazard_id, "actor_name": l.actor_name}
                for l in links
            ])
    
    def _link_hazards_to_lifecycles(self, links):
        """Create OCCURS_DURING relationships between hazards and lifecycle phases."""
        query = """
        UNWIND $links AS link
        MATCH (h:Hazard {id: link.hazard_id})
        MATCH (l:LifecyclePhase {name: link.lifecycle_phase})
        MERGE (h)-[:OCCURS_DURING]->(l)
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {"hazard_id": l.hazard_id, "lifecycle_phase": l.lifecycle_phase}
                for l in links
            ])
    
    def _link_controls_to_standards(self, links):
        """Create REFERENCES relationships between controls and standards."""
        query = """
        UNWIND $links AS link
        MATCH (c:Control {id: link.control_id})
        MATCH (s:Standard {id: link.standard_id})
        MERGE (c)-[:REFERENCES]->(s)
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {"control_id": l.control_id, "standard_id": l.standard_id}
                for l in links
            ])
    
    def _link_controls_to_documents(self, links):
        """Create DOCUMENTED_IN relationships between controls and document sections."""
        query = """
        UNWIND $links AS link
        MATCH (c:Control {id: link.control_id})
        MATCH (d:DocumentSection {id: link.document_id})
        MERGE (c)-[:DOCUMENTED_IN]->(d)
        """
        with self.driver.session() as session:
            session.run(query, links=[
                {"control_id": l.control_id, "document_id": l.document_id}
                for l in links
            ])
    
    def get_statistics(self) -> dict:
        """Get counts of nodes and relationships in the database."""
        stats = {}
        
        queries = {
            "hazards": "MATCH (h:Hazard) RETURN count(h) as count",
            "controls": "MATCH (c:Control) RETURN count(c) as count",
            "causes": "MATCH (ca:Cause) RETURN count(ca) as count",
            "consequences": "MATCH (co:Consequence) RETURN count(co) as count",
            "categories": "MATCH (hc:HazardCategory) RETURN count(hc) as count",
            "actors": "MATCH (a:Actor) RETURN count(a) as count",
            "standards": "MATCH (s:Standard) RETURN count(s) as count",
            "document_sections": "MATCH (d:DocumentSection) RETURN count(d) as count",
            "lifecycle_phases": "MATCH (l:LifecyclePhase) RETURN count(l) as count",
            "hazard_control_links": "MATCH ()-[r:MITIGATED_BY]->() RETURN count(r) as count",
            "hazard_cause_links": "MATCH ()-[r:HAS_CAUSE]->() RETURN count(r) as count",
            "hazard_consequence_links": "MATCH ()-[r:HAS_CONSEQUENCE]->() RETURN count(r) as count",
            "hazard_actor_links": "MATCH ()-[r:AFFECTS]->() RETURN count(r) as count",
            "hazard_lifecycle_links": "MATCH ()-[r:OCCURS_DURING]->() RETURN count(r) as count",
            "control_standard_links": "MATCH ()-[r:REFERENCES]->() RETURN count(r) as count",
            "control_document_links": "MATCH ()-[r:DOCUMENTED_IN]->() RETURN count(r) as count",
        }
        
        with self.driver.session() as session:
            for key, query in queries.items():
                result = session.run(query)
                record = result.single()
                stats[key] = record["count"] if record else 0
        
        return stats


def load_to_neo4j(data: ParsedData, uri: str = None, user: str = None, 
                  password: str = None, clear_existing: bool = True,
                  progress_callback=None) -> dict:
    """Convenience function to load data into Neo4j."""
    loader = GraphLoader(uri, user, password)
    
    try:
        loader.connect()
        
        if clear_existing:
            loader.clear_database()
        
        loader.load_data(data, progress_callback)
        stats = loader.get_statistics()
        
        return stats
    finally:
        loader.close()


if __name__ == "__main__":
    # Test loading
    from excel_parser import parse_excel
    import sys
    
    if len(sys.argv) > 1:
        data = parse_excel(sys.argv[1])
        stats = load_to_neo4j(data)
        print("Database statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")