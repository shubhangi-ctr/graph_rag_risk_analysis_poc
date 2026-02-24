"""
Combined RAG Application - Graph RAG vs Traditional RAG Side-by-Side Comparison
"""
from traditional_rag import query_engine as trad_query_engine
from traditional_rag.vector_store import ChromaVectorStore, build_rag_documents
from graph_rag import query_engine as graph_query_engine
from graph_rag.graph_loader import load_to_neo4j
from graph_rag.chroma_store import GraphChromaStore, sync_graph_knowledge_from_neo4j
import config
from excel_parser import parse_excel
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import zipfile
import tempfile
import shutil
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Neo4j Browser color palette
NODE_COLORS = {
    'Hazard': '#F79767',
    'Control': '#57C7E3',
    'Cause': '#F16667',
    'Consequence': '#D9C8AE',
    'HazardCategory': '#8DCC93',
    'Actor': '#ECB5C9',
    'Standard': '#4C8EDA',
    'DocumentSection': '#FFC454',
    'LifecyclePhase': '#DA7194',
}

NODE_SIZES = {
    'Hazard': 25,
    'Control': 20,
    'Cause': 15,
    'Consequence': 15,
    'HazardCategory': 30,
    'Actor': 20,
    'Standard': 20,
    'DocumentSection': 15,
    'LifecyclePhase': 20,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_LOAD_MARKER = os.path.join(BASE_DIR, ".data_initialized.json")
INDEX_SCHEMA_VERSION = 2


def init_session_state():
    defaults = {
        'messages': [],
        'graph_rag_loaded': False,
        'trad_rag_loaded': False,
        'graph_query_engine': None,
        'trad_query_engine': None,
        'vector_store': None,
        'graph_vector_store': None,
        'graph_stats': None,
        'trad_stats': None,
        'data_bootstrap_done': False,
        'pending_clarification': None,
        'clarification_mode': True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _load_data_marker() -> Dict[str, Any]:
    if not os.path.exists(DATA_LOAD_MARKER):
        return {}
    try:
        with open(DATA_LOAD_MARKER, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_data_marker(sources: List[str]) -> None:
    payload = {
        "initialized_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "index_schema_version": INDEX_SCHEMA_VERSION,
    }
    with open(DATA_LOAD_MARKER, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _discover_data_sources() -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Returns:
        - List[(excel_file_path, source_name)]
        - Temporary directories created for extracted ZIP contents
    """
    sources: List[Tuple[str, str]] = []
    temp_dirs: List[str] = []

    if not os.path.isdir(DATA_DIR):
        return sources, temp_dirs

    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            file_path = os.path.join(root, name)
            lower = name.lower()

            if lower.endswith((".xlsx", ".xls")):
                source_name = os.path.relpath(file_path, DATA_DIR)
                sources.append((file_path, source_name))
                continue

            if lower.endswith(".zip"):
                extract_dir = tempfile.mkdtemp(prefix="rag_zip_")
                temp_dirs.append(extract_dir)
                try:
                    with zipfile.ZipFile(file_path) as zf:
                        zf.extractall(extract_dir)
                except Exception:
                    continue

                for z_root, _, z_files in os.walk(extract_dir):
                    for z_name in z_files:
                        if not z_name.lower().endswith((".xlsx", ".xls")):
                            continue
                        z_path = os.path.join(z_root, z_name)
                        rel_in_zip = os.path.relpath(z_path, extract_dir)
                        source_name = f"{os.path.basename(file_path)}::{rel_in_zip}"
                        sources.append((z_path, source_name))

    sources.sort(key=lambda item: item[1].lower())
    return sources, temp_dirs


def _graph_chroma_store_has_data() -> bool:
    try:
        if not st.session_state.graph_vector_store:
            st.session_state.graph_vector_store = GraphChromaStore(
                persist_dir=config.CHROMA_PERSIST_DIR,
                collection_name=config.GRAPH_CHROMA_COLLECTION,
            )
        return st.session_state.graph_vector_store.count() > 0
    except Exception:
        return False


def _graph_chroma_store_has_product_metadata() -> bool:
    try:
        store = GraphChromaStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.GRAPH_CHROMA_COLLECTION,
        )
        return store.has_metadata_field("product_key")
    except Exception:
        return False


def _traditional_store_has_data() -> bool:
    try:
        store = ChromaVectorStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION,
        )
        return store.count() > 0
    except Exception:
        return False


def _traditional_store_has_product_metadata() -> bool:
    try:
        store = ChromaVectorStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION,
        )
        return store.has_metadata_field("product_key")
    except Exception:
        return False


def _init_graph_engine(uri: str, user: str, pwd: str, openai_key: str) -> bool:
    if st.session_state.graph_rag_loaded and st.session_state.graph_query_engine:
        return True
    try:
        qe = graph_query_engine.QueryEngine(uri, user, pwd, openai_key)
        qe.connect()
        graph_chunks = qe.graph_store.count() if qe.graph_store else 0
        with qe.driver.session() as session:
            row = session.run(
                "MATCH (h:Hazard) RETURN count(h) AS count").single()
        if not row or row["count"] <= 0 or graph_chunks <= 0:
            qe.close()
            return False
        st.session_state.graph_query_engine = qe
        st.session_state.graph_rag_loaded = True
        st.session_state.graph_stats = {
            **(st.session_state.graph_stats or {}),
            "hazards": row["count"],
            "graph_chunks": graph_chunks,
        }
        return True
    except Exception:
        return False


def _init_traditional_engine(openai_key: str) -> bool:
    if st.session_state.trad_rag_loaded and st.session_state.trad_query_engine:
        return True
    try:
        if not st.session_state.vector_store:
            st.session_state.vector_store = ChromaVectorStore(
                persist_dir=config.CHROMA_PERSIST_DIR,
                collection_name=config.CHROMA_COLLECTION,
            )
        store = st.session_state.vector_store
        if store.count() <= 0:
            return False
        qe = trad_query_engine.QueryEngine(
            vector_store=store, openai_api_key=openai_key)
        qe.connect()
        st.session_state.trad_query_engine = qe
        st.session_state.trad_rag_loaded = True
        st.session_state.trad_stats = {'indexed': store.count()}
        return True
    except Exception:
        return False


def bootstrap_data_from_directory() -> None:
    if st.session_state.get('data_bootstrap_done', False):
        return

    marker = _load_data_marker()
    marker_schema_version = int(marker.get(
        "index_schema_version", 0)) if marker else 0
    schema_needs_reindex = marker_schema_version < INDEX_SCHEMA_VERSION
    graph_chroma_has_data = _graph_chroma_store_has_data()
    graph_has_product_metadata = _graph_chroma_store_has_product_metadata(
    ) if graph_chroma_has_data else False
    graph_needs_product_reindex = graph_chroma_has_data and not graph_has_product_metadata
    trad_has_data = _traditional_store_has_data()
    trad_has_product_metadata = _traditional_store_has_product_metadata(
    ) if trad_has_data else False
    trad_needs_product_reindex = trad_has_data and not trad_has_product_metadata

    if (
        graph_chroma_has_data
        and trad_has_data
        and not trad_needs_product_reindex
        and not graph_needs_product_reindex
        and not schema_needs_reindex
    ):
        graph_ok = _init_graph_engine(config.NEO4J_URI, config.NEO4J_USER,
                                      config.NEO4J_PASSWORD, config.OPENAI_API_KEY)
        trad_ok = _init_traditional_engine(config.OPENAI_API_KEY)
        if not marker:
            _write_data_marker([])
        if graph_ok and trad_ok:
            st.sidebar.info(
                "Using previously indexed data. Skipping re-index.")
        elif not graph_ok:
            st.sidebar.warning(
                "Graph-KG Chroma has data, so Neo4j upload is skipped. Graph RAG is unavailable until Neo4j data is restored.")
        st.session_state.data_bootstrap_done = True
        return

    need_graph_ingest = schema_needs_reindex or (
        not graph_chroma_has_data) or graph_needs_product_reindex
    need_trad_ingest = schema_needs_reindex or (
        not trad_has_data) or trad_needs_product_reindex

    if schema_needs_reindex and (graph_chroma_has_data or trad_has_data):
        st.sidebar.warning(
            "Indexed data is from an older parsing schema. Re-indexing to include all workbook tabs and improved retrieval coverage."
        )

    if graph_needs_product_reindex:
        st.sidebar.warning(
            "Graph RAG index is from an older schema. Re-indexing to add product-level graph embeddings.")

    if trad_needs_product_reindex:
        st.sidebar.warning(
            "Traditional RAG index is from an older schema. Re-indexing to add product-level embeddings.")

    if marker and (need_graph_ingest or need_trad_ingest):
        st.sidebar.warning(
            "Existing initialization marker found, but one or both stores are empty. Re-indexing from `data` directory.")

    sources, temp_dirs = _discover_data_sources()
    try:
        if not sources:
            st.sidebar.warning(
                f"No Excel/ZIP files found in `{DATA_DIR}`. Add files and restart the app.")
            st.session_state.data_bootstrap_done = True
            return

        if need_graph_ingest and need_trad_ingest:
            st.sidebar.info(
                f"Loading {len(sources)} file(s) from `{DATA_DIR}` into Graph + Traditional RAG...")
        elif need_graph_ingest:
            st.sidebar.info(
                f"Graph-KG Chroma needs refresh. Loading {len(sources)} file(s) to Neo4j, then indexing graph chunks...")
        elif need_trad_ingest:
            st.sidebar.info(
                f"Traditional RAG embeddings need refresh. Loading {len(sources)} file(s) for vector chunks...")

        graph_clear = True
        trad_clear = True
        for file_path, source_name in sources:
            st.sidebar.write(f"Processing: `{source_name}`")
            if need_graph_ingest:
                load_graph_rag(
                    file_path=file_path,
                    source_name=source_name,
                    uri=config.NEO4J_URI,
                    user=config.NEO4J_USER,
                    pwd=config.NEO4J_PASSWORD,
                    clear=graph_clear,
                )
                graph_clear = False

            if need_trad_ingest:
                load_traditional_rag(
                    file_path=file_path,
                    source_name=source_name,
                    openai_key=config.OPENAI_API_KEY,
                    clear=trad_clear,
                )
                trad_clear = False

        if need_graph_ingest:
            index_graph_kg_chunks_from_neo4j(
                uri=config.NEO4J_URI,
                user=config.NEO4J_USER,
                pwd=config.NEO4J_PASSWORD,
                openai_key=config.OPENAI_API_KEY,
                source_name="|".join([s for _, s in sources])[:400],
                clear=True,
            )

        graph_ready = _init_graph_engine(config.NEO4J_URI, config.NEO4J_USER,
                                         config.NEO4J_PASSWORD, config.OPENAI_API_KEY)
        trad_ready = _init_traditional_engine(config.OPENAI_API_KEY)

        if graph_ready and trad_ready:
            _write_data_marker([s for _, s in sources])
            st.sidebar.success("Data initialized from local `data` directory.")
        else:
            st.sidebar.error(
                "Data initialization did not complete for both RAG systems.")
    finally:
        for temp_dir in temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)
        st.session_state.data_bootstrap_done = True


# ==================== GRAPH VISUALIZATION FUNCTIONS ====================

def get_node_id(node):
    if node is None:
        return None
    props = dict(node)
    return props.get('id', props.get('name', str(node.element_id) if hasattr(node, 'element_id') else str(id(node))))


def get_node_label(node):
    if node is None:
        return "Unknown"
    props = dict(node)
    return props.get('name', props.get('description', props.get('id', 'Unknown')))


def create_neo4j_style_graph(nodes: List[Dict], relationships: List[Dict], height: str = "600px") -> str:
    """Generates the split-view HTML component with Full Screen Capability."""
    nodes_data = {}
    vis_nodes = []
    added_nodes = set()

    for node in nodes:
        node_id = str(node.get('id', ''))
        if not node_id or node_id in added_nodes:
            continue

        label = str(node.get('label', node.get('name', node_id)))
        node_type = node.get('type', 'Unknown')
        color = NODE_COLORS.get(node_type, '#999999')
        size = NODE_SIZES.get(node_type, 20)

        display_label = label[:20] + '...' if len(label) > 25 else label

        nodes_data[node_id] = {
            'id': node_id,
            'type': node_type,
            'label': label,
            **{k: v for k, v in node.items() if k not in ['id', 'type', 'label']}
        }

        vis_nodes.append({
            'id': node_id,
            'label': display_label,
            'color': {'background': color, 'border': color},
            'size': size,
            'shape': 'dot',
            'font': {'color': 'white', 'size': 14, 'face': 'arial'}
        })
        added_nodes.add(node_id)

    vis_edges = []
    added_edges = set()
    for rel in relationships:
        source = str(rel.get('source', ''))
        target = str(rel.get('target', ''))
        rel_type = rel.get('type', 'RELATED')

        edge_key = f"{source}-{rel_type}-{target}"
        if source in added_nodes and target in added_nodes and edge_key not in added_edges:
            vis_edges.append({
                'from': source,
                'to': target,
                'label': rel_type,
                'color': {'color': '#666666', 'highlight': '#ffffff'},
                'arrows': 'to',
                'font': {'color': '#cccccc', 'size': 11, 'align': 'middle', 'strokeWidth': 0}
            })
            added_edges.add(edge_key)

    nodes_json = json.dumps(nodes_data).replace("'", "\\'")
    vis_nodes_json = json.dumps(vis_nodes)
    vis_edges_json = json.dumps(vis_edges)
    node_colors_json = json.dumps(NODE_COLORS)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; background-color: #2B2B2B; color: #fff; overflow: hidden; }}
            #container {{ display: flex; height: {height}; border: 1px solid #444; border-radius: 8px; position: relative; overflow: hidden; }}
            #container:fullscreen {{ width: 100vw; height: 100vh; border-radius: 0; }}
            #mynetwork {{ flex: 75; background-color: #2B2B2B; }}
            #sidebar {{ flex: 25; min-width: 200px; background-color: #333; border-left: 1px solid #555; display: flex; flex-direction: column; }}
            .sidebar-header {{ padding: 12px; background: #444; border-bottom: 1px solid #555; font-weight: 600; font-size: 14px; }}
            .sidebar-body {{ padding: 12px; overflow-y: auto; flex: 1; }}
            .node-badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; margin-bottom: 12px; color: #fff; }}
            .prop-row {{ margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 6px; }}
            .prop-key {{ font-size: 10px; text-transform: uppercase; color: #aaa; margin-bottom: 3px; }}
            .prop-val {{ font-size: 12px; color: #eee; word-wrap: break-word; line-height: 1.3; }}
            .empty-state {{ color: #888; text-align: center; margin-top: 40%; font-style: italic; font-size: 13px; }}
            #fs-btn {{
                position: absolute; top: 10px; right: 10px; z-index: 1000;
                background-color: rgba(60, 60, 60, 0.8); color: white;
                border: 1px solid #666; padding: 6px 12px; border-radius: 4px;
                cursor: pointer; font-size: 12px; transition: background 0.2s;
            }}
            #fs-btn:hover {{ background-color: #555; }}
        </style>
    </head>
    <body>
        <div id="container">
            <button id="fs-btn" onclick="toggleFullScreen()">⛶ Full Screen</button>
            <div id="mynetwork"></div>
            <div id="sidebar">
                <div class="sidebar-header">Node Details</div>
                <div class="sidebar-body" id="details-content">
                    <div class="empty-state">Click a node to view details</div>
                </div>
            </div>
        </div>
        <script type="text/javascript">
            const nodesData = {nodes_json};
            const visNodes = {vis_nodes_json};
            const visEdges = {vis_edges_json};
            const nodeColors = {node_colors_json};
            const container = document.getElementById('mynetwork');
            const data = {{ nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) }};
            const options = {{
                nodes: {{ borderWidth: 2, shadow: true }},
                edges: {{ smooth: {{ type: 'continuous' }} }},
                physics: {{ stabilization: {{ enabled: true, iterations: 200 }}, barnesHut: {{ gravitationalConstant: -3000, springConstant: 0.04, springLength: 150 }} }},
                interaction: {{ hover: true }}
            }};
            const network = new vis.Network(container, data, options);
            function toggleFullScreen() {{
                const elem = document.getElementById("container");
                const btn = document.getElementById("fs-btn");
                if (!document.fullscreenElement) {{
                    elem.requestFullscreen().catch(err => {{ alert(`Error: ${{err.message}}`); }});
                    btn.innerHTML = "❌ Exit Full Screen";
                }} else {{
                    document.exitFullscreen();
                    btn.innerHTML = "⛶ Full Screen";
                }}
            }}
            document.addEventListener("fullscreenchange", function() {{
                const btn = document.getElementById("fs-btn");
                if (!document.fullscreenElement) {{ btn.innerHTML = "⛶ Full Screen"; }}
                else {{ btn.innerHTML = "❌ Exit Full Screen"; }}
            }});
            network.on("click", function (params) {{
                const contentDiv = document.getElementById('details-content');
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    const node = nodesData[nodeId];
                    if (node) {{
                        const color = nodeColors[node.type] || '#888';
                        let html = `<div class="node-badge" style="background-color: ${{color}}">${{node.type}}</div>`;
                        Object.keys(node).forEach(key => {{
                            if (key !== 'type' && key !== 'label') {{
                                html += `<div class="prop-row"><div class="prop-key">${{key}}</div><div class="prop-val">${{node[key]}}</div></div>`;
                            }}
                        }});
                        contentDiv.innerHTML = html;
                    }}
                }} else {{
                    contentDiv.innerHTML = '<div class="empty-state">Click a node to view details</div>';
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html


def render_multiview_component(nodes: List[Dict], rels: List[Dict], height="600px", key_prefix="graph"):
    """Renders Graph | Table | Raw tabs"""
    t_graph, t_table, t_raw = st.tabs(["🕸️ Graph", "📋 Table", "📄 Raw"])

    with t_graph:
        html = create_neo4j_style_graph(nodes, rels, height=height)
        components.html(html, height=int(
            height.replace('px', '')) + 20, scrolling=False)

    with t_table:
        st.markdown("**Nodes**")
        if nodes:
            df_nodes = pd.DataFrame(nodes)
            cols = ['id', 'type', 'label'] + \
                [c for c in df_nodes.columns if c not in ['id', 'type', 'label']]
            st.dataframe(df_nodes[[c for c in cols if c in df_nodes.columns]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No nodes")

        st.markdown("**Relationships**")
        if rels:
            df_rels = pd.DataFrame(rels)
            st.dataframe(df_rels, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships")

    with t_raw:
        st.json({"nodes": nodes[:20], "relationships": rels[:20]})


def fetch_subgraph_for_query_results(query_engine, original_cypher: str):
    """Fetch a subgraph showing ALL relationships."""
    nodes = []
    relationships = []
    seen_nodes = set()
    seen_rels = set()
    final_query = ""

    try:
        with query_engine.driver.session() as session:
            result = session.run(original_cypher)
            records = list(result)

            if not records:
                return [], [], ""

            identifiers = set()

            def extract_ids(item):
                if isinstance(item, str) and item:
                    identifiers.add(item)
                elif isinstance(item, (int, float)):
                    identifiers.add(str(item))
                elif isinstance(item, list):
                    for i in item:
                        extract_ids(i)
                elif isinstance(item, dict):
                    for v in item.values():
                        extract_ids(v)
                elif hasattr(item, 'items'):
                    for k, v in item.items():
                        extract_ids(v)
                    if hasattr(item, 'labels'):
                        nid = get_node_id(item)
                        if nid:
                            identifiers.add(nid)

            for record in records:
                for key, value in record.items():
                    extract_ids(value)

            if not identifiers:
                return [], [], ""

            ids_formatted = str(list(identifiers))
            product_key_match = re.search(
                r"product_key\s*=\s*'([^']+)'", original_cypher or "", flags=re.IGNORECASE
            )
            product_anchor_filter = ""
            if product_key_match:
                product_anchor_filter = f" AND anchor.product_key = '{product_key_match.group(1)}'"

            subgraph_query = f"""
            MATCH (anchor)
            WHERE (anchor.id IN {ids_formatted} OR anchor.uid IN {ids_formatted} OR anchor.name IN {ids_formatted}){product_anchor_filter}
            WITH collect(DISTINCT anchor) as anchors
            
            UNWIND anchors as a
            OPTIONAL MATCH (a)-[r]->(target)
            WITH anchors, collect({{source: a, rel: r, target: target}}) as outgoing
            
            UNWIND anchors as a
            OPTIONAL MATCH (source)-[r2]->(a)
            WITH anchors, outgoing, collect({{source: source, rel: r2, target: a}}) as incoming
            
            WITH anchors, outgoing + incoming as all_rels
            UNWIND all_rels as item
            WITH item.source as src, item.rel as r, item.target as tgt
            WHERE src IS NOT NULL AND tgt IS NOT NULL AND r IS NOT NULL
            RETURN DISTINCT src, r, type(r) as rel_type, tgt, labels(src) as src_labels, labels(tgt) as tgt_labels
            """
            final_query = subgraph_query

            result = session.run(subgraph_query)
            records = list(result)

            for record in records:
                src = record.get('src')
                tgt = record.get('tgt')
                r = record.get('r')
                rel_type = record.get('rel_type')

                if src:
                    src_id = get_node_id(src)
                    if src_id and src_id not in seen_nodes:
                        src_type = record.get('src_labels', ['Unknown'])[
                            0] if record.get('src_labels') else 'Unknown'
                        nodes.append({'id': src_id, 'label': get_node_label(
                            src), 'type': src_type, **dict(src)})
                        seen_nodes.add(src_id)

                if tgt:
                    tgt_id = get_node_id(tgt)
                    if tgt_id and tgt_id not in seen_nodes:
                        tgt_type = record.get('tgt_labels', ['Unknown'])[
                            0] if record.get('tgt_labels') else 'Unknown'
                        nodes.append({'id': tgt_id, 'label': get_node_label(
                            tgt), 'type': tgt_type, **dict(tgt)})
                        seen_nodes.add(tgt_id)

                if src and tgt and rel_type:
                    src_id = get_node_id(src)
                    tgt_id = get_node_id(tgt)
                    rel_key = str(src_id) + "-" + \
                        str(rel_type) + "->" + str(tgt_id)
                    if rel_key not in seen_rels:
                        relationships.append(
                            {'source': src_id, 'target': tgt_id, 'type': rel_type})
                        seen_rels.add(rel_key)

            return nodes, relationships, final_query

    except Exception as e:
        print("Subgraph fetch error: " + str(e))
        return [], [], ""


# ==================== QUESTION CLARIFICATION ====================

def _parse_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse a JSON object from model output safely."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _get_llm_client_for_clarification():
    """Reuse an already-initialized OpenAI client from either engine."""
    trad_engine = st.session_state.get("trad_query_engine")
    if trad_engine and getattr(trad_engine, "client", None):
        return trad_engine.client

    graph_engine = st.session_state.get("graph_query_engine")
    if graph_engine and getattr(graph_engine, "client", None):
        return graph_engine.client
    return None


def _collect_indexed_product_names() -> List[str]:
    names = set()
    trad_engine = st.session_state.get("trad_query_engine")
    graph_engine = st.session_state.get("graph_query_engine")
    for engine in (trad_engine, graph_engine):
        if not engine:
            continue
        for product in getattr(engine, "indexed_products", []) or []:
            name = (product.get("product_name") or "").strip()
            if name:
                names.add(name)
    return sorted(names)


def _extract_assistant_turn_summary(turn: Dict[str, Any]) -> str:
    """Compact assistant-side summary from a stored turn for context grounding."""
    if not isinstance(turn, dict):
        return ""
    if turn.get("type") == "clarification":
        return (turn.get("assistant_message") or "").strip()

    graph_answer = ((turn.get("graph_rag") or {}).get("answer") or "").strip()
    trad_answer = ((turn.get("trad_rag") or {}).get("answer") or "").strip()
    summary = graph_answer or trad_answer
    if len(summary) > 350:
        summary = summary[:350] + "..."
    return summary


def _get_last_user_question_from_history() -> str:
    """Return the most recent resolved or raw user question from history."""
    for turn in reversed(st.session_state.get("messages", [])):
        if not isinstance(turn, dict):
            continue
        resolved = (turn.get("resolved_question") or "").strip()
        if resolved:
            return resolved
        q = (turn.get("question") or "").strip()
        if q:
            return q
    return ""


def _serialize_recent_conversation(max_turns: int = 4) -> str:
    """Serialize recent turns for lightweight LLM contextualization."""
    turns = [t for t in st.session_state.get("messages", []) if isinstance(t, dict)]
    if not turns:
        return ""

    selected = turns[-max_turns:]
    lines: List[str] = []
    for i, turn in enumerate(selected, start=1):
        question = (turn.get("resolved_question") or turn.get("question") or "").strip()
        if question:
            lines.append(f"Turn {i} user: {question}")
        assistant_summary = _extract_assistant_turn_summary(turn)
        if assistant_summary:
            lines.append(f"Turn {i} assistant: {assistant_summary}")
    return "\n".join(lines).strip()


def contextualize_question_with_history(question: str) -> Dict[str, Any]:
    """Rewrite a follow-up question into a standalone query using recent history."""
    raw_question = (question or "").strip()
    if not raw_question:
        return {
            "standalone_question": "",
            "used_history": False,
            "context_confused": False,
            "reason": "",
        }

    history_text = _serialize_recent_conversation(max_turns=4)
    if not history_text:
        return {
            "standalone_question": raw_question,
            "used_history": False,
            "context_confused": False,
            "reason": "",
        }

    referential_patterns = [
        r"^\s*(and|also|then|what about|how about)\b",
        r"\b(it|they|them|those|these|that|this|same|above|previous)\b",
        r"^\s*(why|how|which ones?)\b",
    ]
    looks_referential = any(re.search(p, raw_question, flags=re.IGNORECASE) for p in referential_patterns)
    fallback_question = raw_question
    fallback_used_history = False
    fallback_context_confused = False
    fallback_reason = ""
    if looks_referential:
        prev = _get_last_user_question_from_history()
        if prev:
            fallback_question = f"{prev}\nFollow-up focus: {raw_question}"
            fallback_used_history = True
            fallback_reason = "Heuristic merge for referential follow-up."
        else:
            fallback_context_confused = True
            fallback_reason = "Referential follow-up without prior context."

    client = _get_llm_client_for_clarification()
    if not client:
        return {
            "standalone_question": fallback_question,
            "used_history": fallback_used_history,
            "context_confused": fallback_context_confused,
            "reason": fallback_reason,
        }

    prompt = f"""
You convert follow-up chat questions into standalone questions for retrieval.
Use conversation context only to resolve references.

Recent conversation:
{history_text}

Current user message:
{raw_question}

Return ONLY JSON:
{{
  "standalone_question": "single self-contained retrieval-ready question",
  "used_history": true/false,
  "context_confused": true/false,
  "reason": "short phrase"
}}

Rules:
- Keep the user's intent and constraints unchanged.
- Do not answer the question.
- If already standalone, return it unchanged and used_history=false.
- If references are ambiguous even with context, keep the question and used_history=false.
- Set context_confused=true only when referential language cannot be resolved confidently.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite questions using prior chat context. Output strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=240,
        )
        parsed = _parse_json_object(resp.choices[0].message.content or "")
        if not parsed:
            return {
                "standalone_question": fallback_question,
                "used_history": fallback_used_history,
                "context_confused": fallback_context_confused,
                "reason": fallback_reason,
            }

        standalone = (parsed.get("standalone_question") or "").strip() or fallback_question
        used_history = bool(parsed.get("used_history")) and standalone != raw_question
        context_confused = bool(parsed.get("context_confused"))
        if looks_referential and standalone == raw_question and not used_history:
            context_confused = True
        reason = (parsed.get("reason") or "").strip()
        return {
            "standalone_question": standalone,
            "used_history": used_history,
            "context_confused": context_confused,
            "reason": reason,
        }
    except Exception:
        return {
            "standalone_question": fallback_question,
            "used_history": fallback_used_history,
            "context_confused": fallback_context_confused,
            "reason": fallback_reason,
        }


def _is_question_specific_enough(question: str) -> bool:
    """Conservative sufficiency check to avoid over-clarifying good questions."""
    q = (question or "").strip().lower()
    if not q:
        return False
    token_count = len(re.findall(r"\b\w+\b", q))
    has_domain = bool(
        re.search(
            r"\b(hazard|control|risk|cause|consequence|actor|lifecycle|standard|product|maintenance|service)\b",
            q,
        )
    )
    has_goal = bool(
        re.search(
            r"\b(which|what|list|show|find|count|compare|summari[sz]e|identify|top|highest|lowest|average|trend)\b",
            q,
        )
    )
    has_constraints = bool(
        re.search(
            r"\b(for|during|across|by|with|only|above|below|high|medium|low|residual|initial|after|before|per)\b",
            q,
        )
    )
    return (has_domain and has_goal) or (has_goal and has_constraints) or (token_count >= 9 and (has_domain or has_goal))


def _fallback_question_assessment(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    lowered = q.lower()
    token_count = len(re.findall(r"\b\w+\b", q))

    vague_patterns = [
        r"^what about\b",
        r"^tell me\b",
        r"^explain\b",
        r"^help\b",
        r"^details?\b",
        r"^more\b",
    ]
    starts_vague = any(re.search(p, lowered) for p in vague_patterns)
    has_domain_terms = bool(
        re.search(
            r"\b(hazard|control|risk|cause|consequence|actor|lifecycle|standard|product)\b",
            lowered,
        )
    )
    has_intent_terms = bool(
        re.search(
            r"\b(which|what|list|show|find|count|compare|highest|lowest|average|trend|top)\b",
            lowered,
        )
    )

    clearly_specific = _is_question_specific_enough(q)
    needs_clarification = (not clearly_specific) and (
        token_count <= 2
        or (starts_vague and token_count <= 3)
        or (not has_domain_terms and not has_intent_terms and token_count <= 3)
    )
    clarifying_question = (
        "Could you clarify what you want to analyze: hazards, controls, causes, or consequences, "
        "and the output format (list, count, or comparison)?"
    )
    return {
        "needs_clarification": needs_clarification,
        "clarification_required": needs_clarification,
        "insufficient_info": needs_clarification,
        "context_confused": False,
        "missing_fields": [],
        "clarifying_question": clarifying_question,
        "rewritten_question": q,
        "reason": "Question is too broad for precise retrieval." if needs_clarification else "",
    }


def assess_question_clarity(question: str, context_confused: bool = False) -> Dict[str, Any]:
    """Decide whether a follow-up clarification is required before retrieval."""
    fallback = _fallback_question_assessment(question)
    client = _get_llm_client_for_clarification()
    if not client:
        if context_confused:
            fallback.update(
                {
                    "needs_clarification": True,
                    "clarification_required": True,
                    "insufficient_info": False,
                    "context_confused": True,
                    "missing_fields": ["context_reference"],
                    "clarifying_question": "I might be missing the reference. Which previous hazard/control/result are you referring to?",
                    "reason": "Context reference is ambiguous.",
                    "clarification_questions": [
                        "Which previous hazard, control, or result are you referring to?"
                    ],
                }
            )
        return fallback

    products = _collect_indexed_product_names()
    product_hint = ", ".join(products[:12]) if products else "Unknown"
    prompt = f"""
You are a query triage assistant for a risk-analysis RAG system.
Determine whether the user question is specific enough to run retrieval safely.

Question:
{question}

Indexed products (if relevant):
{product_hint}

Return ONLY valid JSON:
{{
  "needs_clarification": true/false,
  "clarification_required": true/false,
  "insufficient_info": true/false,
  "context_confused": true/false,
  "missing_fields": ["scope","target_entity","product_scope","output_format","context_reference"],
  "reason": "short reason",
  "clarifying_question": "single concise question when clarification is needed; otherwise empty string",
  "clarification_questions": ["2-3 targeted short questions only when clarification is needed"],
  "rewritten_question": "clean retrieval-ready rewrite when no clarification is needed; otherwise empty string"
}}

Rules:
- Ask clarification ONLY when it is required to proceed with retrieval safely.
- Required means: unresolved context reference, conflicting constraints, or no usable target/intent.
- If the question is answerable as-is, set needs_clarification=false and provide rewritten_question.
- Do not answer the user question itself.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You classify question clarity and output strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=220,
        )
        parsed = _parse_json_object(resp.choices[0].message.content or "")
        if not parsed:
            return fallback

        needs = bool(parsed.get("needs_clarification"))
        clarification_required = bool(parsed.get("clarification_required"))
        insufficient_info = bool(parsed.get("insufficient_info"))
        parsed_context_confused = bool(parsed.get("context_confused"))
        missing_fields = parsed.get("missing_fields") or []
        if not isinstance(missing_fields, list):
            missing_fields = []
        missing_fields = [str(field).strip() for field in missing_fields if str(field).strip()]
        clarifying_question = (parsed.get("clarifying_question") or "").strip()
        clarification_questions = parsed.get("clarification_questions") or []
        if not isinstance(clarification_questions, list):
            clarification_questions = []
        clarification_questions = [
            str(q).strip()
            for q in clarification_questions
            if str(q).strip()
        ][:3]
        rewritten_question = (parsed.get("rewritten_question") or "").strip()
        reason = (parsed.get("reason") or "").strip()

        combined_context_confused = context_confused or parsed_context_confused
        clearly_specific = _is_question_specific_enough(question)

        # Final conservative policy:
        # Clarify only when (1) insufficient info OR (2) context is ambiguous.
        if combined_context_confused:
            needs = True
            clarification_required = True
        elif clarification_required and (insufficient_info or missing_fields) and not clearly_specific:
            needs = True
        elif clearly_specific:
            needs = False
        else:
            needs = False

        if needs and not clarifying_question:
            clarifying_question = fallback["clarifying_question"]
        if needs and not clarification_questions:
            clarification_questions = [clarifying_question] if clarifying_question else []
        if not needs and not rewritten_question:
            rewritten_question = (question or "").strip()
        if not needs:
            missing_fields = []
            reason = ""
            clarification_questions = []

        return {
            "needs_clarification": needs,
            "clarification_required": clarification_required,
            "insufficient_info": insufficient_info,
            "context_confused": combined_context_confused,
            "missing_fields": missing_fields,
            "clarifying_question": clarifying_question,
            "clarification_questions": clarification_questions,
            "rewritten_question": rewritten_question,
            "reason": reason,
        }
    except Exception:
        if context_confused:
            fallback.update(
                {
                    "needs_clarification": True,
                    "clarification_required": True,
                    "insufficient_info": False,
                    "context_confused": True,
                    "missing_fields": ["context_reference"],
                    "clarifying_question": "I might be missing the reference. Which previous hazard/control/result are you referring to?",
                    "reason": "Context reference is ambiguous.",
                    "clarification_questions": [
                        "Which previous hazard, control, or result are you referring to?"
                    ],
                }
            )
        return fallback


def _fallback_clarification_payload(
    question: str,
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    products = _collect_indexed_product_names()
    missing_fields = assessment.get("missing_fields") or []
    if not isinstance(missing_fields, list):
        missing_fields = []
    missing = {str(field).strip() for field in missing_fields if str(field).strip()}

    questions: List[str] = []
    if "product_scope" in missing:
        if products:
            questions.append("Which product should I focus on?")
        else:
            questions.append("Which product should I focus on?")
    if "target_entity" in missing:
        questions.append("Do you want hazards, controls, causes, or consequences?")
    if "output_format" in missing:
        questions.append("What output do you want: list, count, or comparison?")
    if "scope" in missing:
        questions.append("Should I scope by lifecycle phase, severity, actor, or standard?")
    if "context_reference" in missing:
        questions.append("Which previous result are you referring to?")

    if not questions:
        generic = (assessment.get("clarification_questions") or [])
        if isinstance(generic, list):
            questions = [str(q).strip() for q in generic if str(q).strip()]
        if not questions and assessment.get("clarifying_question"):
            questions = [str(assessment.get("clarifying_question")).strip()]
        if not questions:
            questions = [
                "Which product should I analyze?",
                "Which hazard or control type should I focus on?",
                "What output format do you want?"
            ]

    return {
        "intro": "To give an exact answer, I need a little more detail.",
        "clarification_questions": questions[:3],
        "product_options": products[:5],
        "reason": (assessment.get("reason") or "").strip(),
        "status": "incomplete",
    }


def build_targeted_clarification_payload(
    user_query: str,
    assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate 2-3 targeted clarification questions using product metadata."""
    fallback = _fallback_clarification_payload(user_query, assessment)
    client = _get_llm_client_for_clarification()
    if not client:
        return fallback

    products = _collect_indexed_product_names()
    product_list = ", ".join(products[:12]) if products else "Unknown"
    missing_fields = assessment.get("missing_fields") or []
    if not isinstance(missing_fields, list):
        missing_fields = []
    missing_text = ", ".join([str(field) for field in missing_fields if str(field).strip()]) or "unspecified"

    prompt = f"""
You are a Risk Analysis Assistant.
The user query is: "{user_query}"
Our product catalog is: {product_list}
Missing fields detected by gatekeeper: {missing_text}

Generate a targeted clarification payload to narrow the query safely.
Return ONLY JSON:
{{
  "status": "incomplete",
  "intro": "one short professional sentence",
  "clarification_questions": ["2-3 short specific questions"],
  "product_options": ["0-5 relevant product names from catalog if product scope is missing"],
  "reason": "short reason"
}}

Rules:
- Keep questions concise and directly actionable.
- Ask at most 3 questions.
- If product scope is missing, include product options from the catalog.
- Do not answer the original query.
""".strip()

    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You generate clarification payloads as strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=260,
        )
        parsed = _parse_json_object(resp.choices[0].message.content or "")
        if not parsed:
            return fallback

        intro = (parsed.get("intro") or "").strip() or fallback["intro"]
        reason = (parsed.get("reason") or "").strip() or fallback["reason"]

        questions = parsed.get("clarification_questions") or []
        if not isinstance(questions, list):
            questions = []
        questions = [str(q).strip() for q in questions if str(q).strip()][:3]
        if not questions:
            questions = fallback["clarification_questions"]

        product_options = parsed.get("product_options") or []
        if not isinstance(product_options, list):
            product_options = []
        product_options = [str(p).strip() for p in product_options if str(p).strip()]
        if products:
            allowed = set(products)
            product_options = [p for p in product_options if p in allowed]
        product_options = product_options[:5] if product_options else fallback["product_options"]

        return {
            "status": "incomplete",
            "intro": intro,
            "clarification_questions": questions,
            "product_options": product_options,
            "reason": reason,
        }
    except Exception:
        return fallback


def format_clarification_message(payload: Dict[str, Any]) -> str:
    intro = (payload.get("intro") or "Please clarify your request.").strip()
    questions = payload.get("clarification_questions") or []
    if not isinstance(questions, list):
        questions = []
    questions = [str(q).strip() for q in questions if str(q).strip()][:3]
    products = payload.get("product_options") or []
    if not isinstance(products, list):
        products = []
    products = [str(p).strip() for p in products if str(p).strip()][:5]

    lines = [intro]
    if questions:
        lines.append("")
        lines.append("Please clarify:")
        for i, q in enumerate(questions, start=1):
            lines.append(f"{i}. {q}")
    if products:
        lines.append("")
        lines.append("Available products:")
        for p in products:
            lines.append(f"- {p}")
    return "\n".join(lines).strip()


def resolve_clarified_question(original_question: str, clarification_answer: str) -> str:
    """Merge user clarification back into a retrieval-ready final question."""
    base = (original_question or "").strip()
    extra = (clarification_answer or "").strip()
    if not base:
        return extra
    if not extra:
        return base

    fallback = f"{base}\n\nClarification provided by user: {extra}"
    client = _get_llm_client_for_clarification()
    if not client:
        return fallback

    prompt = f"""
Combine the original question and user clarification into one retrieval-ready question.
Preserve exact user intent; do not add new constraints.

Original question:
{base}

User clarification:
{extra}

Return ONLY JSON:
{{
  "resolved_question": "single final question"
}}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You rewrite questions. Output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=180,
        )
        parsed = _parse_json_object(resp.choices[0].message.content or "")
        resolved = (parsed.get("resolved_question") or "").strip() if parsed else ""
        return resolved or fallback
    except Exception:
        return fallback


# ==================== SIDEBAR & DATA LOADING ====================

def render_sidebar():
    with st.sidebar:
        # st.title("⚖️ RAG Comparison")
        # st.caption("Graph RAG vs Traditional RAG")
        # st.divider()

        # st.header("📁 Data Source")
        # st.caption(f"Auto-loading from local directory: `{DATA_DIR}`")
        bootstrap_data_from_directory()

        # st.divider()
        # col1, col2 = st.columns(2)
        # with col1:
        #     status = "✅" if st.session_state.get(
        #         'graph_rag_loaded', False) else "❌"
        #     st.metric("Graph RAG", status)
        # with col2:
        #     status = "✅" if st.session_state.get(
        #         'trad_rag_loaded', False) else "❌"
        #     st.metric("Trad RAG", status)

        # st.divider()
        st.toggle(
            "Clarify ambiguous questions before retrieval",
            key="clarification_mode",
        )
        st.caption("When enabled, the assistant asks one follow-up question if your query is too broad.")

        st.divider()
        st.header("🎨 Legend")
        # Ensure NODE_COLORS is defined globally or imported
        for node_type, color in NODE_COLORS.items():
            st.markdown(
                f"<span style='color:{color};font-size:16px'>●</span> {node_type}", unsafe_allow_html=True)


def load_graph_rag(file_path, source_name, uri, user, pwd, clear):
    try:
        with st.sidebar:
            progress = st.progress(0, text="Parsing Excel...")
            data = parse_excel(file_path)
            progress.progress(30, text="Loading to Neo4j...")

            stats = load_to_neo4j(data, uri=uri, user=user, password=pwd, clear_existing=clear,
                                  source_name=source_name,
                                  progress_callback=lambda p, m: progress.progress(min(30 + int(p*60), 90), text=m))

            st.session_state.graph_rag_loaded = False
            st.session_state.graph_stats = stats

            progress.progress(100, text="✅ Neo4j Loaded")
            st.toast(
                f"Neo4j data loaded ({source_name}): {stats.get('hazards', 0)} hazards", icon="🕸️")
    except Exception as e:
        st.sidebar.error("Graph RAG Error: " + str(e))


def index_graph_kg_chunks_from_neo4j(uri, user, pwd, openai_key, source_name, clear):
    """Build Graph-KG chunks from Neo4j and index them in Chroma."""
    try:
        with st.sidebar:
            progress = st.progress(0, text="Reading graph from Neo4j...")
            stats = sync_graph_knowledge_from_neo4j(
                neo4j_uri=uri,
                neo4j_user=user,
                neo4j_password=pwd,
                openai_api_key=openai_key,
                persist_dir=config.CHROMA_PERSIST_DIR,
                collection_name=config.GRAPH_CHROMA_COLLECTION,
                embedding_model=config.OPENAI_EMBEDDING_MODEL,
                source_name=source_name,
                reset=clear,
            )
            progress.progress(100, text="✅ Graph chunks indexed in Chroma")
            st.toast(
                f"Graph-KG chunks indexed: {stats.get('graph_indexed_chunks', 0)}", icon="🧠")

            if not st.session_state.graph_vector_store:
                st.session_state.graph_vector_store = GraphChromaStore(
                    persist_dir=config.CHROMA_PERSIST_DIR,
                    collection_name=config.GRAPH_CHROMA_COLLECTION,
                )
            st.session_state.graph_stats = {
                **(st.session_state.graph_stats or {}),
                **stats,
                "graph_chunks": st.session_state.graph_vector_store.count(),
            }
    except Exception as e:
        st.sidebar.error("Graph Chroma Index Error: " + str(e))


def load_traditional_rag(file_path, source_name, openai_key, clear):
    try:
        with st.sidebar:
            progress = st.progress(0, text="Parsing Excel...")
            data = parse_excel(file_path)

            progress.progress(20, text="Building documents...")
            docs, metas, stats = build_rag_documents(
                data, source_name=source_name)

            progress.progress(40, text="Initializing ChromaDB...")
            if not st.session_state.vector_store:
                st.session_state.vector_store = ChromaVectorStore(
                    persist_dir=config.CHROMA_PERSIST_DIR,
                    collection_name=config.CHROMA_COLLECTION
                )

            store = st.session_state.vector_store
            if clear:
                store.reset()

            progress.progress(50, text="Embedding documents...")
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            added = store.add_documents(
                oai=client, documents=docs, metadatas=metas,
                embedding_model=config.OPENAI_EMBEDDING_MODEL
            )

            progress.progress(90, text="Initializing query engine...")
            qe = trad_query_engine.QueryEngine(
                vector_store=store, openai_api_key=openai_key)
            qe.connect()

            st.session_state.trad_query_engine = qe
            st.session_state.trad_rag_loaded = True
            st.session_state.trad_stats = {**stats, 'indexed': added}

            progress.progress(100, text="✅ Traditional RAG Ready!")
            st.toast(
                f"Traditional RAG Loaded ({source_name}): {added} docs", icon="📚")
    except Exception as e:
        st.sidebar.error("Traditional RAG Error: " + str(e))


# ==================== MAIN CHAT INTERFACE ====================

def main():
    init_session_state()
    render_sidebar()

    # Modern Header
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0;">⚖️ RAG Comparison Engine</h1>
            <p style="color: #666; font-size: 1.1rem;">Comparing Graph Logic vs. Vector Similarity</p>
        </div>
    """, unsafe_allow_html=True)

    # Chat Container (Scrollable)
    chat_container = st.container()

    # Input Area (Fixed at bottom)
    pending = st.session_state.get("pending_clarification")
    placeholder = "Answer the clarification so I can run the analysis..." if pending else "Ask a complex question about hazards..."
    if prompt := st.chat_input(placeholder):
        process_user_query(prompt)

    # Render History inside the container
    with chat_container:
        if not st.session_state.messages:
            # Welcome State
            st.markdown("#### 💡 Suggested Questions")
            samples = [
                "What is the possible cause of the temperature hazard?",
                "Which controls mitigate high-severity hazards?",
                "What hazards occur during maintenance?",
                "What are the consequences of LN2 hazards?"
            ]
            cols = st.columns(2)
            for i, s in enumerate(samples):
                if cols[i % 2].button(s, key=f"sample_{i}", use_container_width=True):
                    process_user_query(s)

        for msg in st.session_state.messages:
            # User Message
            with st.chat_message("user"):
                st.markdown(msg['question'])

            # Assistant Message (The Comparison)
            with st.chat_message("assistant"):
                render_comparison_result(msg)


def process_user_query(question: str):
    """Run comparison and update history."""

    # 1. Check if engines are loaded
    if not st.session_state.graph_rag_loaded and not st.session_state.trad_rag_loaded:
        st.error(
            "⚠️ Data is not ready. Place files in the `data` directory and restart.")
        return

    # >>> ECHO USER MESSAGE IMMEDIATELY <<<
    with st.chat_message("user"):
        st.markdown(question)

    pending = st.session_state.get("pending_clarification")
    effective_question = question
    clarification_reason = ""
    clarification_context = None
    conversation_context_applied = ""
    conversation_context_reason = ""
    context_confused = False

    # If the previous turn asked for clarification, treat this user turn as the clarifying answer.
    if pending and pending.get("original_question"):
        clarification_context = pending
        effective_question = resolve_clarified_question(
            pending.get("base_question_for_resolution", pending.get("original_question", "")),
            question,
        )
        clarification_reason = pending.get("reason", "")
        st.session_state.pending_clarification = None
    elif st.session_state.get("clarification_mode", True):
        contextualized = contextualize_question_with_history(question)
        normalized_question = contextualized.get("standalone_question") or question
        context_confused = bool(contextualized.get("context_confused"))
        if contextualized.get("used_history") and normalized_question.strip() != question.strip():
            conversation_context_applied = normalized_question
            conversation_context_reason = contextualized.get("reason", "")

        assessment = assess_question_clarity(
            normalized_question,
            context_confused=context_confused,
        )
        if assessment.get("needs_clarification"):
            clarification_payload = build_targeted_clarification_payload(
                normalized_question,
                assessment,
            )
            follow_up = format_clarification_message(clarification_payload)
            reason = clarification_payload.get("reason", "") or assessment.get("reason", "")
            clarification_questions = clarification_payload.get("clarification_questions") or []
            product_options = clarification_payload.get("product_options") or []

            with st.chat_message("assistant"):
                st.info(follow_up)
                if reason:
                    st.caption(f"Why this follow-up: {reason}")
                if conversation_context_applied:
                    st.caption(f"Conversation-aware interpretation: {conversation_context_applied}")

            st.session_state.pending_clarification = {
                "original_question": question,
                "base_question_for_resolution": normalized_question,
                "clarifying_question": follow_up,
                "clarification_questions": clarification_questions,
                "product_options": product_options,
                "reason": reason,
            }
            st.session_state.messages.append(
                {
                    "type": "clarification",
                    "question": question,
                    "contextualized_question": conversation_context_applied,
                    "assistant_message": follow_up,
                    "clarification_questions": clarification_questions,
                    "product_options": product_options,
                    "reason": reason,
                    "graph_rag": None,
                    "trad_rag": None,
                }
            )
            st.rerun()
            return

        effective_question = assessment.get("rewritten_question") or normalized_question

    results = {
        'type': 'analysis',
        'question': question,
        'resolved_question': effective_question if effective_question.strip() != question.strip() else '',
        'contextualized_question': conversation_context_applied,
        'contextualized_reason': conversation_context_reason,
        'clarification_reason': clarification_reason,
        'clarified_from': clarification_context.get("original_question", "") if clarification_context else "",
        'graph_rag': None,
        'trad_rag': None
    }

    # >>> SHOW PROCESSING STATUS INSIDE ASSISTANT BLOCK <<<
    with st.chat_message("assistant"):
        with st.status("Running dual-engine analysis...", expanded=True) as status:
            if clarification_context:
                status.write("🧭 Clarification received. Running with resolved question.")
            elif effective_question.strip() != question.strip():
                status.write("🧭 Rewriting question for retrieval precision.")
            if conversation_context_applied:
                status.write("🧠 Using recent conversation context to resolve references.")

            # --- Graph RAG Execution ---
            if st.session_state.graph_rag_loaded:
                status.write("🕸️ Querying Knowledge Graph...")
                try:
                    cypher, raw_results, answer, chunks = st.session_state.graph_query_engine.query(
                        effective_question)
                    nodes, rels, viz_query = fetch_subgraph_for_query_results(
                        st.session_state.graph_query_engine, cypher)
                    results['graph_rag'] = {
                        'answer': answer,
                        'cypher': cypher,
                        'chunks': chunks,
                        'nodes': nodes,
                        'rels': rels,
                        'viz_query': viz_query
                    }
                except Exception as e:
                    results['graph_rag'] = {'error': str(e)}

            # --- Trad RAG Execution ---
            if st.session_state.trad_rag_loaded:
                status.write("📚 Searching Vector Embeddings...")
                try:
                    debug, hits, answer = st.session_state.trad_query_engine.query(
                        effective_question)
                    results['trad_rag'] = {
                        'answer': answer,
                        'debug': debug,
                        'hits': hits
                    }
                except Exception as e:
                    results['trad_rag'] = {'error': str(e)}

            status.update(label="Analysis Complete",
                          state="complete", expanded=False)

    # Save and Refresh
    st.session_state.messages.append(results)
    st.rerun()


def render_comparison_result(results: Dict):
    """Render the side-by-side comparison block inside the chat."""
    if results.get("type") == "clarification":
        st.info(results.get("assistant_message", "Could you clarify your question?"))
        if results.get("reason"):
            st.caption(f"Why this follow-up: {results['reason']}")
        if results.get("contextualized_question"):
            st.caption(f"Conversation-aware interpretation: {results['contextualized_question']}")
        return

    contextualized_question = (results.get("contextualized_question") or "").strip()
    if contextualized_question:
        st.caption(f"Conversation-aware interpretation: {contextualized_question}")

    resolved_question = (results.get("resolved_question") or "").strip()
    if resolved_question:
        st.caption(f"Resolved query used: {resolved_question}")

    col1, col2 = st.columns(2)

    # --- LEFT COLUMN: Graph RAG ---
    with col1:
        st.markdown("### 🕸️ Graph RAG")
        gr = results.get('graph_rag')

        if not st.session_state.graph_rag_loaded:
            st.info("System not loaded")
        elif not gr:
            st.warning("Processing failed")
        elif 'error' in gr:
            st.error(gr['error'])
        else:
            # The Answer
            st.markdown(gr.get('answer', 'No answer produced'))

            # The Evidence (Graph) - ALWAYS VISIBLE, BIGGER
            nodes = gr.get('nodes', [])
            rels = gr.get('rels', [])
            if nodes:
                st.markdown(f"**📊 Live Graph ({len(nodes)} nodes)**")
                # Increased height to 600px
                render_multiview_component(
                    nodes, rels, height="600px", key_prefix=f"chat_graph_{len(nodes)}_{id(gr)}")

            chunks = gr.get('chunks', [])
            if chunks:
                with st.expander(f"📦 Retrieved Graph Chunks ({len(chunks)})", expanded=False):
                    for i, h in enumerate(chunks[:4]):
                        meta = h.get('meta', {})
                        score = round(h.get('distance', 0), 3)
                        st.markdown(
                            f"**{i+1}. {meta.get('doc_type', 'graph_chunk')}** (Dist: {score})")
                        if meta.get('product_name'):
                            st.caption(f"Product: {meta.get('product_name')}")
                        st.text((h.get('text', '') or '')[:300] + "...")
                        st.divider()

            # The Logic (Cypher)
            with st.expander("🔧 Internal Logic (Cypher)"):
                st.code(gr.get('cypher', ''), language='cypher')
                if gr.get('viz_query'):
                    st.caption("Neo4j Browser Sync Code:")
                    st.code(gr['viz_query'], language='cypher')

    # --- RIGHT COLUMN: Traditional RAG ---
    with col2:
        st.markdown("### 📚 Traditional RAG")
        tr = results.get('trad_rag')

        if not st.session_state.trad_rag_loaded:
            st.info("System not loaded")
        elif not tr:
            st.warning("Processing failed")
        elif 'error' in tr:
            st.error(tr['error'])
        else:
            # The Answer
            st.markdown(tr.get('answer', 'No answer produced'))

            # The Evidence (Docs)
            hits = tr.get('hits', [])
            if hits:
                with st.expander(f"📄 Retrieved Segments ({len(hits)})", expanded=False):
                    for i, h in enumerate(hits[:3]):
                        meta = h.get('meta', {})
                        score = round(h.get('distance', 0), 3)
                        st.markdown(
                            f"**{i+1}. {meta.get('doc_type', 'Doc')}** (Dist: {score})")
                        if meta.get('product_name'):
                            st.caption(f"Product: {meta.get('product_name')}")
                        st.text(h.get('text', '')[:200] + "...")
                        st.divider()

            # The Logic (Debug)
            with st.expander("🔧 Internal Logic"):
                st.text(tr.get('debug', ''))


if __name__ == "__main__":
    main()
