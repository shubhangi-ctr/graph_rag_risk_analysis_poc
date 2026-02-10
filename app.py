"""
Combined RAG Application - Graph RAG vs Traditional RAG Side-by-Side Comparison
"""
from traditional_rag import query_engine as trad_query_engine
from traditional_rag.vector_store import ChromaVectorStore, build_rag_documents
from graph_rag import query_engine as graph_query_engine
from graph_rag.graph_loader import GraphLoader, load_to_neo4j
import config
from excel_parser import parse_excel, ParsedData
import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import tempfile
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import zipfile
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
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


def init_session_state():
    defaults = {
        'messages': [],
        'graph_rag_loaded': False,
        'trad_rag_loaded': False,
        'graph_query_engine': None,
        'trad_query_engine': None,
        'vector_store': None,
        'graph_stats': None,
        'trad_stats': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


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

            subgraph_query = f"""
            MATCH (anchor)
            WHERE anchor.id IN {ids_formatted} OR anchor.name IN {ids_formatted}
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


# ==================== SIDEBAR & DATA LOADING ====================

def render_sidebar():
    with st.sidebar:
        st.title("⚖️ RAG Comparison")
        st.caption("Graph RAG vs Traditional RAG")
        st.divider()

        st.header("📁 Upload Data")
        uploaded = st.file_uploader(
            "Risk Analysis Excel", type=['xlsx', 'xls', 'zip'])
        # uploaded = st.file_uploader("Upload Zip containing Excel files", type=['zip'])

        if uploaded:
            clear = st.checkbox("Clear existing data", value=True)

            if st.button("🚀 Load Both RAG Systems", type="primary", use_container_width=True):
                # 2. Extract and process the zip file
                with zipfile.ZipFile(uploaded) as z:
                    # Filter for only excel files inside the zip
                    excel_files = [f for f in z.namelist(
                    ) if f.endswith(('.xlsx', '.xls'))]

                    if not excel_files:
                        st.error("No Excel files found in the ZIP.")
                    else:
                        for file_name in excel_files:
                            with z.open(file_name) as f:
                                # We wrap in BytesIO so the RAG loaders treat it like a file object
                                file_content = io.BytesIO(f.read())
                                file_content.name = file_name  # Preserve filename for metadata

                                st.write(f"Processing: {file_name}...")
                                load_graph_rag(file_content, config.NEO4J_URI, config.NEO4J_USER,
                                               config.NEO4J_PASSWORD, config.OPENAI_API_KEY, clear)
                                load_traditional_rag(
                                    file_content, config.OPENAI_API_KEY, clear)

                                # After the first file is loaded, we don't want to 'clear' the DB anymore
                                # or we will wipe the previous file's data
                                clear = False
                        st.success("All files from ZIP loaded!")

            # Optional: Add single-RAG loading logic here using the same loop as above

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            status = "✅" if st.session_state.get(
                'graph_rag_loaded', False) else "❌"
            st.metric("Graph RAG", status)
        with col2:
            status = "✅" if st.session_state.get(
                'trad_rag_loaded', False) else "❌"
            st.metric("Trad RAG", status)

        st.divider()
        st.header("🎨 Legend")
        # Ensure NODE_COLORS is defined globally or imported
        for node_type, color in NODE_COLORS.items():
            st.markdown(
                f"<span style='color:{color};font-size:16px'>●</span> {node_type}", unsafe_allow_html=True)


def load_graph_rag(uploaded, uri, user, pwd, openai_key, clear):
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        tmp.write(uploaded.getvalue())
        tmp.close()

        with st.sidebar:
            progress = st.progress(0, text="Parsing Excel...")
            data = parse_excel(tmp.name)
            progress.progress(30, text="Loading to Neo4j...")

            stats = load_to_neo4j(data, uri=uri, user=user, password=pwd, clear_existing=clear,
                                  progress_callback=lambda p, m: progress.progress(min(30 + int(p*60), 90), text=m))

            progress.progress(95, text="Initializing query engine...")
            qe = graph_query_engine.QueryEngine(uri, user, pwd, openai_key)
            qe.connect()

            st.session_state.graph_query_engine = qe
            st.session_state.graph_rag_loaded = True
            st.session_state.graph_stats = stats

            progress.progress(100, text="✅ Graph RAG Ready!")
            st.toast(
                f"Graph RAG Loaded: {stats.get('hazards', 0)} hazards", icon="🕸️")
    except Exception as e:
        st.sidebar.error("Graph RAG Error: " + str(e))
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except:
                pass


def load_traditional_rag(uploaded, openai_key, clear):
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        tmp.write(uploaded.getvalue())
        tmp.close()

        with st.sidebar:
            progress = st.progress(0, text="Parsing Excel...")
            data = parse_excel(tmp.name)

            progress.progress(20, text="Building documents...")
            docs, metas, stats = build_rag_documents(
                data, source_name=uploaded.name)

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
            st.toast(f"Traditional RAG Loaded: {added} docs", icon="📚")
    except Exception as e:
        st.sidebar.error("Traditional RAG Error: " + str(e))
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except:
                pass


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
    if prompt := st.chat_input("Ask a complex question about hazards..."):
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
        st.error("⚠️ Please load data first using the sidebar.")
        return

    # >>> ECHO USER MESSAGE IMMEDIATELY <<<
    with st.chat_message("user"):
        st.markdown(question)

    results = {'question': question, 'graph_rag': None, 'trad_rag': None}

    # >>> SHOW PROCESSING STATUS INSIDE ASSISTANT BLOCK <<<
    with st.chat_message("assistant"):
        with st.status("Running dual-engine analysis...", expanded=True) as status:

            # --- Graph RAG Execution ---
            if st.session_state.graph_rag_loaded:
                status.write("🕸️ Querying Knowledge Graph...")
                try:
                    cypher, raw_results, answer = st.session_state.graph_query_engine.query(
                        question)
                    nodes, rels, viz_query = fetch_subgraph_for_query_results(
                        st.session_state.graph_query_engine, cypher)
                    results['graph_rag'] = {
                        'answer': answer,
                        'cypher': cypher,
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
                        question)
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
                        st.text(h.get('text', '')[:200] + "...")
                        st.divider()

            # The Logic (Debug)
            with st.expander("🔧 Internal Logic"):
                st.text(tr.get('debug', ''))


if __name__ == "__main__":
    main()
