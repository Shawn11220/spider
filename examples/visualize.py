import spider
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sentence_transformers import SentenceTransformer
import argparse
import json

DATABASE_FILE = "spider_graph.db"

def create_enhanced_visualization(
    db_path=DATABASE_FILE,
    output_file="enhanced_graph.html",
    search_query=None,
    highlight_clusters=None,
    min_significance=0,
    show_edge_weights=True
):
    """
    Create an enhanced interactive graph visualization with multiple features.
    
    Args:
        db_path: Path to the Spider database
        output_file: Output HTML file
        search_query: Optional search query to highlight results
        highlight_clusters: Optional list of cluster IDs to highlight
        min_significance: Minimum significance to display (0-9)
        show_edge_weights: Whether to show edge weights (requires computation)
    """
    
    print("🕸️  Loading Spider database...")
    db = spider.SpiderDB(db_path)
    
    # Get all graph data
    nodes, edges = db.get_all_graph_data()
    
    # Store full content for each node
    node_contents = {}
    for node_id, _, _, _ in nodes:
        full_content = db.get_node(node_id)
        if full_content:
            node_contents[node_id] = full_content
    
    # Perform search if query provided
    search_results = []
    if search_query:
        print(f"🔍 Searching for: '{search_query}'")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        query_vec = encoder.encode(search_query).tolist()
        search_results = db.hybrid_search(query_vec, k=10, ef_search=100)
        print(f"   Found {len(search_results)} results")
    
    # Get cluster statistics
    cluster_stats = db.get_cluster_stats()
    if cluster_stats:
        total_clusters, avg_size, avg_sig = cluster_stats
        print(f"📊 Clusters: {total_clusters}, Avg Size: {avg_size:.1f}, Avg Significance: {avg_sig:.2f}")
    
    # Initialize PyVis Network with enhanced options
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="white",
        select_menu=True,
        filter_menu=True,
    )
    
    # Enhanced physics for better clustering
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.001,
        damping=0.09,
        overlap=0
    )
    
    # Generate cluster colors with distinct, vibrant hues
    unique_clusters = sorted(list(set(n[3] for n in nodes if n[3] is not None)))
    cluster_colors = {}
    
    if unique_clusters:
        # Use HSL-based colors for maximum visual distinction
        num_clusters = len(unique_clusters)
        for i, cid in enumerate(unique_clusters):
            # Distribute hues evenly across the spectrum
            hue = i / num_clusters
            # Use high saturation and medium lightness for vibrant colors
            saturation = 0.75
            lightness = 0.55
            # Convert HSL to RGB
            import colorsys
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            cluster_colors[cid] = mcolors.to_hex(rgb)
    
    print(f"🎨 Generated {len(cluster_colors)} distinct cluster colors")
    
    # Create search result lookup
    search_node_ids = {node_id for node_id, _ in search_results}
    search_scores = {node_id: score for node_id, score in search_results}
    
    # Create highlight cluster lookup
    highlight_set = set(highlight_clusters) if highlight_clusters else set()
    
    print(f"📍 Adding {len(nodes)} nodes...")
    
    # Add nodes with enhanced styling
    for node_id, label, significance, cluster_id in nodes:
        # Filter by significance
        if significance < min_significance:
            continue
        
        # Determine node color
        is_search_result = node_id in search_node_ids
        is_highlighted_cluster = cluster_id in highlight_set if cluster_id else False
        
        if is_search_result:
            # Search results in bright yellow/gold gradient
            rank = list(search_node_ids).index(node_id)
            intensity = 1.0 - (rank / len(search_node_ids)) * 0.5
            color = mcolors.to_hex((1.0, intensity * 0.8, 0.0))
            border_color = "#FFD700"
            border_width = 4
        elif is_highlighted_cluster:
            # Highlighted clusters in bright cyan
            color = "#00FFFF"
            border_color = "#FFFFFF"
            border_width = 3
        else:
            # Normal cluster coloring
            color = cluster_colors.get(cluster_id, "#808080")
            border_color = "#555555"
            border_width = 1
        
        # Scale size by significance (bigger = more significant)
        base_size = significance * 3 + 15
        
        # Boost size for search results
        if is_search_result:
            base_size *= 1.5
        
        # Build rich tooltip (plain text since PyVis escapes HTML)
        tooltip_parts = [
            f"Node {node_id}",
            f"Cluster: {cluster_id if cluster_id else 'None'}",
            f"Significance: {significance}/9",
        ]
        
        if is_search_result:
            score = search_scores.get(node_id, 0)
            tooltip_parts.insert(1, f"🎯 Search Score: {score:.3f}")
        
        # Add life score
        life_score = db.calculate_life_score(node_id)
        tooltip_parts.append(f"Life Score: {life_score:.2f}")
        
        # Add content preview with proper escaping
        content_preview = node_contents.get(node_id, label)[:100]
        tooltip_parts.append(f"---\n{content_preview}...")
        
        title = "\n".join(tooltip_parts)
        
        # Create label (show snippet of actual text)
        label_text = content_preview[:30] if len(content_preview) > 30 else content_preview
        display_label = f"{node_id}: {label_text}..."
        if is_search_result:
            display_label = f"⭐ {node_id}: {label_text}..."
        
        net.add_node(
            node_id,
            label=display_label,
            title=title,
            color={
                'background': color,
                'border': border_color,
                'highlight': {
                    'background': '#FF6B6B',
                    'border': '#FF0000'
                }
            },
            size=base_size,
            borderWidth=border_width,
            font={'size': 14 if is_search_result else 12}
        )
    
    print(f"🔗 Adding {len(edges)} edges...")
    
    # Calculate edge weights if needed (cosine similarity between connected nodes)
    if show_edge_weights:
        # This would require embedding data - simplified version
        for source, target in edges:
            # Skip if nodes were filtered out
            if source >= len(nodes) or target >= len(nodes):
                continue
            
            # Check if both nodes pass significance filter
            source_sig = nodes[source][2]
            target_sig = nodes[target][2]
            
            if source_sig < min_significance or target_sig < min_significance:
                continue
            
            # Highlight edges connected to search results
            if source in search_node_ids or target in search_node_ids:
                edge_color = "rgba(255, 215, 0, 0.6)"  # Gold
                edge_width = 3
            else:
                edge_color = "rgba(100, 100, 100, 0.3)"  # Gray
                edge_width = 1
            
            net.add_edge(
                source,
                target,
                color=edge_color,
                width=edge_width,
                smooth={'type': 'continuous'}
            )
    else:
        # Simple edges
        for source, target in edges:
            if source >= len(nodes) or target >= len(nodes):
                continue
            
            source_sig = nodes[source][2]
            target_sig = nodes[target][2]
            
            if source_sig < min_significance or target_sig < min_significance:
                continue
            
            net.add_edge(source, target, color="rgba(85, 85, 85, 0.3)")
    
    # Add custom buttons and controls
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 150,
                "springConstant": 0.001,
                "damping": 0.09,
                "avoidOverlap": 0.1
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true
        },
        "nodes": {
            "shape": "dot",
            "font": {
                "size": 12,
                "color": "#ffffff"
            },
            "scaling": {
                "min": 10,
                "max": 50
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous"
            },
            "arrows": {
                "to": {
                    "enabled": false
                }
            }
        }
    }
    """)
    
    # Save with custom HTML enhancements
    html = net.generate_html()
    
    # Add custom controls and info panel
    custom_html = f"""
    <style>
        #info-panel {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            border: 2px solid #333;
        }}
        #info-panel h3 {{
            margin: 0 0 10px 0;
            color: #00ff00;
        }}
        #info-panel .stat {{
            margin: 5px 0;
        }}
        #legend {{
            position: fixed;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 11px;
            z-index: 1000;
            border: 2px solid #333;
        }}
        .legend-item {{
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
            border: 1px solid #555;
        }}
    </style>
    
    <div id="legend">
        <b>Legend:</b>
        <div class="legend-item">
            <div class="legend-color" style="background: #FFD700; border-color: #FFD700;"></div>
            <span>Search Results</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #00FFFF;"></div>
            <span>Highlighted Clusters</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #808080;"></div>
            <span>Unclustered Nodes</span>
        </div>
        <hr style="border-color: #555; margin: 10px 0;">
        <b>Clusters:</b>
        {''.join(f'''<div class="legend-item"><div class="legend-color" style="background: {color};"></div><span>Cluster {cid}</span></div>''' for cid, color in cluster_colors.items())}
    </div>
    """
    
    # Insert custom HTML before closing body tag
    html = html.replace('</body>', f'{custom_html}</body>')
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Enhanced visualization saved to {output_file}")
    print(f"   Open in browser to explore!")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Enhanced Spider Graph Visualization')
    parser.add_argument('--db', default=DATABASE_FILE, help='Database file path')
    parser.add_argument('--output', default='enhanced_graph.html', help='Output HTML file')
    parser.add_argument('--search', help='Search query to highlight results')
    parser.add_argument('--clusters', help='Comma-separated cluster IDs to highlight')
    parser.add_argument('--min-sig', type=int, default=0, help='Minimum significance (0-9)')
    parser.add_argument('--no-edge-weights', action='store_true', help='Disable edge weight visualization')
    
    args = parser.parse_args()
    
    highlight_clusters = None
    if args.clusters:
        highlight_clusters = [int(c.strip()) for c in args.clusters.split(',')]
    
    create_enhanced_visualization(
        db_path=args.db,
        output_file=args.output,
        search_query=args.search,
        highlight_clusters=highlight_clusters,
        min_significance=args.min_sig,
        show_edge_weights=not args.no_edge_weights
    )


if __name__ == "__main__":
    main()