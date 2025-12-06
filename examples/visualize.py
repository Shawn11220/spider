import spider
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DATABASE_FILE = "spider_graph.db"

db = spider.SpiderDB(DATABASE_FILE)

nodes, edges = db.get_all_graph_data()

# Initialize PyVis Network
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)

# Generate Colors
unique_clusters = sorted(list(set(n[3] for n in nodes if n[3] is not None)))
if unique_clusters:
    # Generate distinct hex colors
    cmap = cm.get_cmap('rainbow', len(unique_clusters))
    cluster_colors = {}
    for i, cid in enumerate(unique_clusters):
        rgba = cmap(i)
        cluster_colors[cid] = mcolors.to_hex(rgba)
else:
    cluster_colors = {}

# Add Nodes
for node_id, label, significance, cluster_id in nodes:
    color = cluster_colors.get(cluster_id, "#808080") # Default grey
    # Scale size
    size = significance * 5 + 10
        
    # Tooltip with full text
    title = f"ID: {node_id}\nCluster: {cluster_id}\nSig: {significance}\n\n{label}"
        
    net.add_node(node_id, label=str(node_id), title=title, color=color, size=size)

# Add Edges
for source, target in edges:
    net.add_edge(source, target, color="#555555")

# Physics Options
net.barnes_hut()
    
output_file = "graph_visualization.html"
net.save_graph(output_file)
print(f"✅ Saved interactive visualization to {output_file}")