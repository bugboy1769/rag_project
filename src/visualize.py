import matplotlib.pyplot as plt
import networkx as nx

def draw_graph(g_store, output_path="graph_debug.png"):
    """
    Renders the graph using Matplotlib for debugging.
    """
    plt.figure(figsize=(12, 8))
    G = g_store.graph
    
    if not G.nodes:
        print("Visualization: Graph is empty.")
        return

    # Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.8)
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='gray')
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    plt.title("Knowledge Graph Debug View")
    plt.axis('off')
    
    print(f"Saving graph visualization to {output_path}...")
    plt.savefig(output_path)
    plt.close()
