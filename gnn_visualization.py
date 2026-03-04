from pyvis.network import Network
import random

def visualize_gnn_graph(sample, output_file="gnn_graph.html", max_2d_nodes=None):
    net = Network(height="750px", width="100%", bgcolor="#07090f", font_color="white")
    net.barnes_hut()

    coupled_2d = set(sample.edge_index_1d_2d[1].tolist())
    coupled_1d = set(sample.edge_index_1d_2d[0].tolist())
    total_2d = sample.x.size(0)

    # Determine which 2D nodes to render
    if max_2d_nodes is None or max_2d_nodes >= total_2d:
        visible_2d = set(range(total_2d))
    else:
        non_coupled = [i for i in range(total_2d) if i not in coupled_2d]
        n_sample = max(0, max_2d_nodes - len(coupled_2d))
        visible_2d = coupled_2d | set(random.sample(non_coupled, min(n_sample, len(non_coupled))))

    # 2D nodes
    for i in visible_2d:
        color = "#a855f7" if i in coupled_2d else "#00d4ff"
        net.add_node(
            f"2d_{i}",
            label=f"2D-{i}" if i in coupled_2d else "",
            color=color,
            size=10,
            title=f"2D Node {i}<br>Features shape: {list(sample.x[i].shape)}"
        )

    # 1D nodes (always render all — typically small)
    for i in range(sample.x_1d.size(0)):
        color = "#f59e0b" if i in coupled_1d else "#ff6b35"
        net.add_node(
            f"1d_{i}",
            label=f"1D-{i}",
            color=color,
            size=16,
            title=f"1D Node {i}<br>Features shape: {list(sample.x_1d[i].shape)}"
        )

    # 2D-2D edges (only if both endpoints are visible)
    for src, dst in sample.edge_index.t().tolist():
        if src in visible_2d and dst in visible_2d:
            net.add_edge(f"2d_{src}", f"2d_{dst}", color="#00d4ff33", width=1)

    # 1D-1D edges
    for src, dst in sample.edge_index_1d.t().tolist():
        net.add_edge(f"1d_{src}", f"1d_{dst}", color="#ff6b3588", width=2)

    # Coupling edges (only if the 2D endpoint is visible)
    for src, dst in sample.edge_index_1d_2d.t().tolist():
        if dst in visible_2d:
            net.add_edge(f"1d_{src}", f"2d_{dst}", color="#22c55e", width=3,
                         dashes=True, title="1D↔2D coupling")

    net.show_buttons(filter_=["physics"])
    net.save_graph(output_file)