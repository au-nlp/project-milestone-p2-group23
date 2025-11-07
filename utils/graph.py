import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def build_graph_from_vec_df(vec_df, vec_column, weight_from_sim, node_column=None):
    edges = []

    G = nx.Graph()

    for i in range(len(vec_df)):
        for j in range(i + 1, len(vec_df)):
            sim = np.dot(vec_df[vec_column].iloc[i], vec_df[vec_column].iloc[j])
            if node_column is not None:
                node1 = vec_df[node_column].iloc[i]
                node2 = vec_df[node_column].iloc[j]
            else:
                node1 = str(i)
                node2 = str(j)
            weight = weight_from_sim(sim)
            if weight < float("inf"):
                edges.append({"from": node1, "to": node2, "weight": weight})

    edges_df = pd.DataFrame(edges)

    for _, row in edges_df.iterrows():
        a, b, w = row["from"], row["to"], row["weight"]
        if node_column is None:
            a, b = int(a), int(b)
        if G.has_edge(a,b):
            G[a][b]['weight'] += w
        else:
            G.add_edge(a, b, weight=w)

    return G

def find_shortest_path(graph, source, target):
    try:
        path = nx.shortest_path(graph, source=source, target=target, weight='weight')
        length = nx.shortest_path_length(graph, source=source, target=target, weight='weight')
        return path, length
    except nx.NetworkXNoPath:
        return None, float("inf")

def plot_graph_plotly(
    G,
    path_nodes=None,
    pos=None,
    attr_keys=("name", "label"),  # which node attrs to show if present
    title="Graph",
    layout_kwargs=None,
    node_size_by_degree=True,
    width=900,
    height=700,
):
    """
    G: networkx.Graph
    path_nodes: list of nodes to highlight in order (optional)
    pos: dict {node: (x,y)} to reuse a layout (optional)
    attr_keys: node attributes to include in hover if they exist
    layout_kwargs: args passed to nx.spring_layout if pos is None
    """

    if pos is None:
        layout_kwargs = layout_kwargs or {}
        # sensible fast defaults
        pos = nx.spring_layout(G, seed=42, k=layout_kwargs.get("k", 0.6),
                               iterations=layout_kwargs.get("iterations", 200))

    # --- edges (no hover for speed) ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scattergl(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        opacity=0.25,
        hoverinfo="skip",
        name="edges"
    )

    # --- nodes with hover ---
    degrees = dict(G.degree())
    node_x, node_y, hover_texts, sizes = [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)

        # Determine display name: node itself if str, else from attrs
        data = G.nodes[n]
        display_name = None
        if isinstance(n, str):
            display_name = n
        else:
            for k in attr_keys:
                if k in data:
                    display_name = str(data[k]); break
        if display_name is None:
            display_name = str(n)

        # Build tooltip lines from existing attributes in attr_keys
        extra_lines = []
        for k in attr_keys:
            if k in data and str(data[k]) != display_name:
                extra_lines.append(f"{k}: {data[k]}")

        hover_text = f"<b>{display_name}</b><br>degree: {degrees[n]}"
        if extra_lines:
            hover_text += "<br>" + "<br>".join(extra_lines)
        hover_texts.append(hover_text)

        sizes.append(6 + 2*np.sqrt(degrees[n]) if node_size_by_degree else 8)

    node_trace = go.Scattergl(
        x=node_x, y=node_y,
        mode="markers",
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",  # hides trace name
        marker=dict(size=sizes),
        name="nodes"
    )

    traces = [edge_trace, node_trace]

    # --- optional: highlight a path ---
    if path_nodes and len(path_nodes) > 1:
        px, py = [], []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            x0, y0 = pos[a]; x1, y1 = pos[b]
            px += [x0, x1, None]; py += [y0, y1, None]

        path_edge = go.Scattergl(
            x=px, y=py, mode="lines",
            line=dict(width=4),
            name="shortest path"
        )
        pnx = [pos[n][0] for n in path_nodes]
        pny = [pos[n][1] for n in path_nodes]
        path_node = go.Scattergl(
            x=pnx, y=pny, mode="markers",
            marker=dict(size=11, symbol="circle-open-dot", line=dict(width=2)),
            text=[str(n) for n in path_nodes],
            hovertemplate="%{text}<extra>path</extra>",
            name="path nodes",
        )
        traces += [path_edge, path_node]

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=40, b=10),
        dragmode="pan",
        width=width, height=height,
    )
    return fig