import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tqdm

# Bulid similarity graph from vector dataframe


def build_graph_from_vec_df(vec_df, vec_column, weight_from_sim, node_column=None, node_attrs=None):
    edges = []

    G = nx.Graph()

    for i in tqdm.tqdm(range(len(vec_df))):
        for j in range(i + 1, len(vec_df)):
            sim = np.dot(vec_df[vec_column].iloc[i],
                         vec_df[vec_column].iloc[j])
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
        if G.has_edge(a, b):
            G[a][b]['weight'] += w
        else:
            G.add_edge(a, b, weight=w)

    if node_attrs is not None:
        for i, row in vec_df.iterrows():
            if node_column is not None:
                node_name = row[node_column]
            else:
                node_name = str(i)

            if node_name in G.nodes():
                for attr in node_attrs:
                    if attr in row:
                        G.nodes[node_name][attr] = row[attr]

    return G


def find_shortest_path(graph, source, target):
    try:
        path = nx.shortest_path(graph, source=source,
                                target=target, weight='weight')
        length = nx.shortest_path_length(
            graph, source=source, target=target, weight='weight')
        return path, length
    except nx.NetworkXNoPath:
        return None, float("inf")


def plot_graph_plotly(
    G,
    path_nodes=None,
    pos=None,
    attr_keys=("name", "label"),
    title="Graph",
    layout_kwargs=None,
    node_size_by_degree=True,
    width=900,
    height=700,
    layout_algorithm="spring",  # "spring", "kamada_kawai", "spectral", "circular"
):
    """
    G: networkx.Graph
    path_nodes: list of nodes to highlight in order (optional)
    pos: dict {node: (x,y)} to reuse a layout (optional)
    attr_keys: node attributes to include in hover if they exist
    layout_kwargs: args passed to layout algorithm if pos is None
    layout_algorithm: which layout to use for better node spacing
    """

    if pos is None:
        layout_kwargs = layout_kwargs or {}

        # Choose layout algorithm
        if layout_algorithm == "kamada_kawai":
            # Better for spreading nodes, considers edge weights
            pos = nx.kamada_kawai_layout(G, scale=2)
        elif layout_algorithm == "spectral":
            # Good for spreading clusters
            pos = nx.spectral_layout(G, scale=2)
        elif layout_algorithm == "circular":
            # Circular arrangement
            pos = nx.circular_layout(G, scale=2)
        else:  # "spring" (default)
            # Increased k parameter spreads nodes more
            pos = nx.spring_layout(
                G,
                seed=42,
                k=layout_kwargs.get("k", 1.5),  # erhöht von 0.6 auf 1.5
                iterations=layout_kwargs.get(
                    "iterations", 300),  # mehr Iterationen
                scale=2  # größerer Bereich
            )

    # --- edges (no hover for speed) ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
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
        node_x.append(x)
        node_y.append(y)

        # Determine display name: node itself if str, else from attrs
        data = G.nodes[n]
        display_name = None
        if isinstance(n, str):
            display_name = n
        else:
            for k in attr_keys:
                if k in data:
                    display_name = str(data[k])
                    break
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
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=sizes),
        name="nodes"
    )

    traces = [edge_trace, node_trace]

    # --- optional: highlight a path ---
    if path_nodes and len(path_nodes) > 1:
        px, py = [], []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            px += [x0, x1, None]
            py += [y0, y1, None]

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


def plot_graph_plotly_color(
    G,
    path_nodes=None,
    pos=None,
    attr_keys=("name", "label"),
    title="Graph",
    layout_kwargs=None,
    node_size_by_degree=True,
    width=900,
    height=700,
    layout_algorithm="spring",
    color_by=None,
    color_map=None,
):
    """
    G: networkx.Graph
    path_nodes: list of nodes to highlight in order (optional)
    pos: dict {node: (x,y)} to reuse a layout (optional)
    attr_keys: node attributes to include in hover if they exist
    layout_kwargs: args passed to layout algorithm if pos is None
    layout_algorithm: which layout to use for better node spacing
    color_by: node attribute name to color by (e.g., "category", "type")
    color_map: dict mapping category values to colors, or None for automatic
    """

    if pos is None:
        layout_kwargs = layout_kwargs or {}

        if layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, scale=2)
        elif layout_algorithm == "spectral":
            pos = nx.spectral_layout(G, scale=2)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(G, scale=2)
        else:
            pos = nx.spring_layout(
                G,
                seed=42,
                k=layout_kwargs.get("k", 1.5),
                iterations=layout_kwargs.get("iterations", 300),
                scale=2
            )

    # --- edges ---
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scattergl(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1),
        opacity=0.25,
        hoverinfo="skip",
        showlegend=False,
        name="edges"
    )

    traces = [edge_trace]
    degrees = dict(G.degree())

    if color_by:
        categories = set()
        for n in G.nodes():
            if color_by in G.nodes[n] and G.nodes[n][color_by] is not None:
                categories.add(G.nodes[n][color_by])

        if color_map is None:
            default_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            categories_list = sorted(list(categories))
            color_map = {cat: default_colors[i % len(default_colors)]
                         for i, cat in enumerate(categories_list)}

        nodes_by_category = {cat: [] for cat in color_map.keys()}
        nodes_by_category[None] = []

        for n in G.nodes():
            if color_by in G.nodes[n] and G.nodes[n][color_by] is not None:
                cat = G.nodes[n][color_by]
                nodes_by_category[cat].append(n)
            else:
                nodes_by_category[None].append(n)

        for cat, nodes in nodes_by_category.items():
            if not nodes:
                continue

            node_x, node_y, hover_texts, sizes = [], [], [], []
            for n in nodes:
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)

                data = G.nodes[n]
                display_name = None
                if isinstance(n, str):
                    display_name = n
                else:
                    for k in attr_keys:
                        if k in data:
                            display_name = str(data[k])
                            break
                if display_name is None:
                    display_name = str(n)

                extra_lines = []
                for k in attr_keys:
                    if k in data and str(data[k]) != display_name:
                        extra_lines.append(f"{k}: {data[k]}")

                hover_text = f"<b>{display_name}</b><br>degree: {degrees[n]}"
                if extra_lines:
                    hover_text += "<br>" + "<br>".join(extra_lines)
                hover_texts.append(hover_text)

                sizes.append(
                    6 + 2*np.sqrt(degrees[n]) if node_size_by_degree else 8)

            color = color_map.get(
                cat, '#CCCCCC') if cat is not None else '#CCCCCC'
            legend_name = str(cat) if cat is not None else "No category"

            node_trace = go.Scattergl(
                x=node_x, y=node_y,
                mode="markers",
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=0.5, color='white')
                ),
                name=legend_name,
                legendgroup=legend_name,
            )
            traces.append(node_trace)

    else:
        node_x, node_y, hover_texts, sizes = [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)

            data = G.nodes[n]
            display_name = None
            if isinstance(n, str):
                display_name = n
            else:
                for k in attr_keys:
                    if k in data:
                        display_name = str(data[k])
                        break
            if display_name is None:
                display_name = str(n)

            extra_lines = []
            for k in attr_keys:
                if k in data and str(data[k]) != display_name:
                    extra_lines.append(f"{k}: {data[k]}")

            hover_text = f"<b>{display_name}</b><br>degree: {degrees[n]}"
            if extra_lines:
                hover_text += "<br>" + "<br>".join(extra_lines)
            hover_texts.append(hover_text)

            sizes.append(
                6 + 2*np.sqrt(degrees[n]) if node_size_by_degree else 8)

        node_trace = go.Scattergl(
            x=node_x, y=node_y,
            mode="markers",
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(size=sizes),
            name="nodes"
        )
        traces.append(node_trace)

    # --- optional: highlight a path ---
    if path_nodes and len(path_nodes) > 1:
        px, py = [], []
        for a, b in zip(path_nodes[:-1], path_nodes[1:]):
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            px += [x0, x1, None]
            py += [y0, y1, None]

        path_edge = go.Scattergl(
            x=px, y=py, mode="lines",
            line=dict(width=4, color='red'),
            name="shortest path",
            showlegend=True
        )
        pnx = [pos[n][0] for n in path_nodes]
        pny = [pos[n][1] for n in path_nodes]
        path_node = go.Scattergl(
            x=pnx, y=pny, mode="markers",
            marker=dict(size=11, symbol="circle-open-dot",
                        line=dict(width=2, color='red')),
            text=[str(n) for n in path_nodes],
            hovertemplate="%{text}<extra>path</extra>",
            name="path nodes",
            showlegend=False
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


def calculate_category_modularity_scores(G, category_attr='category1'):
    """
    Calculate modularity contribution for each category
    """
    # Group nodes by category
    categories = {}
    for node in G.nodes():
        if category_attr in G.nodes[node] and pd.notna(G.nodes[node][category_attr]):
            cat = G.nodes[node][category_attr]
            if cat not in categories:
                categories[cat] = set()
            categories[cat].add(node)

    total_edges = G.number_of_edges()
    total_weight = sum(d['weight'] for _, _, d in G.edges(data=True))

    results = []

    for category, nodes in categories.items():
        # Edges within community
        internal_edges = 0
        internal_weight = 0
        for u in nodes:
            for v in nodes:
                if G.has_edge(u, v):
                    internal_edges += 1
                    internal_weight += G[u][v]['weight']

        internal_edges = internal_edges // 2  # Each edge counted twice

        # Degree sum of nodes in community
        degree_sum = sum(G.degree(n, weight='weight') for n in nodes)

        # modularity contribution
        modularity_contribution = (
            internal_weight / total_weight) - (degree_sum / (2 * total_weight)) ** 2

        # Intra-community density
        n_nodes = len(nodes)
        max_internal_edges = n_nodes * (n_nodes - 1) / 2
        density = internal_edges / max_internal_edges if max_internal_edges > 0 else 0

        results.append({
            'category': category,
            'nodes': n_nodes,
            'internal_edges': internal_edges,
            'internal_weight': internal_weight,
            'modularity_contribution': modularity_contribution,
            'degree_sum': degree_sum,
            'density': density
        })

    return pd.DataFrame(results).sort_values(by='modularity_contribution', ascending=False)
