import pandas as pd
import networkx as nx
import random
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_graph(csv_path):
    df = pd.read_csv(csv_path, header='infer')
    df.columns = ["u","v","w1","w2","w3"][:df.shape[1]]
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(int(row.u), int(row.v),
                   **{col: row[col] for col in WEIGHT_COLUMNS if col in df.columns})
    return G, df


def is_connected(G):
    is_con = nx.is_connected(G)
    # print("Connected:" , is_con )
    return is_con


def connected_60_40_split(G):
    edges = list(G.edges())
    random.shuffle(edges)

    target = int(len(edges) * 0.6)
    G60 = nx.Graph()
    G60.add_nodes_from(G.nodes())

    for u, v in edges:
        if len(G60.edges()) >= target:
            break
        G60.add_edge(u, v, **G[u][v])
        # keep only if graph stays connected or is growing toward connected
        # later we check fully
    # Ensure actually connected: enforce spanning tree
    if not nx.is_connected(G60):
        # Build minimum spanning tree (structure only)
        T = nx.minimum_spanning_tree(G)
        print("Spanning tree forced.")
        G60 = nx.Graph()
        for u, v in T.edges():
            G60.add_edge(u, v, **G[u][v])

        print("Spanning tree enforced for 60% graph.")
        # Fill remaining edges randomly until reaching target
        remaining_edges = [e for e in edges if e not in T.edges()]
        for u, v in remaining_edges:
            print("Adding edge:", u, v, end='\r')
            if len(G60.edges()) >= target:
                break
            G60.add_edge(u, v, **G[u][v])

    G40_edges = [e for e in G.edges() if e not in G60.edges()]
    return G60, G40_edges


def safe_delete_candidates(G):
    """Edges that are NOT bridges and can be removed while keeping connected."""
    return [e for e in G.edges() if e not in nx.bridges(G)]


def export_graph(G, path):
    with open(path, "w") as f:
        for u, v, data in G.edges(data=True):
            w = [str(data.get(col, 0)) for col in WEIGHT_COLUMNS]
            f.write(f"{u},{v},{','.join(w)}\n")

# ======================================================
# MAIN
# ======================================================

INPUT_CSV = "./graph_edges.csv"     # Your full graph
OUTPUT_DIR = "./data_dynamic"       # Output directory
WEIGHT_COLUMNS = ["w1", "w2", "w3"]  # Weight columns to preserve

ensure_dir(OUTPUT_DIR)
G, df_full = read_graph(INPUT_CSV)
print("Connected?" , is_connected(G))

if not is_connected(G):
    print("ERROR: Input graph is not connected.")
    exit(1)

# Create split
G60, G40 = connected_60_40_split(G)

export_graph(G60, f"{OUTPUT_DIR}/split_60.csv")
print("Split 60-40 done.")
# print(G40)
with open(f"{OUTPUT_DIR}/split_40.csv", "w") as f:
    for (u, v) in G40:
        mask = ((df_full.u == u) & (df_full.v == v)) | ((df_full.u == v) & (df_full.v == u))
        row = df_full[mask].iloc[0]

        w = [str(row[col]) for col in WEIGHT_COLUMNS]
        f.write(f"{u},{v},{','.join(w)}\n")

