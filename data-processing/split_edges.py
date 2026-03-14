import random
import networkx as nx
import csv
import os
from collections import namedtuple
from datetime import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Represents an edge with arbitrary number of weight columns
Edge = namedtuple('Edge', ['u', 'v', 'weights'])  # weights is a list/tuple of values

def normalize(u, v):
    return (u, v) if u <= v else (v, u)


def read_graph_csv(path, delimiter=',', has_header=True):
    G = nx.Graph()
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            u = row[0].strip()
            v = row[1].strip()
            if u == v:
                continue
            weights = [None if c.strip()=='' else c.strip() for c in row[2:]]
            G.add_edge(u, v, _weights=tuple(weights))  # store as tuple for hashing
    print(f"Loaded graph from {path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def read_available_edges_csv(path, delimiter=',', has_header=True):
    available = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            u = row[0].strip()
            v = row[1].strip()
            if u == v:
                continue
            weights = [None if c.strip()=='' else c.strip() for c in row[2:]]
            key = normalize(u, v)
            available[key] = Edge(u, v, weights)
    print(f"Loaded {len(available)} available weighted edges from {path}")
    return available


def save_operation_csv(edges_with_weights, filename):
    os.makedirs("data_dynamic", exist_ok=True)
    path = os.path.join("data_dynamic", filename)
    if not edges_with_weights:
        return
    # infer max number of weight columns
    max_w = max(len(ew[2]) for ew in edges_with_weights)
    header = ['u', 'v'] + [f'w{i+1}' for i in range(max_w)]
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for u, v, weights in edges_with_weights:
            row = [u, v] + list(weights) + [''] * (max_w - len(weights))
            writer.writerow(row)
    print(f"Saved {len(edges_with_weights)} edges → {path}")


def insert_edges(graph: nx.Graph, available: dict, n=1000, seed=None, auto_save=True, filename=None):
    if seed is not None:
        random.seed(seed)
    if len(available) < n:
        raise ValueError(f"Only {len(available)} available edges, need {n}")

    candidates = list(available.keys())
    random.shuffle(candidates)

    inserted = []
    count = 0
    for key in candidates:
        if count >= n:
            break
        u, v = key
        if graph.has_edge(u, v):
            continue  # should never happen, but safe
        edge_obj = available.pop(key)
        graph.add_edge(u, v, _weights=edge_obj.weights)
        inserted.append((edge_obj.u, edge_obj.v, edge_obj.weights))
        count += 1

    if auto_save:
        save_operation_csv(inserted, filename)

    # print(f"Inserted {len(inserted)} weighted edges → graph now has {graph.number_of_edges()} edges")
    return inserted



def delete_edges_keep_connected(graph: nx.Graph, available: dict, n=1000, seed=None, auto_save=True, filename=None):
    
    if seed is not None:
        random.seed(seed)
    
    deleted = []
    removed_count = 0

    bridges = set(normalize(u, v) for u, v in nx.bridges(graph))
    safe_edges = []
    for u, v in graph.edges():
        if normalize(u, v) not in bridges:
            safe_edges.append((u, v))

    while removed_count < n:
         
        if not safe_edges:
            print("No more non-bridge edges left → stopping early (graph is a tree or close)")
            break

        u, v = random.choice(safe_edges)
        weights = graph[u][v].get('_weights', ())
        graph.remove_edge(u, v)

        if nx.is_connected(graph):
            # Safe! Keep it removed
            key = normalize(u, v)
            available[key] = Edge(u, v, weights)
            deleted.append((u, v, weights))
            removed_count += 1
            safe_edges.remove((u, v))
        else:
            graph.add_edge(u, v, _weights=weights)
            print("    Hit a critical edge → refreshing bridge set (rare)")
            bridges = set(normalize(a, b) for a, b in nx.bridges(graph))
            safe_edges = []
            for u, v in graph.edges():
                if normalize(u, v) not in bridges:
                    safe_edges.append((u, v))
            
        print(f"    Safely deleted {removed_count}/{n} edges → {graph.number_of_edges()} left", end='\r')
    print()
    
    if auto_save:
        save_operation_csv(deleted, filename)
        
    return deleted


# ===================================================================
# Example usage
# ===================================================================

if __name__ == "__main__":
      

    OUTPUT_DIR = "./data_dynamic"       # Output directory
    ALT_DIR = "./alt_batches"
    INS_DIR = "./insertion_batches"
    DEL_DIR = "./deletion_batches"
    ensure_dir(OUTPUT_DIR)
    ensure_dir(f'{OUTPUT_DIR}/{ALT_DIR}')
    ensure_dir(f'{OUTPUT_DIR}/{INS_DIR}')
    ensure_dir(f'{OUTPUT_DIR}/{DEL_DIR}')
    # Load once
    G = read_graph_csv(f"{OUTPUT_DIR}/base_60.csv")
    available = read_available_edges_csv(f"{OUTPUT_DIR}/remaining_40.csv")

    for i in range(100):
        delete_edges_keep_connected(G, available, n=100, filename=f"{ALT_DIR}/alt_delete_{i+1}.csv")
        insert_edges(G, available, n=100, filename=f"{ALT_DIR}/alt_insert_{i+1}.csv")

    G = read_graph_csv(f"{OUTPUT_DIR}/base_60.csv")
    available = read_available_edges_csv(f"{OUTPUT_DIR}/remaining_40.csv")
    for i in range(10):
        delete_edges_keep_connected(G, available, n=500, filename=f"{DEL_DIR}/del_batches_{i+1}.csv")    
    
    
    G = read_graph_csv(f"{OUTPUT_DIR}/base_60.csv")
    available = read_available_edges_csv(f"{OUTPUT_DIR}/remaining_40.csv")
    for i in range(10):
        insert_edges(G, available, n=500, filename=f"{INS_DIR}/ins_batches_{i+1}.csv")
    
    print("\nAll done. Final graph and remaining available edges saved.")