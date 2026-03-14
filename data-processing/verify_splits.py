#!/usr/bin/env python3
import csv
import glob
import os
import re
import sys
import networkx as nx

# Edit these parameters if needed
GS_DIR = "./data_dynamic"  # Graph splits directory

def norm(u, v):
    return (u, v) if u <= v else (v, u)


def read_edge_pairs(path):
    with open(path, 'r', newline='') as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return []
    first = rows[0]
    has_header = False
    if len(first) >= 2 and any(c.isalpha() for c in ''.join(first)):
        has_header = True
    data = rows[1:] if has_header else rows
    edges = []
    for r in data:
        if len(r) < 2:
            continue
        u = r[0].strip()
        v = r[1].strip()
        edges.append(norm(u, v))
    return edges


def build_graph(edges):
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v)
    return G


def check_no_duplicates(edges):
    seen = set()
    dup = 0
    for e in edges:
        if e in seen:
            dup += 1
        else:
            seen.add(e)
    return dup == 0, dup


def sorted_by_index(pattern):
    files = glob.glob(pattern)
    def idx(path):
        m = re.search(r"(\d+)(?=\.csv$)", os.path.basename(path))
        return int(m.group(1)) if m else -1
    return sorted(files, key=idx)


# 1) Base checks
base_path = os.path.join(GS_DIR, 'base_60.csv')
base_edges = read_edge_pairs(base_path)
ok, dup = check_no_duplicates(base_edges)
if not ok:
    print(f"ERROR: base has {dup} duplicate undirected edges", file=sys.stderr)
    sys.exit(1)
G_base = build_graph(base_edges)
if not nx.is_connected(G_base):
    print("ERROR: base graph is not connected", file=sys.stderr)
    sys.exit(1)
print(f"Base OK: {len(base_edges)} edges, connected, no duplicates")


# 2) Alternative insert/delete iterations
G_alt = G_base.copy()
seen_edges = set(base_edges)
del_files = sorted_by_index(os.path.join(GS_DIR, 'alt_batches/alt_delete_*.csv'))
ins_files = sorted_by_index(os.path.join(GS_DIR, 'alt_batches/alt_insert_*.csv'))

for i, (ins_path, del_path) in enumerate(zip(ins_files, del_files), start=1):
    ins_edges = read_edge_pairs(ins_path)
    del_edges = read_edge_pairs(del_path)

    for e in del_edges:
        if e not in seen_edges:
            print(f"ERROR: alt set {i} tries to delete missing edge {e}", file=sys.stderr)
            sys.exit(1)
        seen_edges.remove(e)
        if G_alt.has_edge(*e):
            G_alt.remove_edge(*e)

    if not nx.is_connected(G_alt):
        print(f"ERROR: graph disconnected after alt set {i} - while removing {e}", file=sys.stderr)
        G_alt.add_edge(*e)  # revert

    dup_on_insert = 0
    for e in ins_edges:
        if e in seen_edges:
            dup_on_insert += 1
        else:
            seen_edges.add(e)
            G_alt.add_edge(*e)
    if dup_on_insert:
        print(f"ERROR: alt set {i} has {dup_on_insert} duplicate insertions", file=sys.stderr)
        sys.exit(1)

    ok, dup = check_no_duplicates(list(seen_edges))
    if not ok:
        print(f"ERROR: duplicates found after alt set {i}", file=sys.stderr)
        sys.exit(1)
    print(f"Alt set {i} OK: +{len(ins_edges)} / -{len(del_edges)} → edges: {G_alt.number_of_edges()}")


# 3) Deletion batches on a fresh base
G_del = G_base.copy()
seen_edges_del = set(base_edges)
del_dir = os.path.join(GS_DIR, 'deletion_batches')
del_batch_files = sorted_by_index(os.path.join(del_dir, 'del_batches_*.csv'))

for i, path in enumerate(del_batch_files, start=1):
    dels = read_edge_pairs(path)
    for e in dels:
        if e in seen_edges_del:
            seen_edges_del.remove(e)
            if G_del.has_edge(*e):
                G_del.remove_edge(*e)
        else:
            print(f"ERROR: deletion batch {i} tries to delete missing edge {e}", file=sys.stderr)
            sys.exit(1)
    if not nx.is_connected(G_del):
        print(f"ERROR: graph disconnected after deletion batch {i}", file=sys.stderr)
        sys.exit(1)
    print(f"Deletion batch {i} OK: -{len(dels)} → edges: {G_del.number_of_edges()}")


# 4) Insertion batches on a fresh base
G_ins = G_base.copy()
seen_edges_ins = set(base_edges)
ins_dir = os.path.join(GS_DIR, 'insertion_batches')
ins_batch_files = sorted_by_index(os.path.join(ins_dir, 'ins_batches_*.csv'))

for i, path in enumerate(ins_batch_files, start=1):
    ins = read_edge_pairs(path)
    dup = 0
    for e in ins:
        if e in seen_edges_ins:
            dup += 1
        else:
            seen_edges_ins.add(e)
            G_ins.add_edge(*e)
    if dup:
        print(f"ERROR: insertion batch {i} contains {dup} duplicate edges", file=sys.stderr)
        sys.exit(1)
    print(f"Insertion batch {i} OK: +{len(ins)} → edges: {G_ins.number_of_edges()}")

print("All checks passed.")

