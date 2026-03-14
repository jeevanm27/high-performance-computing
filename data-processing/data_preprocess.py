import pandas as pd
import numpy as np

# load: replace with your filename
df = pd.read_csv("../data/udt/18_Phoenix/02_TransCAD_results/Phoenix_link_LinkFlows.csv", sep=",")  # adjust delimiter

# Ensure numeric columns
# Columns used (A->B): AB_Length, AB_Free_Speed, AB_Free_Time, AB_Time, AB_UE_Time, AB_Flow, AB_Capacity, AB_VDF
# Fallback rules below:

# 1) Distance (meters)
if 'AB_Length' in df:
    distance_m = df['AB_Length'].astype(float)
else:
    raise RuntimeError("AB_Length required")

# 2) Actual travel time (seconds)
# Prefer AB_Time, else AB_UE_Time, else compute from free speed/time
if 'AB_Time' in df and not df['AB_Time'].isna().all():
    time_s = df['AB_Time'].astype(float)
elif 'AB_UE_Time' in df and not df['AB_UE_Time'].isna().all():
    time_s = df['AB_UE_Time'].astype(float)
elif 'AB_Free_Time' in df and not df['AB_Free_Time'].isna().all():
    # AB_Free_Time is likely in same units as AB_Time
    time_s = df['AB_Free_Time'].astype(float)
elif 'AB_Free_Speed' in df:
    # compute: length (km) / speed (km/h) -> hours -> seconds
    time_s = (df['AB_Length'].astype(float)/1000.0) / df['AB_Free_Speed'].astype(float) * 3600.0
else:
    raise RuntimeError("No time info found (AB_Time / AB_UE_Time / AB_Free_Time / AB_Free_Speed).")

# 3) Congestion ratio: prefer AB_VOC, else compute AB_Flow/AB_Capacity, else compute delay ratio
if 'AB_VOC' in df and not df['AB_VOC'].isna().all():
    cong = df['AB_VOC'].astype(float)
elif 'AB_Flow' in df and 'AB_Capacity' in df:
    # Avoid zero division
    cap = df['AB_Capacity'].astype(float).replace(0, np.nan).fillna(1.0)
    cong = df['AB_Flow'].astype(float) / cap
else:
    # fallback to delay ratio
    if 'AB_Free_Time' in df:
        free_t = df['AB_Free_Time'].astype(float).replace(0, np.nan).fillna(1.0)
        cong = (time_s - free_t) / free_t
    else:
        cong = np.ones(len(df))  # degenerate fallback

# Make sure all are non-negative
distance_m = np.maximum(0.0, distance_m)
time_s = np.maximum(0.0, time_s)
cong = np.maximum(0.0, cong)

# Scale each objective to avoid domination by magnitude differences
def scale_to_target(x, target_median=100.0):
    med = np.median(x[x>0]) if np.any(x>0) else 1.0
    scale = target_median / med if med>0 else 1.0
    xi = np.round(np.maximum(1, x * scale)).astype(int)
    return xi

w1 = scale_to_target(distance_m, target_median=100)
w2 = scale_to_target(time_s, target_median=100)
w3 = scale_to_target(cong, target_median=100)

# --- Build contiguous node id mapping ---
if 'AB_From_Node_' not in df or 'AB_To_Node_ID' not in df:
    raise RuntimeError("Missing node id columns AB_From_Node_ / AB_To_Node_ID")

all_nodes = pd.unique(df[['AB_From_Node_', 'AB_To_Node_ID']].values.ravel())
all_nodes_sorted = np.sort(all_nodes)
id_map = {orig: idx for idx, orig in enumerate(all_nodes_sorted)}
with open("node_id_map_debug.txt", "w") as f:
    for orig, idx in id_map.items():
        f.write(f"{orig} -> {idx}\n")

mapped_u = df['AB_From_Node_'].map(id_map).astype(int)
mapped_v = df['AB_To_Node_ID'].map(id_map).astype(int)

num_vertices = len(all_nodes_sorted)
num_edges = len(df)

print(f"Vertices: {num_vertices}")
print(f"Edges: {num_edges}")

# --- Write output CSV with header (compatible with C reader expecting header) ---
with open("graph_edges.csv", "w") as out:
    out.write("u,v,w1,w2,w3\n")
    for i in range(num_edges):
        out.write(f"{mapped_u[i]},{mapped_v[i]},{w1[i]},{w2[i]},{w3[i]}\n")

# Optional: save mapping for reference
map_df = pd.DataFrame({"original_id": all_nodes_sorted, "new_id": np.arange(num_vertices)})
map_df.to_csv("node_id_mapping.csv", index=False)

print("Wrote graph_edges.csv and node_id_mapping.csv")
