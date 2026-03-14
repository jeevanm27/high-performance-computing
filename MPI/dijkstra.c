#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include "MOSP.h"
#include <mpi.h>

// Priority queue node for Dijkstra (simple array-based min-heap would be faster)
typedef struct {
    int vertex;
    int dist;
} PQNode;

void swap(PQNode* a, PQNode* b) {
    PQNode t = *a; *a = *b;*b = t;
}

void dijkstra_build_SOSP(SOSPTree* tree, GroupedEdgesArray all_edges, int source, int objective, int num_vertices) {
    bool visited[num_vertices];
    int  distance[num_vertices];
    int  parent[num_vertices];

    // Initialize
    for (int i = 0; i < num_vertices; i++) {
        visited[i] = false;
        distance[i] = INT_MAX;
        parent[i]   = -1;
    }
    distance[source] = 0;

    for (int count = 0; count < num_vertices; count++) {
        // Find unvisited vertex with minimum distance
        int u = -1;
        int min_dist = INT_MAX;

        for (int i = 0; i < num_vertices; i++) {
            if (!visited[i] && distance[i] < min_dist) {
                min_dist = distance[i];
                u = i;
            }
        }
        if (u == -1){
            break;  // all remaining are unreachable
        }

        visited[u] = true;

        // Relax all neighbors of u using the adjacency list
        for (GroupedEdgeNode* node = all_edges[u]; node != NULL; node = node->next) {
            int v = node->source;
            int weight;

            if (objective >= 0){
                weight = node->weight[objective];
            }else{
                weight = node->combinedWeightInt;  // assuming you store this
            }
            if (!visited[v] && distance[u] != INT_MAX) {
                int new_dist = distance[u] + weight;
                if (new_dist < distance[v]) {
                    distance[v] = new_dist;
                    parent[v]   = u;
                }
            }
        }
    }

    // Copy results into tree
    for (int i = 0; i < num_vertices; i++) {
        tree->distance[i] = distance[i];
        tree->parent[i]   = parent[i];
    }
}
 


// Corrected parallel Dijkstra (naive distributed) using owner-based relaxation and broadcasting updates.
void dijkstra_build_SOSP_parallel(SOSPTree* tree, GroupedEdgesArray all_edges, int source, int objective, int num_vertices, MPI_Comm team_comm) {
    int rank, size; MPI_Comm_rank(team_comm, &rank); MPI_Comm_size(team_comm, &size);

    // Partition (contiguous)
    int start = (rank * num_vertices) / size;
    int end   = ((rank + 1) * num_vertices) / size; // [start,end)
    int local_n = end - start;

    // Local arrays (only for owned vertices)
    int *dist_local = (int*)malloc(local_n * sizeof(int));
    int *parent_local = (int*)malloc(local_n * sizeof(int));
    unsigned char *visited_local = (unsigned char*)calloc(local_n, 1);
    for (int i=0;i<local_n;i++){ dist_local[i]=INT_MAX; parent_local[i]=-1; }
    if (source >= start && source < end) dist_local[source - start] = 0;

    // Message struct for remote relax updates
    typedef struct { int v; int dist; int parent; } RelaxUpdate;

    // Iterative Dijkstra
    for (int iter=0; iter < num_vertices; iter++) {
        // Local min search
        int local_best_v = -1; int local_best_dist = INT_MAX;
        for (int i=0;i<local_n;i++) {
            if (!visited_local[i] && dist_local[i] < local_best_dist) {
                local_best_dist = dist_local[i];
                local_best_v = start + i;
            }
        }
        struct { int dist; int vertex; } in = { local_best_dist, local_best_v }, out;
        MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MINLOC, team_comm);
        int u = out.vertex;
        if (u == -1 || out.dist == INT_MAX) break; // done

        int owner = (u * size) / num_vertices;
        if (rank == owner) {
            // Mark visited locally
            visited_local[u - start] = 1;

            // Build per-target buffers for remote updates
            // For simplicity allocate dynamic vectors per iteration.
            RelaxUpdate **buffers = (RelaxUpdate**)calloc(size, sizeof(RelaxUpdate*));
            int *counts = (int*)calloc(size, sizeof(int));
            int *cap    = (int*)calloc(size, sizeof(int));

            for (GroupedEdgeNode* node = all_edges[u]; node; node = node->next) {
                int v = node->source;
                if (v == u) continue; // ignore self-loop
                int w = (objective >= 0) ? node->weight[objective] : node->combinedWeightInt;
                int dist_u = dist_local[u - start];
                if (dist_u == INT_MAX) continue;
                int cand = dist_u + w;
                int v_owner = (v * size) / num_vertices;
                if (v_owner == rank) {
                    int lv = v - start;
                    if (!visited_local[lv] && cand < dist_local[lv]) {
                        dist_local[lv] = cand; parent_local[lv] = u;
                    }
                } else {
                    // store candidate update for remote owner (v_owner)
                    if (cap[v_owner] == counts[v_owner]) {
                        int newCap = cap[v_owner] == 0 ? 8 : cap[v_owner] * 2;
                        buffers[v_owner] = (RelaxUpdate*)realloc(buffers[v_owner], newCap * sizeof(RelaxUpdate));
                        cap[v_owner] = newCap;
                    }
                    buffers[v_owner][counts[v_owner]++] = (RelaxUpdate){ v, cand, u };
                }
            }

            // Non-blocking sends: first send counts to every OTHER rank
            MPI_Request *reqs = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
            int req_idx = 0;
            for (int r=0; r<size; r++) {
                if (r == rank) continue; // skip self
                int c = counts[r];
                MPI_Isend(&c, 1, MPI_INT, r, 1000 + iter, team_comm, &reqs[req_idx++]);
                if (c > 0) {
                    MPI_Isend(buffers[r], c * sizeof(RelaxUpdate), MPI_BYTE, r, 2000 + iter, team_comm, &reqs[req_idx++]);
                }
            }

            // Receive nothing (owner) but wait for sends completion to avoid buffer reuse
            if (req_idx > 0) MPI_Waitall(req_idx, reqs, MPI_STATUSES_IGNORE);
            free(reqs);
            for (int r=0;r<size;r++) free(buffers[r]);
            free(buffers); free(counts); free(cap);
        } else {
            // Non-owner: receive count from owner of u
            int c = 0; MPI_Request rq_cnt; MPI_Irecv(&c, 1, MPI_INT, owner, 1000 + iter, team_comm, &rq_cnt);
            MPI_Wait(&rq_cnt, MPI_STATUS_IGNORE);
            if (c > 0) {
                RelaxUpdate *recvBuf = (RelaxUpdate*)malloc(c * sizeof(RelaxUpdate));
                MPI_Request rq_upd; MPI_Irecv(recvBuf, c * sizeof(RelaxUpdate), MPI_BYTE, owner, 2000 + iter, team_comm, &rq_upd);
                MPI_Wait(&rq_upd, MPI_STATUS_IGNORE);
                for (int k=0;k<c;k++) {
                    int v = recvBuf[k].v; int cand = recvBuf[k].dist; int p = recvBuf[k].parent;
                    if (v >= start && v < end) {
                        int lv = v - start;
                        if (!visited_local[lv] && cand < dist_local[lv]) { dist_local[lv] = cand; parent_local[lv] = p; }
                    }
                }
                free(recvBuf);
            }
        }

    }

    // Share distances and parents with ALL ranks (not just root) so every rank has full tree
    int *counts = (int*)malloc(size*sizeof(int));
    int *displs = (int*)malloc(size*sizeof(int));
    for (int r=0;r<size;r++) {
        int s = (r * num_vertices)/size;
        int e = ((r+1)*num_vertices)/size;
        counts[r] = e - s;
        displs[r] = s;
    }
    // Allgatherv directly into tree->distance / tree->parent (global arrays inside SOSPTree)
    MPI_Allgatherv(dist_local,   local_n, MPI_INT,
                   tree->distance, counts, displs, MPI_INT, team_comm);
    MPI_Allgatherv(parent_local, local_n, MPI_INT,
                   tree->parent,  counts, displs, MPI_INT, team_comm);

    free(dist_local); free(parent_local); free(visited_local); free(counts); free(displs);
}