#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOSP.h"
#include <mpi.h>

// Structure to transfer edge weight reductions
typedef struct { 
    int u; 
    int v; 
    double reduction; 
} WeightedEdge;

// Helper to update nodes locally on Rank 0
void upsert_edge_reduction(GroupedEdgesArray graph, int u, int v, double reduction) {
    GroupedEdgeNode* node = graph[u];
    bool found = false;
    
    while (node) {
        if (node->source == v) {
            node->combinedWeightDouble -= reduction;
            node->combinedWeightInt = (int)node->combinedWeightDouble;
            found = true;
            break;
        }
        node = node->next;
    }

    if (!found) {
        GroupedEdgeNode* new_node = createCombinedEdgeNode(v); 
        new_node->combinedWeightDouble = (double)(NUM_OBJECTIVES + 1);
        new_node->combinedWeightDouble -= reduction;
        new_node->combinedWeightInt = (int)new_node->combinedWeightDouble;
        new_node->next = graph[u];
        graph[u] = new_node;
    }
}

void combine_sosp_parallel(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, preferenceVector* preferences, MPI_Comm comm) {    
    int rank, size; 
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &size);

    // 1. Sync Graph Size
    int local_n = 0;
    for(int i=0; i<NUM_OBJECTIVES; i++) {
        if (sosp_trees[i].num_vertices > local_n) local_n = sosp_trees[i].num_vertices;
    }
    int num_vertices = 0;
    MPI_Allreduce(&local_n, &num_vertices, 1, MPI_INT, MPI_MAX, comm);

    // Calculate counts and displacements for distributing vertices
    int *scat_counts = (int*)malloc(size * sizeof(int));
    int *scat_displs = (int*)malloc(size * sizeof(int));
    
    int chunk_size = num_vertices / size;
    int remainder = num_vertices % size;
    int offset = 0;
    
    for (int r = 0; r < size; r++) {
        scat_counts[r] = chunk_size + (r < remainder ? 1 : 0);
        scat_displs[r] = offset;
        offset += scat_counts[r];
    }

    int my_count = scat_counts[rank];
    int my_offset = scat_displs[rank]; 

    int* local_parents = (int*)malloc(NUM_OBJECTIVES * my_count * sizeof(int));

    for (int obj = 0; obj < NUM_OBJECTIVES; obj++) {
        int root = obj % size; 
        
        int* send_buf = NULL;
        if (rank == root) {
            send_buf = sosp_trees[obj].parent;
        }

        // Root scatters 'parent' array. Rank 0 gets 0-100. Rank 1 gets 101-200.
        MPI_Scatterv(
            send_buf, scat_counts, scat_displs, MPI_INT, // Send args (used by root)
            &local_parents[obj * my_count], my_count, MPI_INT, // Recv args (used by all)
            root, comm
        );
    }

    int cap = 4096;
    int cnt = 0;
    WeightedEdge* edge_buf = (WeightedEdge*)malloc(sizeof(WeightedEdge) * cap);

    // Iterate ONLY through the vertices I received in the Scatter
    for (int i = 0; i < my_count; i++) {
        int v = my_offset + i; 
        
        for (int obj = 0; obj < NUM_OBJECTIVES; obj++) {
            int p = local_parents[obj * my_count + i];
            
            if (p < 0 || p >= num_vertices) continue;

            double reduction = (1.0 / (double)preferences->data[obj]);

            if (cnt + 1 > cap) { 
                cap *= 2; 
                edge_buf = (WeightedEdge*)realloc(edge_buf, sizeof(WeightedEdge) * cap); 
            }

            edge_buf[cnt++] = (WeightedEdge){ p, v, reduction };
        }
    }

    // Gather how many edges each rank found
    int* recv_counts = NULL;
    if (rank == 0) recv_counts = (int*)malloc(size * sizeof(int));
    MPI_Gather(&cnt, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, comm);

    int* displs = NULL;
    WeightedEdge* all_edges_flat = NULL;
    int total_edges = 0;

    if (rank == 0) {
        displs = (int*)malloc(size * sizeof(int));
        displs[0] = 0;
        for (int i=0; i<size; i++) {
            if (i>0) displs[i] = displs[i-1] + recv_counts[i-1];
            total_edges += recv_counts[i];
        }
        for (int i=0; i<size; i++) {
            recv_counts[i] *= sizeof(WeightedEdge);
            displs[i] *= sizeof(WeightedEdge);
        }
        all_edges_flat = (WeightedEdge*)malloc(total_edges * sizeof(WeightedEdge));
    }

    MPI_Gatherv(edge_buf, cnt * sizeof(WeightedEdge), MPI_BYTE, 
                all_edges_flat, recv_counts, displs, MPI_BYTE, 
                0, comm);

    // Rank 0 processes all gathered edges to build the combined graph
    if (rank == 0) {
        for (int i = 0; i < total_edges; i++) {
            WeightedEdge e = all_edges_flat[i];
            upsert_edge_reduction(combined_graph, e.u, e.v, e.reduction);
            upsert_edge_reduction(combined_graph, e.v, e.u, e.reduction);
        }
        free(recv_counts);
        free(displs);
        free(all_edges_flat);
    }

    free(scat_counts);
    free(scat_displs);
    free(local_parents);
    free(edge_buf);
}

// Broadcast combined graph from rank 0 to all ranks
void broadcast_combined_graph(GroupedEdgesArray combined_graph, int num_vertices, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    // Step 1: Count total edges on rank 0
    int total_edges = 0;
    if (rank == 0) {
        for (int v = 0; v < num_vertices; v++) {
            GroupedEdgeNode* node = combined_graph[v];
            while (node) {
                total_edges++;
                node = node->next;
            }
        }
    }
    
    // Step 2: Broadcast total edge count
    MPI_Bcast(&total_edges, 1, MPI_INT, 0, comm);
    
    if (total_edges == 0) return; // No edges to broadcast
    
    // Step 3: Flatten the graph on rank 0
    typedef struct {
        int dest;
        int source;
        double weight;
    } FlatEdge;
    
    FlatEdge* flat_edges = (FlatEdge*)malloc(total_edges * sizeof(FlatEdge));
    
    if (rank == 0) {
        int idx = 0;
        for (int v = 0; v < num_vertices; v++) {
            GroupedEdgeNode* node = combined_graph[v];
            while (node) {
                flat_edges[idx].dest = v;
                flat_edges[idx].source = node->source;
                flat_edges[idx].weight = node->combinedWeightDouble;
                idx++;
                node = node->next;
            }
        }
    }
    
    // Step 4: Broadcast flattened edges
    MPI_Bcast(flat_edges, total_edges * sizeof(FlatEdge), MPI_BYTE, 0, comm);
    
    // Step 5: Reconstruct graph on all non-root ranks
    if (rank != 0) {
        for (int i = 0; i < total_edges; i++) {
            int dest = flat_edges[i].dest;
            int source = flat_edges[i].source;
            double weight = flat_edges[i].weight;
            
            // Create new node
            GroupedEdgeNode* new_node = createCombinedEdgeNode(source);
            new_node->combinedWeightDouble = weight;
            new_node->combinedWeightInt = (int)weight;
            new_node->next = combined_graph[dest];
            combined_graph[dest] = new_node;
        }
    }
    
    free(flat_edges);
}

void mosp_parallel(SOSPTree *final_output, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector pref, MPI_Comm team_comm) {
    
    int rank, size; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int team_rank; 
    MPI_Comm_rank(team_comm, &team_rank);
    int team_color = rank % NUM_OBJECTIVES;

    // 1. Sync Max Vertices
    int local_n = 0;
    for (int i = 0; i < NUM_OBJECTIVES; i++) if (sosp_trees[i].num_vertices > local_n) local_n = sosp_trees[i].num_vertices;
    int num_vertices = 0;
    MPI_Allreduce(&local_n, &num_vertices, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    for (int i=0; i<NUM_OBJECTIVES; i++) sosp_trees[i].num_vertices = num_vertices;

    // 2. Dynamic Updates
    GroupedEdgesArray ins_grouped = {NULL};
    group_edges(ins_grouped, insertions.data, insertions.size);
    insert_edges(all_edges, ins_grouped);

    GroupedEdgesArray del_grouped = {NULL};
    group_edges(del_grouped, deletions.data, deletions.size);
    delete_edges(all_edges, del_grouped);

    // 3. SOSP Update
    for (int objective = 0; objective < NUM_OBJECTIVES; objective++) {
        if (team_color == objective) {
            update_SOSP_parallel(&sosp_trees[objective], all_edges, ins_grouped, del_grouped, objective, team_comm);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    // 4. Combine & Dijkstra
    static GroupedEdgeNode* combined_graph[MAX_VERTICES] = {NULL};

    // Calls the new SCATTER-based function (builds graph on rank 0)
    combine_sosp_parallel(combined_graph, sosp_trees, all_edges, &pref, MPI_COMM_WORLD);

    // Broadcast combined graph from rank 0 to all ranks
    broadcast_combined_graph(combined_graph, num_vertices, MPI_COMM_WORLD);

    final_output->num_vertices = num_vertices;
        
    // Now ALL ranks have the combined graph and can run parallel Dijkstra
    dijkstra_build_SOSP_parallel(final_output, combined_graph, 0, -1, num_vertices, MPI_COMM_WORLD);
        
    // Optional: Print here for logging, but Main will also print the final result now.
    // fprintf(out, "\nFinal MOSP Heuristic Path (Combined Weights):\n");
    // print_SOSP_tree(out, final_output);
    // fflush(out); 
    

    freeGroupedEdges(ins_grouped, MAX_VERTICES);
    freeGroupedEdges(del_grouped, MAX_VERTICES);
    freeGroupedEdges(combined_graph, MAX_VERTICES);
}