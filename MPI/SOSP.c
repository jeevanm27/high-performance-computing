#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include "MOSP.h"
#include <mpi.h>
#include <stdint.h>




GroupedEdgeNode* createGroupedEdgeNode(int source, int weight[]) {
    GroupedEdgeNode* node = (GroupedEdgeNode*) malloc(sizeof(GroupedEdgeNode));
    node->source = source;
    memcpy(node->weight, weight, NUM_OBJECTIVES*sizeof(int));
    node->next = NULL;
    return node;
}

GroupedEdgeNode* createCombinedEdgeNode(int source){
    GroupedEdgeNode* node = (GroupedEdgeNode*) malloc(sizeof(GroupedEdgeNode));
    node->source = source;
    node->combinedWeightDouble = NUM_OBJECTIVES + 1 ;
    node->combinedWeightInt = NUM_OBJECTIVES + 1 ;
    node->next = NULL;
    return node;
}


void group_edges(GroupedEdgesArray grouped, Edge *edges, int num){
    for (int i = 0; i < num; i++) {
        int dest = edges[i].v;
        int source = edges[i].u;
        int* weight = edges[i].weight;

        GroupedEdgeNode* node = createGroupedEdgeNode(source, weight);
        node->next = grouped[dest];
        grouped[dest] = node;

        GroupedEdgeNode* node2 = createGroupedEdgeNode(dest, weight);
        node2->next = grouped[source];
        grouped[source] = node2;
    }
}

void freeGroupedEdges(GroupedEdgesArray arr, int num_vertices) {
    for (int i = 0; i < num_vertices; i++) {
        GroupedEdgeNode* current = arr[i];
        while (current) {
            GroupedEdgeNode* tmp = current;
            current = current->next;
            free(tmp);
        }
        arr[i] = NULL;
    }
}

void insert_edges(GroupedEdgesArray all_edges, GroupedEdgesArray inserted) {
    for (int v = 0; v < MAX_VERTICES; v++) {
        if (inserted[v] != NULL) {
            GroupedEdgeNode* curr = inserted[v];
            while (curr) {
                // Create a copy of the current node and insert
                GroupedEdgeNode* copy = createGroupedEdgeNode(curr->source, curr->weight);
                copy->next = all_edges[v];
                all_edges[v] = copy;

                curr = curr->next;
            }
        }
    }
}

void delete_edges(GroupedEdgesArray all_edges, GroupedEdgesArray deleted) {
    for (int v = 0; v < MAX_VERTICES; v++) {
        if (deleted[v] == NULL) continue;
        GroupedEdgeNode* del = deleted[v];
        while (del) {
            int src_to_remove = del->source;
            GroupedEdgeNode* curr = all_edges[v];
            GroupedEdgeNode* prev = NULL;
            while (curr) {
                if (curr->source == src_to_remove) {
                    GroupedEdgeNode* tmp = curr;
                    if (prev) prev->next = curr->next;
                    else      all_edges[v] = curr->next;
                    curr = curr->next;
                    free(tmp);
                } else {
                    prev = curr;
                    curr = curr->next;
                }
            }
            del = del->next;
        }
    }
}


void print_grouped_edges(GroupedEdgesArray all_edges, bool combined) {
    fprintf(fp, "Src\tDest\tObjectives\n");  
    for (int v = 0; v < MAX_VERTICES; v++) {
        if (all_edges[v] != NULL) {
            GroupedEdgeNode* curr = all_edges[v];
            while (curr) {
                int u = curr->source;
                if (combined)
                   fprintf(fp, "%d\t%d\t%d\n", u, v, curr->combinedWeightInt);
                else{
                    fprintf(fp, "%d\t%d\t", u, v);
                    for (int i = 0; i < NUM_OBJECTIVES; i++)
                        fprintf(fp, "%2d ", curr->weight[i]);
                    fprintf(fp, "\n");
                }
                curr = curr->next;
            }
        }
    }
    fprintf(fp, "\n");   
}

void print_SOSP_tree(FILE *fp, SOSPTree* sosp_tree) {
    fprintf(fp, "Vertex\tDistance\tParent\n");
    for (int i = 0; i < sosp_tree->num_vertices; i++) {
        if (sosp_tree->distance[i] == INT_MAX) {
            fprintf(fp, "%d\tINF\t\t%d\n", i, sosp_tree->parent[i]);
        } else {
            fprintf(fp, "%d\t%d\t\t%d\n", i, sosp_tree->distance[i], sosp_tree->parent[i]);
        }
    }
    fprintf(fp, "\n");
}

void print_affected(AffectedSet* affected) {
    fprintf(fp, "Affected vertices: ");
    for (int i = 0; i < MAX_VERTICES; i++) {
        if (affected->affected[i]) {
            fprintf(fp, "%d ", i);
        }        
    }    
    fprintf(fp,"\n");        
}

void process_insertions_parallel(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray ins_grouped, int objective, MPI_Comm comm){
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    int n = sosp_tree->num_vertices;
    int start = (rank * n) / size; int end = ((rank+1)*n)/size;

    for (int v = start; v < end; v++) {
        if (ins_grouped[v] != NULL) {
            GroupedEdgeNode* curr = ins_grouped[v];
            int best_dist = sosp_tree->distance[v];
            int best_parent = sosp_tree->parent[v];
            bool updated = false;
            while (curr) {
                int u = curr->source;
                if (sosp_tree->distance[u] != INT_MAX) {
                    int new_dist = sosp_tree->distance[u] + curr->weight[objective];
                    if (new_dist < best_dist) { best_dist = new_dist; best_parent = u; updated = true; }
                }
                curr = curr->next;
            }
            if (updated) {
                sosp_tree->distance[v] = best_dist;
                sosp_tree->parent[v] = best_parent;
                if (!affected->affected[v]) { affected->affected[v] = true; affected->count++; }
            }
        }
    }

    // Share distances/parents across ranks
    int* counts = (int*)malloc(size*sizeof(int));
    int* displs = (int*)malloc(size*sizeof(int));
    for (int r=0;r<size;r++){ displs[r] = (r*n)/size; counts[r] = ((r+1)*n)/size - displs[r]; }
    MPI_Allgatherv(sosp_tree->distance + start, end-start, MPI_INT, sosp_tree->distance, counts, displs, MPI_INT, comm);
    MPI_Allgatherv(sosp_tree->parent + start, end-start, MPI_INT, sosp_tree->parent, counts, displs, MPI_INT, comm);
    
    // Combine affected flags (logical OR)
    int* aff_int = (int*)calloc(n, sizeof(int));
    for (int i=0;i<n;i++) aff_int[i] = affected->affected[i] ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, aff_int, n, MPI_INT, MPI_LOR, comm);
    affected->count = 0; for (int i=0;i<n;i++){ affected->affected[i] = (aff_int[i]!=0); if (affected->affected[i]) affected->count++; }
    free(aff_int); free(counts); free(displs);
}

void process_deletions_parallel(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray del_grouped, int objective, GroupedEdgesArray all_edges, MPI_Comm comm){
    int rank, size; MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    int n = sosp_tree->num_vertices; int start=(rank*n)/size; int end=((rank+1)*n)/size;
    for (int v = start; v < end; v++) {
        if (del_grouped[v] != NULL) {
            GroupedEdgeNode* curr = del_grouped[v];
            while (curr) {
                int u = curr->source;
                if (sosp_tree->parent[v] == u) {
                    int best_alt_dist = INT_MAX; int best_alt_parent = -1;
                    GroupedEdgeNode* p = all_edges[v];
                    while (p) {
                        if (p->source != u && sosp_tree->distance[p->source] != INT_MAX) {
                            int alt_dist = sosp_tree->distance[p->source] + p->weight[objective];
                            if (alt_dist < best_alt_dist) { best_alt_dist = alt_dist; best_alt_parent = p->source; }
                        }
                        p = p->next;
                    }
                    if (best_alt_parent != -1) { sosp_tree->distance[v] = best_alt_dist; sosp_tree->parent[v] = best_alt_parent; }
                    else { sosp_tree->distance[v] = INT_MAX; sosp_tree->parent[v] = -1; }
                    if (!affected->affected[v]) { affected->affected[v] = true; affected->count++; }
                    break;
                }
                curr = curr->next;
            }
        }
    }
    // Share distances/parents and affected flags
    int* counts = (int*)malloc(size*sizeof(int)); int* displs=(int*)malloc(size*sizeof(int));
    for (int r=0;r<size;r++){ displs[r]=(r*n)/size; counts[r]=((r+1)*n)/size - displs[r]; }
    MPI_Allgatherv(sosp_tree->distance + start, end-start, MPI_INT, sosp_tree->distance, counts, displs, MPI_INT, comm);
    MPI_Allgatherv(sosp_tree->parent + start, end-start, MPI_INT, sosp_tree->parent, counts, displs, MPI_INT, comm);
    int* aff_int = (int*)calloc(n, sizeof(int)); for (int i=0;i<n;i++) aff_int[i]=affected->affected[i]?1:0;
    MPI_Allreduce(MPI_IN_PLACE, aff_int, n, MPI_INT, MPI_LOR, comm);
    affected->count=0; for (int i=0;i<n;i++){ affected->affected[i]=(aff_int[i]!=0); if (affected->affected[i]) affected->count++; }
    free(aff_int); free(counts); free(displs);
}

void propagate_affected_parallel(SOSPTree* tree, AffectedSet* affected,  GroupedEdgesArray all_edges, int objective, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int n = tree->num_vertices;
    
    bool *local_marked = calloc(n, sizeof(bool));
    bool *global_marked = calloc(n, sizeof(bool));
    bool *next_affected = calloc(n, sizeof(bool));
    
    int changed = 1;
    
    while (changed) {
        changed = 0;
        memset(local_marked, 0, n * sizeof(bool));
        
        // PHASE 1: Mark neighbors of affected vertices
        for (int v = 0; v < n; v++) {
            if (!affected->affected[v]) continue;
            for (GroupedEdgeNode* node = all_edges[v]; node; node = node->next) {
                local_marked[node->source] = true;
            }
        }
        
        // PHASE 2: Global OR of marked vertices
        MPI_Allreduce(local_marked, global_marked, n, MPI_C_BOOL, MPI_LOR, comm);
        
        // PHASE 3: Update marked vertices (distributed)
        int local_changed = 0;
        for (int u = rank; u < n; u += size) {
            if (!global_marked[u]) continue;
            
            int best_dist = tree->distance[u];
            int best_parent = tree->parent[u];
            
            for (GroupedEdgeNode* node = all_edges[u]; node; node = node->next) {
                int v = node->source;
                if (affected->affected[v] && tree->distance[v] != INT_MAX) {
                    int w = (objective >= 0) ? node->weight[objective] : node->combinedWeightInt;
                    int cand_dist = tree->distance[v] + w;
                    if (cand_dist < best_dist) {
                        best_dist = cand_dist;
                        best_parent = v;
                    }
                }
            }
            
            if (best_dist < tree->distance[u]) {
                tree->distance[u] = best_dist;
                tree->parent[u] = best_parent;
                next_affected[u] = true;
                local_changed = 1;
            }
        }
        
        // Synchronize updates
        MPI_Allreduce(MPI_IN_PLACE, tree->distance, n, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, tree->parent, n, MPI_INT, MPI_MAX, comm);
        MPI_Allreduce(MPI_IN_PLACE, next_affected, n, MPI_C_BOOL, MPI_LOR, comm);
        MPI_Allreduce(&local_changed, &changed, 1, MPI_INT, MPI_LOR, comm);
        
        // Update affected set for next iteration
        memcpy(affected->affected, next_affected, n * sizeof(bool));
        memset(next_affected, 0, n * sizeof(bool));
        affected->count = 0;
        for (int i = 0; i < n; i++) {
            if (affected->affected[i]) affected->count++;
        }
    }
    
    free(local_marked);
    free(global_marked);
    free(next_affected);
}



void update_SOSP_parallel(SOSPTree* sosp_tree, GroupedEdgesArray all_edges, GroupedEdgesArray insertions, GroupedEdgesArray deletions, int objective, MPI_Comm comm){
    AffectedSet affected; for (int i=0;i<sosp_tree->num_vertices;i++) affected.affected[i]=false; affected.count=0;
    process_insertions_parallel(sosp_tree, &affected, insertions, objective, comm);
    process_deletions_parallel(sosp_tree, &affected, deletions, objective, all_edges, comm);
    propagate_affected_parallel(sosp_tree, &affected, all_edges, objective, comm);
}





