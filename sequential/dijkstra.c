#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include "MOSP.h"

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
