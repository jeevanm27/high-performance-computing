#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include "MOSP.h"

typedef struct {
	int vertex;
	int dist;
} PQNode;

void swap(PQNode* a, PQNode* b) {
	PQNode t = *a;
	*a = *b;
	*b = t;
}

int find_min_distance_parallel(int *distance, bool *visited, int num_vertices) {
    int min_dist = INT_MAX;
    int min_vertex = -1;
    
    #pragma omp parallel
    {
        int local_min_dist = INT_MAX;
        int local_min_vertex = -1;
        
        // Each thread finds minimum in its chunk
        #pragma omp for nowait
        for (int i = 0; i < num_vertices; i++) {
            if (!visited[i] && distance[i] < local_min_dist) {
                local_min_dist = distance[i];
                local_min_vertex = i;
            }
        }
        
        // Combine results with critical section
        #pragma omp critical
        {
            if (local_min_dist < min_dist) {
                min_dist = local_min_dist;
                min_vertex = local_min_vertex;
            }
        }
    }
    
    return min_vertex;
}

void dijkstra_build_SOSP(SOSPTree* tree, GroupedEdgesArray all_edges,
                                int source, int objective, int num_vertices) {
    bool *visited = (bool*)calloc(num_vertices, sizeof(bool));
    int *distance = (int*)malloc(num_vertices * sizeof(int));
    int *parent = (int*)malloc(num_vertices * sizeof(int));
    
    // Parallel initialization for large graphs
    if (num_vertices > 1000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_vertices; i++) {
            distance[i] = INT_MAX;
            parent[i] = -1;
        }
    } else {
        for (int i = 0; i < num_vertices; i++) {
            distance[i] = INT_MAX;
            parent[i] = -1;
        }
    }
    
    distance[source] = 0;
    int visited_count = 0;
    
    // Main loop
    while (visited_count < num_vertices) {
        int u;
        
        // Parallel minimum search for large graphs
        if (num_vertices - visited_count > 100) {
            u = find_min_distance_parallel(distance, visited, num_vertices);
        } else {
            // Sequential for small remaining set
            int min_dist = INT_MAX;
            u = -1;
            for (int i = 0; i < num_vertices; i++) {
                if (!visited[i] && distance[i] < min_dist) {
                    min_dist = distance[i];
                    u = i;
                }
            }
        }
        
        if (u == -1 || distance[u] == INT_MAX) break;
        
        visited[u] = true;
        visited_count++;
        
        // Sequential edge relaxation (works best for sparse graphs)
        for (GroupedEdgeNode* node = all_edges[u]; node != NULL; node = node->next) {
            int v = node->source;
            int weight = (objective >= 0) ? node->weight[objective] 
                                           : node->combinedWeightInt;
            
            if (!visited[v] && distance[u] != INT_MAX) {
                int new_dist = distance[u] + weight;
                if (new_dist < distance[v]) {
                    distance[v] = new_dist;
                    parent[v] = u;
                }
            }
        }
    }
    
    // Parallel copy
    if (num_vertices > 1000) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_vertices; i++) {
            tree->distance[i] = distance[i];
            tree->parent[i] = parent[i];
        }
    } else {
        for (int i = 0; i < num_vertices; i++) {
            tree->distance[i] = distance[i];
            tree->parent[i] = parent[i];
        }
    }
    
    free(visited);
    free(distance);
    free(parent);
}
