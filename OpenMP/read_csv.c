#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "MOSP.h"

#define INITIAL_CAPACITY 1024

// Initialize empty dynamic array
void edge_array_init(EdgeArray *arr) {
    arr->capacity = INITIAL_CAPACITY;
    arr->size = 0;
    arr->data = malloc(arr->capacity * sizeof(Edge));
    if (!arr->data) {
        perror("malloc");
        exit(1);
    }
}

// Double capacity when full
void edge_array_reserve(EdgeArray *arr) {
    if (arr->size >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, arr->capacity * sizeof(Edge));
        if (!arr->data) {
            perror("realloc");
            exit(1);
        }
    }
}

// Final cleanup
void edge_array_free(EdgeArray *arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = arr->capacity = 0;
}

// Read edges from CSV file into dynamic array
int read_data(EdgeArray *edges, const char *file_name, bool del) {
    FILE *fp = fopen(file_name, "r");
    if (!fp) {
        perror("Error opening file");
        return -1;
    }
    
    edge_array_init(edges);
    char line[2048];
    if (!fgets(line, sizeof(line), fp)) { // skip header
        fclose(fp);
        return -1;
    }
    
    int u, v;
    int weights[NUM_OBJECTIVES];
    int max_vertex = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        char *p = line;
        
        // Parse u
        u = (int)strtol(p, &p, 10);
        if (*p != ',') continue;
        p++;
        
        // Parse v
        v = (int)strtol(p, &p, 10);
        if (u == v) continue; // skip self-loops
        
        // Update max vertex ID
        if (u > max_vertex) max_vertex = u;
        if (v > max_vertex) max_vertex = v;
        
        // Parse weights only if not in deletion mode
        if (!del) {
            for (int i = 0; i < NUM_OBJECTIVES; i++) {
                if (*p == ',') p++;
                weights[i] = (int)strtol(p, &p, 10);
            }
        } else {
            // zero weights for deletion batches
            for (int i = 0; i < NUM_OBJECTIVES; i++) weights[i] = 0;
        }
        
        // Grow if needed and store
        edge_array_reserve(edges);
        edges->data[edges->size].u = u;
        edges->data[edges->size].v = v;
        memcpy(edges->data[edges->size].weight, weights, sizeof(weights));
        edges->size++;
    }
    
    fclose(fp);
    return max_vertex + 1; // number of vertices
}
