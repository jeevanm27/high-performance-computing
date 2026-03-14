#ifndef MOSP_H
#define MOSP_H

#include <stdio.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_VERTICES 16000
#define NUM_OBJECTIVES 3

extern FILE* fp; // Log file pointer

typedef struct {
    int u, v;
    int weight[NUM_OBJECTIVES];
} Edge;

typedef struct {
    Edge *data;
    int size;
    int capacity;
} EdgeArray;

typedef struct {
    int distance[MAX_VERTICES];
    int parent[MAX_VERTICES];
    int num_vertices;
} SOSPTree;

typedef struct {
    bool affected[MAX_VERTICES];
    int count;
} AffectedSet;

typedef struct GroupedEdgeNode {
    int source;
    union {
        int weight[NUM_OBJECTIVES];
        struct {
            int combinedWeightInt;
            double combinedWeightDouble;
        };
    };
    struct GroupedEdgeNode* next;
} GroupedEdgeNode;

typedef GroupedEdgeNode* GroupedEdgesArray[MAX_VERTICES];

typedef struct {
    int data[NUM_OBJECTIVES];
} preferenceVector;

GroupedEdgeNode* createGroupedEdgeNode(int source, int weight[]);
GroupedEdgeNode* createCombinedEdgeNode(int source);
void group_edges(GroupedEdgesArray grouped, Edge *edges, int num);
void freeGroupedEdges(GroupedEdgesArray arr, int num_vertices);
void insert_edges(GroupedEdgesArray all_edges, GroupedEdgesArray inserted);
void delete_edges(GroupedEdgesArray all_edges, GroupedEdgesArray deleted);
void print_grouped_edges(FILE *fp, GroupedEdgesArray all_edges, bool combined);
void print_SOSP_tree(FILE *fp, SOSPTree* sosp_tree);
void print_affected(FILE *fp, AffectedSet* affected);
void dijkstra_build_SOSP(SOSPTree* tree, GroupedEdgesArray all_edges, int source, int objective, int num_vertices);
void process_insertions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray ins_grouped, int objective);
void process_deletions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray del_grouped, int objective, GroupedEdgesArray all_edges);
void propagate_affected(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray all_edges, int objective);
void update_SOSP(SOSPTree* sosp_tree, GroupedEdgesArray all_edges, GroupedEdgesArray insertions, GroupedEdgesArray deletions, int objective);
void combine_sosp(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, preferenceVector* preferences);
void mosp(SOSPTree *final_combined_tree, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector p);

#endif // MOSP_H
