#ifndef MOSP_H
#define MOSP_H

#include <stdbool.h>
#include <mpi.h>

#define MAX_VERTICES 16000
#define NUM_OBJECTIVES 3

extern FILE* fp; // Log file pointer
extern FILE* out; // Output file pointer

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


GroupedEdgeNode* createGroupedEdgeNode(int source, int weight[]) ;
GroupedEdgeNode* createCombinedEdgeNode(int source) ;

void group_edges(GroupedEdgesArray grouped, Edge *edges, int num);

void freeGroupedEdges(GroupedEdgesArray arr, int num_vertices);

void insert_edges(GroupedEdgesArray all_edges, GroupedEdgesArray inserted);

void delete_edges(GroupedEdgesArray all_edges, GroupedEdgesArray deleted);

void print_grouped_edges(GroupedEdgesArray all_edges, bool combined) ;
void print_SOSP_tree(FILE *fp, SOSPTree* sosp_tree) ;
void print_affected(AffectedSet* affected) ;
void dijkstra_build_SOSP(SOSPTree* tree, GroupedEdgesArray all_edges, int source, int objective, int num_vertices) ;
void dijkstra_build_SOSP_parallel(SOSPTree* tree, GroupedEdgesArray all_edges, int source, int objective, int num_vertices, MPI_Comm team_comm);
void process_insertions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray ins_grouped, int objective);
void process_deletions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray del_grouped, int objective, GroupedEdgesArray all_edges);
void propagate_affected(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray all_edges, int objective);
void update_SOSP(SOSPTree* sosp_tree, GroupedEdgesArray all_edges, GroupedEdgesArray insertions, GroupedEdgesArray deletions, int objective);

// MPI-parallel variants
void process_insertions_parallel(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray ins_grouped, int objective, MPI_Comm comm);
void process_deletions_parallel(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray del_grouped, int objective, GroupedEdgesArray all_edges, MPI_Comm comm);
void propagate_affected_parallel(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray all_edges, int objective, MPI_Comm comm);
void update_SOSP_parallel(SOSPTree* sosp_tree, GroupedEdgesArray all_edges, GroupedEdgesArray insertions, GroupedEdgesArray deletions, int objective, MPI_Comm comm);

void combine_sosp(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, preferenceVector* preferences);
void combine_sosp_parallel(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges,  preferenceVector* preferences, MPI_Comm comm);
void broadcast_combined_graph(GroupedEdgesArray combined_graph, int num_vertices, MPI_Comm comm);

void mosp(SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector p);
void mosp_parallel(SOSPTree *final_output, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector pref, MPI_Comm team_comm);
extern FILE *fp;
extern FILE *out;
#endif // MOSP_H
