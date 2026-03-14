#include <stdio.h>
#include <stdlib.h>
#include "MOSP.h"


void combine_sosp(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, preferenceVector* preferences) {
    int num_vertices = sosp_trees[0].num_vertices;
    for (int vertex = 0; vertex < num_vertices; vertex++) {
        for (int objective = 0; objective < NUM_OBJECTIVES; objective++) {
            int parent = sosp_trees[objective].parent[vertex];

            // Skip invalid parents
            if (parent < 0 || parent >= num_vertices) {
                continue;
            }


            {
                // Upsert edge parent -> vertex
                GroupedEdgeNode* node = combined_graph[parent];
                int found = 0;
                while (node) {
                    if (node->source == vertex) {
                        node->combinedWeightDouble -= (1.0 / (double)preferences->data[objective]);
                        node->combinedWeightInt = (int)node->combinedWeightDouble;
                        found = 1;
                        break;
                    }
                    node = node->next;
                }
                if (!found) {
                    GroupedEdgeNode* new_node = createCombinedEdgeNode(vertex);
                    new_node->combinedWeightDouble -= (1.0 / (double)preferences->data[objective]);
                    new_node->combinedWeightInt = (int)new_node->combinedWeightDouble;
                    new_node->next = combined_graph[parent];
                    combined_graph[parent] = new_node;
                }
            }

            {
                // Upsert edge vertex -> parent (to make the graph undirected)
                GroupedEdgeNode* node = combined_graph[vertex];
                int found = 0;
                while (node) {
                    if (node->source == parent) {
                        node->combinedWeightDouble -= (1.0 / (double)preferences->data[objective]);
                        node->combinedWeightInt = (int)node->combinedWeightDouble;
                        found = 1;
                        break;
                    }
                    node = node->next;
                }
                if (!found) {
                    GroupedEdgeNode* new_node = createCombinedEdgeNode(parent);
                    new_node->combinedWeightDouble -= (1.0 / (double)preferences->data[objective]);
                    new_node->combinedWeightInt = (int)new_node->combinedWeightDouble;
                    new_node->next = combined_graph[vertex];
                    combined_graph[vertex] = new_node;
                }
            }
        }
    }
}


// MOSP logic function
void mosp(SOSPTree *final_combined_tree, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector pref) {

    int num_vertices = sosp_trees[0].num_vertices;

        
    // STEP 0: Group insertion edges by destination vertex
    GroupedEdgesArray ins_grouped = {NULL};         
    group_edges(ins_grouped, insertions.data, insertions.size);    
    insert_edges(all_edges, ins_grouped);
    // print_grouped_edges(fp, ins_grouped, false);

    
    // Group deletion edges by destination vertex
    GroupedEdgesArray del_grouped = {NULL};
    group_edges(del_grouped, deletions.data, deletions.size);        
    delete_edges(all_edges, del_grouped);
    // print_grouped_edges(fp, del_grouped, false);
    
    // STEP 1: configure initial sosp trees for each objective
    for(int objective = 0; objective < NUM_OBJECTIVES; objective++) {

        // printf("Updating SOSP tree for objective %d\n", objective);
        update_SOSP(&sosp_trees[objective], all_edges, ins_grouped, del_grouped, objective);
        // fprintf(fp, "Updated SOSP Tree (after dynamic changes) for objective %d:\n", objective);
        // print_SOSP_tree(fp, &sosp_trees[objective]);
    }

    freeGroupedEdges(ins_grouped, MAX_VERTICES);
    freeGroupedEdges(del_grouped, MAX_VERTICES);
    
    // printf("Combining SOSP trees into a single graph based on preference vector...\n");
    GroupedEdgesArray combined_graph = {NULL};
    combine_sosp(combined_graph, sosp_trees, all_edges, &pref);
    // fprintf(fp, "Combined graph edges with combined weights:\n");
    // print_grouped_edges(fp, combined_graph,true);

    // printf("Building final combined SOSP tree on combined graph...\n");
    (*final_combined_tree).num_vertices = num_vertices;
    dijkstra_build_SOSP(final_combined_tree, combined_graph, 0, -1, num_vertices);
    // fprintf(fp, "\nFinal combined SOSP tree (on combined graph):\n");
    // print_SOSP_tree(fp, &final_combined_tree);
}