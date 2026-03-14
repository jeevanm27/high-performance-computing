#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOSP.h"

void combine_sosp(GroupedEdgesArray combined_graph, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, preferenceVector* preferences) {
	int num_vertices = sosp_trees[0].num_vertices;
	
	for (int vertex = 0; vertex < num_vertices; vertex++) {
		for (int objective = 0; objective < NUM_OBJECTIVES; objective++) {
			int parent = sosp_trees[objective].parent[vertex];
			if (parent < 0 || parent >= num_vertices) {
				continue;
			}
			
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
			
			// Upsert edge vertex -> parent
			node = combined_graph[vertex];
			found = 0;
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

void mosp(SOSPTree *final_combined_tree, SOSPTree sosp_trees[], GroupedEdgesArray all_edges, EdgeArray insertions, EdgeArray deletions, preferenceVector pref) {
	int num_vertices = sosp_trees[0].num_vertices;
	
	GroupedEdgesArray ins_grouped = {NULL};
	group_edges(ins_grouped, insertions.data, insertions.size);
	insert_edges(all_edges, ins_grouped);
	
	GroupedEdgesArray del_grouped = {NULL};
	group_edges(del_grouped, deletions.data, deletions.size);
	delete_edges(all_edges, del_grouped);
	
	// Parallel SOSP tree updates - no critical section needed for logging
	#pragma omp parallel for schedule(static) num_threads(3)
	for(int objective = 0; objective < NUM_OBJECTIVES; objective++) {
		update_SOSP(&sosp_trees[objective], all_edges, ins_grouped, del_grouped, objective);
		// Log without critical section (output may be scrambled but faster)
		// fprintf(fp, "Updated SOSP Tree (after dynamic changes) for objective %d:\n", objective);
		// print_SOSP_tree(fp, &sosp_trees[objective]);
	}
	
	freeGroupedEdges(ins_grouped, MAX_VERTICES);
	freeGroupedEdges(del_grouped, MAX_VERTICES);
	
	GroupedEdgesArray combined_graph = {NULL};
	combine_sosp(combined_graph, sosp_trees, all_edges, &pref);
	
	(*final_combined_tree).num_vertices = num_vertices;
	dijkstra_build_SOSP(final_combined_tree, combined_graph, 0, -1, num_vertices);
}
