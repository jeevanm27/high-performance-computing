#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include "MOSP.h"


 int INNER_THREADS = 1;
 int OUTER_THREADS = 3;


void configure_thread_hierarchy(int max_threads, int num_objectives) {

	OUTER_THREADS = num_objectives;  
	
	if (max_threads <= num_objectives) {
		
		INNER_THREADS = 1;
	} else {
		// Distribute remaining threads to inner level
		// max_threads = OUTER_THREADS * INNER_THREADS
		INNER_THREADS = max_threads / OUTER_THREADS;
		
		// Ensure at least 1 inner thread
		if (INNER_THREADS < 1) INNER_THREADS = 1;
	}
	
	// Enable nested parallelism
	omp_set_nested(1);
	omp_set_max_active_levels(2);
	
	printf("Thread Configuration:\n");
	printf("  Max threads: %d\n", max_threads);
	printf("  Outer level (objectives): %d threads\n", OUTER_THREADS);
	printf("  Inner level (per objective): %d threads\n", INNER_THREADS);
	printf("  Total thread usage: %d threads\n\n", OUTER_THREADS * INNER_THREADS);
}

GroupedEdgeNode* createGroupedEdgeNode(int source, int weight[]) {
	GroupedEdgeNode* node = (GroupedEdgeNode*) malloc(sizeof(GroupedEdgeNode));
	node->source = source;
	memcpy(node->weight, weight, NUM_OBJECTIVES*sizeof(int));
	node->next = NULL;
	return node;
}

GroupedEdgeNode* createCombinedEdgeNode(int source) {
	GroupedEdgeNode* node = (GroupedEdgeNode*) malloc(sizeof(GroupedEdgeNode));
	node->source = source;
	node->combinedWeightDouble = NUM_OBJECTIVES + 1;
	node->combinedWeightInt = NUM_OBJECTIVES + 1;
	node->next = NULL;
	return node;
}

void group_edges(GroupedEdgesArray grouped, Edge *edges, int num) {
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
	if (num_vertices > 5000) {
		#pragma omp parallel for schedule(static) num_threads(INNER_THREADS)
		for (int i = 0; i < num_vertices; i++) {
			GroupedEdgeNode* current = arr[i];
			while (current) {
				GroupedEdgeNode* tmp = current;
				current = current->next;
				free(tmp);
			}
			arr[i] = NULL;
		}
	} else {
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
}

void insert_edges(GroupedEdgesArray all_edges, GroupedEdgesArray inserted) {
	for (int v = 0; v < MAX_VERTICES; v++) {
		if (inserted[v] != NULL) {
			GroupedEdgeNode* curr = inserted[v];
			while (curr) {
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
					else all_edges[v] = curr->next;
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

void print_grouped_edges(FILE *fp, GroupedEdgesArray all_edges, bool combined) {
	fprintf(fp, "Src\tDest\tObjectives\n");
	for (int v = 0; v < MAX_VERTICES; v++) {
		if (all_edges[v] != NULL) {
			GroupedEdgeNode* curr = all_edges[v];
			while (curr) {
				int u = curr->source;
				if (combined)
					fprintf(fp, "%d\t%d\t%d\n", u, v, curr->combinedWeightInt);
				else {
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

void print_affected(FILE *fp, AffectedSet* affected) {
	fprintf(fp, "Affected vertices: ");
	for (int i = 0; i < MAX_VERTICES; i++) {
		if (affected->affected[i]) {
			fprintf(fp, "%d ", i);
		}
	}
	fprintf(fp,"\n");
}

void process_insertions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray ins_grouped, int objective) {
	// Use INNER_THREADS - we're already inside an outer parallel region
	#pragma omp parallel for schedule(static) num_threads(INNER_THREADS) if(INNER_THREADS > 1)
	for (int v = 0; v < sosp_tree->num_vertices; v++) {
		if (ins_grouped[v] != NULL) {
			GroupedEdgeNode* curr = ins_grouped[v];
			int best_dist = sosp_tree->distance[v];
			int best_parent = sosp_tree->parent[v];
			bool updated = false;
			
			while (curr) {
				int u = curr->source;
				if (sosp_tree->distance[u] != INT_MAX) {
					int new_dist = sosp_tree->distance[u] + curr->weight[objective];
					if (new_dist < best_dist) {
						best_dist = new_dist;
						best_parent = u;
						updated = true;
					}
				}
				curr = curr->next;
			}
			
			if (updated) {
				sosp_tree->distance[v] = best_dist;
				sosp_tree->parent[v] = best_parent;
				affected->affected[v] = true;
			}
		}
	}
	
	affected->count = 0;
	for (int i = 0; i < sosp_tree->num_vertices; i++) {
		if (affected->affected[i]) affected->count++;
	}
}

void process_deletions(SOSPTree *sosp_tree, AffectedSet* affected, GroupedEdgesArray del_grouped, int objective, GroupedEdgesArray all_edges) {
	// Use INNER_THREADS - we're already inside an outer parallel region
	#pragma omp parallel for schedule(static) num_threads(INNER_THREADS) if(INNER_THREADS > 1)
	for (int v = 0; v < sosp_tree->num_vertices; v++) {
		if (del_grouped[v] != NULL) {
			GroupedEdgeNode* curr = del_grouped[v];
			
			while (curr) {
				int u = curr->source;
				if (sosp_tree->parent[v] == u) {
					int best_alt_dist = INT_MAX;
					int best_alt_parent = -1;
					
					GroupedEdgeNode* p = all_edges[v];
					while (p) {
						if (p->source != u) {
							if (sosp_tree->distance[p->source] != INT_MAX) {
								int alt_weight = p->weight[objective];
								int alt_dist = sosp_tree->distance[p->source] + alt_weight;
								if (alt_dist < best_alt_dist) {
									best_alt_dist = alt_dist;
									best_alt_parent = p->source;
								}
							}
						}
						p = p->next;
					}
					
					if (best_alt_parent != -1) {
						sosp_tree->distance[v] = best_alt_dist;
						sosp_tree->parent[v] = best_alt_parent;
					} else {
						sosp_tree->distance[v] = INT_MAX;
						sosp_tree->parent[v] = -1;
					}
					
					affected->affected[v] = true;
					break;
				}
				curr = curr->next;
			}
		}
	}
	
	affected->count = 0;
	for (int i = 0; i < sosp_tree->num_vertices; i++) {
		if (affected->affected[i]) affected->count++;
	}
}

void propagate_affected(SOSPTree *tree, AffectedSet *affected, GroupedEdgesArray all_edges, int objective) {
	bool changed;
	do {
		changed = false;
		bool next_affected[MAX_VERTICES] = {0};

		// Use INNER_THREADS - we're already inside an outer parallel region
		#pragma omp parallel num_threads(INNER_THREADS) reduction(||:changed) if(INNER_THREADS > 1)
		{
			#pragma omp for schedule(static)
			for (int v = 0; v < tree->num_vertices; v++) {
				if (!affected->affected[v]) continue;

				GroupedEdgeNode *curr = all_edges[v];
				while (curr) {
					int u = curr->source;
					int cand_dist = tree->distance[v] + curr->weight[objective];

					if (cand_dist < tree->distance[u]) {
						// ATOMIC NEEDED: Multiple threads can update same destination u
						int old_dist;
						#pragma omp atomic read
						old_dist = tree->distance[u];
						
						if (cand_dist < old_dist) {
							#pragma omp atomic write
							tree->distance[u] = cand_dist;
							#pragma omp atomic write
							tree->parent[u] = v;
							
							next_affected[u] = true;
							changed = true;
						}
					}
					curr = curr->next;
				}
			}
		}

		int next_count = 0;
		#pragma omp parallel for num_threads(INNER_THREADS) reduction(+:next_count) schedule(static) if(INNER_THREADS > 1)
		for (int i = 0; i < tree->num_vertices; i++) {
			if (next_affected[i]) next_count++;
		}

		memcpy(affected->affected, next_affected, sizeof(next_affected));
		affected->count = next_count;
	} while (changed);
}

void update_SOSP(SOSPTree* sosp_tree, GroupedEdgesArray all_edges, GroupedEdgesArray insertions, GroupedEdgesArray deletions, int objective) {
	AffectedSet affected;
	for (int i = 0; i < sosp_tree->num_vertices; i++)
		affected.affected[i] = false;
	affected.count = 0;

	process_insertions(sosp_tree, &affected, insertions, objective);
	process_deletions(sosp_tree, &affected, deletions, objective, all_edges);
	propagate_affected(sosp_tree, &affected, all_edges, objective);
}
