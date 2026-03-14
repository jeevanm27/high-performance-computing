#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "MOSP.h"
#include "read_csv.c"
#include "dijkstra.c"
#include "SOSP.c"
#include "MOSP.c"

// Platform-specific timing headers
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

FILE *fp = NULL;

// Platform-specific timer
double get_time() {
#ifdef _WIN32
	LARGE_INTEGER frequency, counter;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

double run_one_test(int dataset_id, int batch_size,int iterations, FILE *bench, SOSPTree *final_combined_tree, SOSPTree sosp_trees[], GroupedEdgesArray all_edges) {
	char ins_path[256], del_path[256];
	snprintf(ins_path, sizeof(ins_path), "../data_dynamic_%d_%d/alt_batches/alt_insert_%d.csv", batch_size, iterations, dataset_id);
	snprintf(del_path, sizeof(del_path), "../data_dynamic_%d_%d/alt_batches/alt_delete_%d.csv", batch_size, iterations, dataset_id);
	
	EdgeArray insertions, deletions;
	if (read_data(&insertions, ins_path, false) < 0) {
		printf("Failed to read insertion file: %s\n", ins_path);
		return -1;
	}
	
	if (read_data(&deletions, del_path, true) < 0) {
		printf("Failed to read deletion file: %s\n", del_path);
		return -1;
	}
	
	printf("Running dataset %d with batch size %d\n", dataset_id, batch_size);
	
	preferenceVector pref;
	for (int i = 0; i < NUM_OBJECTIVES; i++) pref.data[i] = 1;
	
	double t_start = get_time();
	mosp(final_combined_tree, sosp_trees, all_edges, insertions, deletions, pref);
	double t_end = get_time();
	double time_ms = (t_end - t_start) * 1000.0;
	
	fprintf(bench, "Dataset %2d | Batch %4d | Time: %8.2f ms\n",
			dataset_id, batch_size, time_ms);
	
	edge_array_free(&insertions);
	edge_array_free(&deletions);
	fflush(stdout);
	
	return time_ms;
}

int setup_initial_sosp_trees(SOSPTree sosp_trees[], GroupedEdgesArray all_edges) {
	EdgeArray edges;
	int num_vertices = read_data(&edges, "../data_dynamic_100_100/base_60.csv", false);
	
	group_edges(all_edges, edges.data, edges.size);
	
	double s_start, s_end;
	s_start = get_time();
	printf("SETUP: Initial SOSP Trees for each objective\n");
	
	// No critical section needed - logging can be scrambled for speed
	#pragma omp parallel for schedule(static) num_threads(3)
	for(int i = 0; i < NUM_OBJECTIVES; i++) {
		sosp_trees[i].num_vertices = num_vertices;
		dijkstra_build_SOSP(&sosp_trees[i], all_edges, 0, i, num_vertices);
		// fprintf(fp, "Initial SOSP Tree for objective %d:\n", i);
		// print_SOSP_tree(fp, &sosp_trees[i]);
	}
	
	s_end = get_time();
	printf("Setup time: %.2f ms\n\n", (s_end - s_start) * 1000.0);
	
	edge_array_free(&edges);
	return num_vertices;
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		printf("Usage: %s -n [iterations]\n", argv[0]);
		printf("Examples:\n");
		printf("  %s -n 1000 6\n", argv[0]);
		printf("  %s -n 500 6\n", argv[0]);
		return 1;
	}
	
	omp_set_num_threads(6);
	
	printf("========================================\n");
	printf("OpenMP MOSP with Parallel Propagation\n");
	printf("========================================\n");
	printf("Threads: %d (optimized for 3 objectives)\n", omp_get_max_threads());
	printf("Features: Parallel tree updates + parallel BFS propagation\n");
	printf("========================================\n\n");
	
	FILE *bench = fopen("benchmark_log.txt", "w");
	if (!bench) { perror("fopen log"); return 1; }
	
	FILE *out = fopen("output.txt", "w");
	if (!out) { perror("cannot open output.txt"); return 1; }
	
	fp = fopen("log.txt", "w");
	if (!fp) { perror("Cannot open file"); return 1; }
	
	int batch_size = -1;
	int iterations = -1;
	
	if (strcmp(argv[1], "-n") == 0 && argc >= 3) {
		if (strcmp(argv[2], "all") == 0) {
			iterations = 6;
			batch_size = 1000;
		} else {
			batch_size = atoi(argv[2]);
			iterations = 6;
			if (argc >= 4) iterations = atoi(argv[3]);
		}
	}
	
	GroupedEdgesArray all_edges = {NULL};
	SOSPTree sosp_trees[NUM_OBJECTIVES];
	SOSPTree final_combined_tree;
	
	int num_vertices = setup_initial_sosp_trees(sosp_trees, all_edges);
	
	double total = 0.0;
	int valid_runs = 0;
	
	for (int d = 1; d <= iterations; d++) {
		double time = run_one_test(d, batch_size, iterations, bench, &final_combined_tree, sosp_trees, all_edges);
		if (time > 0) {
			total += time;
			valid_runs++;
		}
	}
	
	fprintf(out, "Final Combined SOSP Tree:\n");
	print_SOSP_tree(out, &final_combined_tree);
	
	if (valid_runs > 0) {
		printf("\n========================================\n");
		printf("Performance Summary\n");
		printf("========================================\n");
		printf("Batch size: %d\n", batch_size);
		printf("Valid runs: %d\n", valid_runs);
		printf("Average time: %.2f ms\n", total / valid_runs);
		printf("Total time: %.2f ms\n", total);
		printf("========================================\n\n");
		
		fprintf(bench, "\n========================================\n");
		fprintf(bench, "Configuration: %dx%d (batch=%d, threads=%d)\n", valid_runs, batch_size, batch_size, 3);
		fprintf(bench, "Average Time: %.2f ms\n", total / valid_runs);
		fprintf(bench, "Total Time: %.2f ms\n", total);
		fprintf(bench, "========================================\n");
	}
	
	fclose(bench);
	fclose(out);
	fclose(fp);
	
	return 0;
}
