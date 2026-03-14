#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOSP.h"
#include "read_csv.c"
#include "dijkstra.c"
#include <time.h>
#include <linux/time.h>

FILE *fp = NULL;
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double run_one_test(int dataset_id, int batch_size, int iterations, FILE *bench, SOSPTree *final_combined_tree, SOSPTree sosp_trees[], GroupedEdgesArray all_edges) {
    char ins_path[256], del_path[256];
    snprintf(ins_path, sizeof(ins_path), "../data_dynamic_%d_%d/alt_batches/alt_insert_%d.csv",batch_size, iterations, dataset_id);
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
    // fprintf(fp, "All edges after grouping:\n");
    // print_grouped_edges(fp, all_edges, false);
    
    double s_start, s_end;
    s_start = get_time();
    printf("SETUP: Initial SOSP Trees for each objective\n");
    for(int i=0; i<NUM_OBJECTIVES; i++){
        sosp_trees[i].num_vertices = num_vertices;
        // Build initial SOSP tree using Dijkstra 
        dijkstra_build_SOSP(&sosp_trees[i], all_edges, 0, i, num_vertices);
        printf("Initial SOSP Tree Built for objective %d\n", i);
        // fprintf(fp, "Initial SOSP Tree for objective %d:\n", i);
        // print_SOSP_tree(fp, &sosp_trees[i]);
    }
    s_end = get_time();
    printf("Setup Time %.6f\n\n", s_end - s_start);
    
    fflush(stdout);
    edge_array_free(&edges);
    return num_vertices;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s -n <batch_size> [iterations]  or  -n all\n", argv[0]);
        printf("Examples:\n");
        printf("  %s -n 1000      # 10 runs of 1000 changes\n", argv[0]);
        printf("  %s -n 500 20    # 20 iterations of 500 changes\n", argv[0]);
        printf("  %s -n all       # all 6 configurations\n", argv[0]);
        return 1;
    }

    FILE *bench = fopen("benchmark_log.txt", "w");
    if (!bench) { perror("fopen log"); return 1; }


    FILE *out = fopen("output.txt", "w");
    if (!out) { perror("cannot open output/txt"); return 1; }

    fp = fopen("log.txt", "w"); 
    if (!fp) { perror("Cannot open file"); return 1; }

    // Parse arguments
    int batch_size = -1;
    int iterations = -1;
    int run_all = 0;

    if (strcmp(argv[1], "-n") == 0 && argc >= 3) {
        if (strcmp(argv[2], "all") == 0) {
            run_all = 1;
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
    for (int d = 1; d <= iterations; d++) {
        total += run_one_test(d, batch_size, iterations, bench, &final_combined_tree, sosp_trees, all_edges);
    }

    fprintf(out, "Final Combined SOSP Tree:\n");
    print_SOSP_tree(out, &final_combined_tree);

    printf("\nAvergage Time for batch size %d over %d iterations: %.2f ms\n\n", batch_size, iterations, total / iterations);
    
    fprintf(bench, "+----------------------+----------+----------+----------+\n");
    fprintf(bench, "| Configuration         | Batch    | Avg Time | Total    |\n");
    fprintf(bench, "+----------------------+----------+----------+----------+\n");
    fprintf(bench, "| Custom: %dx%d       | %8d | %4.4f | %4.4f |\n", iterations, batch_size, batch_size, total/iterations, total);
    fprintf(bench, "+----------------------+----------+----------+----------+\n");

    fclose(bench);
    fclose(out);
    // fclose(fp);

    return 0;
}