#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOSP.h"
#include "read_csv.c"   // Assuming these exist in the dir
#include "dijkstra.c"   // Assuming these exist in the dir
#include <mpi.h>

// --- FIX: Remove global file pointer definition here. 
// --- It is defined in main() (Rank 0) and used as extern elsewhere.
FILE *fp = NULL; // Keep fp definition here or move to main, but it's used globally.

// Wrapper for MPI Timing
double get_time() {
    return MPI_Wtime();
}


// Function to run a single test iteration
double run_one_test(int dataset_id, int batch_size, int iterations, FILE *bench, 
                    SOSPTree *final_combined_tree, SOSPTree sosp_trees[], 
                    GroupedEdgesArray all_edges, MPI_Comm team_comm) {
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Construct file paths based on dataset ID (Same as sequential)
    char ins_path[256], del_path[256];
    snprintf(ins_path, sizeof(ins_path), "../data_dynamic_%d_%d/alt_batches/alt_insert_%d.csv", batch_size, iterations, dataset_id);
    snprintf(del_path, sizeof(del_path), "../data_dynamic_%d_%d/alt_batches/alt_delete_%d.csv", batch_size, iterations, dataset_id);

    EdgeArray insertions, deletions;
    
    if (read_data(&insertions, ins_path, false) < 0) {
        if(rank == 0) printf("Failed to read insertion file: %s\n", ins_path);
        return -1;
    }
    if (read_data(&deletions, del_path, true) < 0) {
        if(rank == 0) printf("Failed to read deletion file: %s\n", del_path);
        return -1;
    }

    if(rank == 0) printf("Running dataset %d with batch size %d\n", dataset_id, batch_size);

    preferenceVector pref;
    for (int i = 0; i < NUM_OBJECTIVES; i++) pref.data[i] = 1;

    // --- START TIMING ---
    MPI_Barrier(MPI_COMM_WORLD); // Sync before starting timer
    double t_start = get_time();

    // Call the MPI Parallel MOSP function
mosp_parallel(final_combined_tree, sosp_trees, all_edges, insertions, deletions, pref, team_comm);
    // Sync after completion to capture the time of the slowest rank
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = get_time();
    // --- END TIMING ---

    double time_ms = (t_end - t_start) * 1000.0;

    // Only Rank 0 logs the metrics
    if (rank == 0) {
        fprintf(bench, "Dataset %2d | Batch %4d | Time: %8.2f ms\n",
               dataset_id, batch_size, time_ms);
        fflush(bench);
        fflush(stdout);
    }

    edge_array_free(&insertions);
    edge_array_free(&deletions);

    return time_ms;
}

int setup_initial_sosp_trees_parallel(SOSPTree sosp_trees[], GroupedEdgesArray all_edges, MPI_Comm team_comm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 1. Read Base Graph (All ranks read to build local adjacency lists)
    EdgeArray edges;
    int num_vertices = read_data(&edges, "../data_dynamic_100_100/base_60.csv", false);

    group_edges(all_edges, edges.data, edges.size);
    
    double s_start, s_end;
    
    MPI_Barrier(MPI_COMM_WORLD);
    s_start = get_time();
    
    if(rank == 0) printf("SETUP: Initial SOSP Trees (Parallel distributed by Objective)\n");
    
    // 2. Distribute Initial Work
    int team_color = rank % NUM_OBJECTIVES;

    for(int i=0; i<NUM_OBJECTIVES; i++){
        sosp_trees[i].num_vertices = num_vertices;
        
        // Only the team responsible for this objective runs the initial Dijkstra
        if (team_color == i) {
            dijkstra_build_SOSP_parallel(&sosp_trees[i], all_edges, 0, i, num_vertices, team_comm);
            printf("[Rank %d] Initial SOSP Tree Built for objective %d\n", rank, i);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    s_end = get_time();
    
    if(rank == 0) printf( "Setup Time %.6f s\n\n", s_end - s_start);
    
    edge_array_free(&edges);
    return num_vertices;
}

// Global declaration for the file pointer used in MOSP.c (must be defined in main)
FILE *out = NULL;

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if(rank == 0) {
            printf("Usage: %s -n <batch_size> [iterations]  or  -n all\n", argv[0]);
            printf("Examples:\n");
            printf("  mpiexec -np 3 %s -n 1000      # 10 runs of 1000 changes\n", argv[0]);
            printf("  mpiexec -np 3 %s -n 500 20    # 20 iterations of 500 changes\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // --- FIX: Only Rank 0 opens the files, which defines the global pointers ---
    FILE *bench = NULL;

    if (rank == 0) {
        // bench pointer is local to main scope, out/fp are global scope (for MOSP.c)
        bench = fopen("benchmark_log_mpi.txt", "w");
        if (!bench) { perror("fopen log"); MPI_Abort(MPI_COMM_WORLD, 1); }

        out = fopen("output_mpi.txt", "w"); // DEFINES THE GLOBAL 'out' USED BY MOSP.C
        if (!out) { perror("cannot open output/txt"); MPI_Abort(MPI_COMM_WORLD, 1); }

        fp = fopen("log_mpi.txt", "w"); 
        if (!fp) { perror("Cannot open file"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

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

    // Create Team Communicator for Task Parallelism
    int color = rank % NUM_OBJECTIVES;
    MPI_Comm team_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &team_comm);

    // Data Structures
    static GroupedEdgeNode* all_edges[MAX_VERTICES] = {NULL}; // Global graph
    SOSPTree sosp_trees[NUM_OBJECTIVES];
    SOSPTree final_combined_tree;

    // Setup
    int num_vertices = setup_initial_sosp_trees_parallel(sosp_trees, all_edges, team_comm);

    // Run Tests
    double total = 0.0;
    for (int d = 1; d <= iterations; d++) {
        total += run_one_test(d, batch_size, iterations, bench, &final_combined_tree, sosp_trees, all_edges, team_comm);
    }

    // Reporting (Rank 0)
    if (rank == 0) {
        fprintf(out, "Final Combined SOSP Tree:\n");
        // FIX: The print_SOSP_tree call in MOSP.c will use the global 'out' defined here.
        print_SOSP_tree(out, &final_combined_tree); 

        printf("\nAverage Time for batch size %d over %d iterations: %.2f ms\n\n", batch_size, iterations, total / iterations);
        
        fprintf(bench, "+----------------------+----------+----------+----------+\n");
        fprintf(bench, "| Configuration        | Batch    | Avg Time | Total    |\n");
        fprintf(bench, "+----------------------+----------+----------+----------+\n");
        fprintf(bench, "| MPI: %dx%d           | %8d | %4.4f | %4.4f |\n", iterations, batch_size, batch_size, total/iterations, total);
        fprintf(bench, "+----------------------+----------+----------+----------+\n");

        fclose(bench);
        fclose(out);
        fclose(fp);
    }

    // Cleanup
    freeGroupedEdges(all_edges, MAX_VERTICES);
    MPI_Comm_free(&team_comm);
    MPI_Finalize();

    return 0;
}