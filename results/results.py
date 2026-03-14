import matplotlib.pyplot as plt
import numpy as np
import re
import os


NUM_PROCESSORS = 3 
SEQ_LOG_FILE = "benchmark_seq.txt"
MPI_LOG_FILE = "benchmark_mpi.txt"
OPENMP_LOG_FILE = "benchmark_openmp.txt"

def parse_benchmark_log(file_path):
    if not os.path.exists(file_path):
        print(f"Error: Log file not found at {file_path}")
        return {}

    data = {}
    pattern = r"Dataset\s+(\d+)\s+\|\s+Batch\s+(\d+)\s+\|\s+Time:\s+([\d\.]+)\s+ms"
    
    with open(file_path, 'r') as f:
        content = f.read()

    for line in content.strip().split('\n'):
        match = re.search(pattern, line)
        if match:
            batch_size = int(match.group(2))
            time_ms = float(match.group(3))
            
            if batch_size not in data:
                data[batch_size] = []
            data[batch_size].append(time_ms)
            
    if not data:
        print(f"Warning: No valid data lines found in {file_path}.")
        return {}

    averaged_data = {k: np.mean(v) for k, v in data.items()}
    return dict(sorted(averaged_data.items()))

def plot_performance_graphs(batches, t_seq, t_mpi, t_openmp, num_processors):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'MOSP Algorithm Performance Analysis', fontsize=16)

    axes[0].plot(batches, t_seq, 'o--', label='Sequential', color='red', linewidth=2, alpha=0.7)
    axes[0].plot(batches, t_mpi, 's-', label='MPI', color='blue', linewidth=2)
    axes[0].plot(batches, t_openmp, '^-', label='OpenMP', color='green', linewidth=2)
    
    axes[0].set_xlabel('Batch Size (Edges)')
    axes[0].set_ylabel('Execution Time (ms)')
    axes[0].set_title('Metric 1: Execution Time')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    speedup_mpi = [ts / tm for ts, tm in zip(t_seq, t_mpi)]
    speedup_openmp = [ts / to for ts, to in zip(t_seq, t_openmp)]
    
    axes[1].plot(batches, speedup_mpi, 's-', color='blue', linewidth=2, label='MPI Speedup')
    axes[1].plot(batches, speedup_openmp, '^-', color='green', linewidth=2, label='OpenMP Speedup')
    axes[1].axhline(y=1.0, color='red', linestyle=':', label='Baseline (1.0)', alpha=0.5)
    
    axes[1].set_xlabel('Batch Size (Edges)')
    axes[1].set_ylabel('Speedup Factor')
    axes[1].set_title('Metric 2: Speedup ')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    cost_seq = t_seq  # 1 processor * T_seq
    cost_mpi = [num_processors * tm for tm in t_mpi]
    cost_openmp = [num_processors * to for to in t_openmp]
    
    axes[2].plot(batches, cost_seq, 'o--', color='red', linewidth=2, label='Sequential Cost', alpha=0.5)
    axes[2].plot(batches, cost_mpi, 's-', color='blue', linewidth=2, label='MPI Cost')
    axes[2].plot(batches, cost_openmp, '^-', color='green', linewidth=2, label='OpenMP Cost')
    
    axes[2].set_xlabel('Batch Size (Edges)')
    axes[2].set_ylabel('Cost (ms)')
    axes[2].set_title('Metric 3: Cost Efficiency ')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('mosp_performance_comparison.png')
    print("Graph saved as 'mosp_performance_comparison.png'")
    plt.show() 

if __name__ == "__main__":
    # 1. Parse Logs
    print(f"Reading {SEQ_LOG_FILE}...")
    seq_data = parse_benchmark_log(SEQ_LOG_FILE)
    
    print(f"Reading {MPI_LOG_FILE}...")
    mpi_data = parse_benchmark_log(MPI_LOG_FILE)
    
    print(f"Reading {OPENMP_LOG_FILE}...")
    openmp_data = parse_benchmark_log(OPENMP_LOG_FILE)

    # 2. Align Data (Find common batch sizes across all 3 logs)
    common_batches = sorted(list(
        set(seq_data.keys()) & 
        set(mpi_data.keys()) & 
        set(openmp_data.keys())
    ))
    
    if not common_batches:
        print("Error: No matching batch sizes found in all three logs. Cannot plot.")
        print(f"Seq Batches: {list(seq_data.keys())}")
        print(f"MPI Batches: {list(mpi_data.keys())}")
        print(f"OpenMP Batches: {list(openmp_data.keys())}")
    else:
        print(f"Plotting data for batch sizes: {common_batches}")
        
        # Extract aligned lists
        t_seq = [seq_data[b] for b in common_batches]
        t_mpi = [mpi_data[b] for b in common_batches]
        t_openmp = [openmp_data[b] for b in common_batches]
        
        plot_performance_graphs(common_batches, t_seq, t_mpi, t_openmp, NUM_PROCESSORS)