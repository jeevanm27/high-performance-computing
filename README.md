# Multi-Objective Shortest Path (MOSP) - A HPC Project

This project implements the Multi-Objective Shortest Path (MOSP) algorithm with two different parallel computing approaches: **OpenMP**, and **MPI**, along with a **sequential** implementation. The implementation handles dynamic graph updates (edge insertions and deletions) efficiently.

---

## 📁 Project Structure

```
HPC_project/
├── Sequential/          # Serial baseline implementation
├── OpenMP/              # Shared-memory parallel version
├── MPI/                 # Distributed-memory parallel version
├── data_dynamic_*/      # Test datasets (different batch sizes and iterations)
├── data-processing/     # Data generation and preprocessing scripts
└── results/             # Benchmark results and analysis
```

---

## 🔧 Prerequisites

### Required Software
- **GCC/G++**: Version 7.0 or higher (for C99/C11 support)
- **OpenMP**: Included with most modern GCC installations
- **MPI**: OpenMPI or MPICH implementation
  - Install on Linux: `sudo apt-get install openmpi-bin libopenmpi-dev`
  - Install on Windows: Use Microsoft MPI or WSL with OpenMPI

### Verify Installations
```bash
# Check GCC
gcc --version

# Check OpenMP support
gcc -fopenmp --version

# Check MPI
mpicc --version
mpiexec --version
```

---

## 🚀 How to Build and Run

### 1️⃣ Sequential Version

Navigate to the Sequential directory:
```bash
cd Sequential
```

**Compile:**
```bash
gcc -o seq run.c MOSP.c SOSP.c -lm
```

**Run:**
```bash
# Run with specific batch size and iterations
./seq -n <batch_size> <iterations>

# Examples:
./seq -n 1000 10      # 1000 changes per batch, 10 iterations
./seq -n 500 20       # 500 changes per batch, 20 iterations
./seq -n 100 100      # 100 changes per batch, 100 iterations
```

**Output:**
- `benchmark_log.txt` - Timing results for each iteration
- `output.txt` - Final combined SOSP tree
- `log.txt` - Detailed logs of tree updates

---

### 2️⃣ OpenMP Version

Navigate to the OpenMP directory:
```bash
cd OpenMP
```

**Compile:**
```bash
gcc -o openmp run.c -fopenmp -lm
```

**Run:**
```bash
# Set number of threads (optional, default uses all available cores)
export OMP_NUM_THREADS=8

# Run with specific configuration
./openmp -n <batch_size> <iterations>

# Examples:
./openmp -n 1000 10      # 1000 changes per batch, 10 iterations
./openmp -n 500 20       # 500 changes per batch, 20 iterations
OMP_NUM_THREADS=4 ./openmp -n 1000 10  # Force 4 threads
```

**Output:**
- `benchmark_log.txt` - Timing results and speedup analysis
- `output.txt` - Final combined SOSP tree
- `log.txt` - Detailed logs with thread information

---

### 3️⃣ MPI Version

Navigate to the MPI directory:
```bash
cd MPI
```

**Compile:**
```bash
mpicc -o mpi_mosp run.c MOSP.c SOSP.c -lm
```

**Run:**
```bash
# Run with specific number of processes
mpiexec -n <num_processes> ./mpi_mosp -n <batch_size> <iterations>

# Examples:
mpiexec -n 3 ./mpi_mosp -n 1000 10      # 3 processes, 1000 changes/batch, 10 iterations

# Note: Number of processes should be divisible by NUM_OBJECTIVES (3)
```

**Output:**
- `benchmark_log_mpi.txt` - Timing results from rank 0
- `output_mpi.txt` - Final combined SOSP tree
- `log_mpi.txt` - Detailed MPI communication logs

---

## ⚙️ Configuration Options

### Modifying Graph Parameters

Edit `MOSP.h` in each directory to change:

```c
#define MAX_VERTICES 16000    // Maximum number of vertices
#define NUM_OBJECTIVES 3      // Number of optimization objectives
```

### Thread Configuration (OpenMP only)

```bash
# Set threads before running
export OMP_NUM_THREADS=8

# Or inline
OMP_NUM_THREADS=4 ./openmp -n 1000 10
```

### MPI Process Distribution

The MPI implementation uses a team-based approach:
- Total processes should ideally be `NUM_OBJECTIVES * K` where K ≥ 1
- Each objective (SOSP tree) gets its own team of K processes
- Example: 3 processes = 1 processes per objective (for 3 objectives)


### Analyzing Results

Results are collected in `results/` directory:
```bash
cd results
python3 results.py  # Generate performance plots and analysis
```
---
