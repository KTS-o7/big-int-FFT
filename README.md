# N-Body Simulation Parallel Implementations

This project implements an N-body gravitational simulation using different parallel programming paradigms. The simulation calculates the gravitational forces between N bodies (particles/planets) and updates their positions and velocities accordingly.

## Algorithm Overview

### Basic N-body Algorithm

1. Initialize N bodies with random positions, zero initial velocities, and random masses
2. For each time step:
   - Calculate forces between all pairs of bodies
   - Update velocities based on forces
   - Update positions based on velocities
3. Calculate final statistics (kinetic energy, average distance)

### Physics

- Uses Newton's law of universal gravitation: F = G _ (m1 _ m2) / r²
- Force components are calculated along x and y axes
- The following approximations are made using the eulers method:
- Velocity updates use v = v + (F/m) \* dt
- Position updates use p = p + v \* dt
  Here dt is the time step. In the code we have used `TIME_STEP` as the time step.

### Computational Complexity

- Force calculation: O(n²) per time step
- Position/velocity updates: O(n) per time step
- Overall complexity: O(n² \* timesteps)

## Parallel Implementations

### 1. Serial (Baseline)

- Single-threaded implementation
- Used as baseline for speedup calculations
- Direct implementation of the algorithm without parallelization

### 2. OpenMP

- Shared memory parallelization
- Key parallelization strategies:
  - Parallel force computation using thread-local storage
  - Dynamic scheduling for load balancing
  - Reduction for final statistics calculation
  - False sharing prevention using padded force structures
- Uses `#pragma omp parallel for` directives
- Automatically handles thread creation and work distribution

### 3. Pthreads

- Low-level thread-based parallelization
- Features:
  - Manual thread management
  - Cyclic work distribution
  - Barrier synchronization between phases
  - Atomic operations for force updates
  - Padded force structure to prevent false sharing
- Each thread handles a subset of bodies throughout the simulation

### 4. MPI

- Distributed memory parallelization
- Implementation details:
  - Domain decomposition: bodies divided among processes
  - Custom MPI datatype for Body structure
  - Collective communications:
    - `MPI_Allreduce` for force combination
    - `MPI_Allgatherv` for position updates
  - Local computations followed by global synchronization
  - Root process handles I/O and statistics

### 5. CUDA

- GPU parallelization
- Features:
  - Two CUDA kernels:
    1. `computeForcesKernel`: One thread per body
    2. `updateBodiesKernel`: One thread per body
  - Coalesced memory access patterns
  - Efficient data transfer between CPU and GPU
  - Block size optimization for GPU occupancy
  - Minimal host-device communication

## Performance Considerations

### Memory Access Patterns

- OpenMP: Uses padded structures to prevent false sharing
- Pthreads: Employs aligned and padded force structures
- MPI: Minimizes communication overhead with efficient collective operations
- CUDA: Optimizes for coalesced memory access

### Load Balancing

- OpenMP: Dynamic scheduling for irregular workloads
- Pthreads: Cyclic distribution of bodies
- MPI: Even distribution of bodies across processes
- CUDA: Equal work per thread with efficient block sizing

### Synchronization

- OpenMP: Implicit barriers and critical sections
- Pthreads: Explicit barriers between computation phases
- MPI: Collective communication operations
- CUDA: Kernel synchronization and device synchronization

## Building and Running

### Prerequisites

- C++ compiler with OpenMP support
- CUDA toolkit (for GPU version)
- MPI implementation (OpenMPI or MPICH)
- POSIX threads support

### Compilation

```bash
# build all versions
./run_simulation.sh
```

## Performance Analysis

The script runs each implementation 3 times and reports:

- Average execution time
- Relative speedup compared to serial version
- Implementation-specific metrics (threads, processes, block size)
- Final simulation statistics (energy, average distance)

## Parameters

All implementations use the same simulation parameters:

- `NUM_BODIES`: Number of bodies (default: 1000)
- `STEPS`: Number of simulation steps (default: 10000)
- `TIME_STEP`: Simulation time step (default: 0.01)
- `G`: Gravitational constant

## Future Improvements

1. Implement Barnes-Hut algorithm to reduce complexity to O(n log n)
2. Add visualization capabilities
3. Hybrid parallelization (MPI + OpenMP)
4. CUDA shared memory optimization
5. Support for 3D simulation

## Results

```bash
Running N-body Simulation Implementations
------------------------------------------------------------
Compiling Serial implementation...
Running Serial implementation...
Run 1 of 3...
Run 2 of 3...
Run 3 of 3...
Serial: 24991.70 ms (average of 3 runs)
------------------------------------------------------------
Compiling OpenMP implementation...
Running OpenMP implementation...
Run 1 of 3...
Run 2 of 3...
Run 3 of 3...
OpenMP: 5696.67 ms (average of 3 runs)
------------------------------------------------------------
Compiling Pthreads implementation...
Running Pthreads implementation...
Run 1 of 3...
Run 2 of 3...
Run 3 of 3...
Pthreads: 12469.30 ms (average of 3 runs)
------------------------------------------------------------
Compiling MPI implementation...
Running MPI implementation...
Run 1 of 3...
Run 2 of 3...
Run 3 of 3...
MPI: 12472.30 ms (average of 3 runs)
------------------------------------------------------------
Compiling CUDA implementation...
Running CUDA implementation...
Run 1 of 3...
Run 2 of 3...
Run 3 of 3...
CUDA: 12.67 ms (average of 3 runs)
------------------------------------------------------------

Performance Comparison
------------------------------------------------------------
Implementation  | Time (ms)       | Speedup
------------------------------------------------------------
Serial          | 24991.70        | 1.00
OpenMP          | 5696.67         | 4.39
Pthreads        | 12469.30        | 2.00
MPI             | 12472.30        | 2.00
CUDA            | 12.67           | 1972.51
------------------------------------------------------------

Note: Times are averages of 3 runs. Speedup is relative to serial implementation.
MPI was run with 4 processes. Actual performance may vary based on hardware.
```
