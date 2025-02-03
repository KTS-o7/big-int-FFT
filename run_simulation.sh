#!/bin/bash

# This script will run all the different implementations of the n-body simulation
# and display a comparison table of execution times

# Function to print horizontal line
print_line() {
    printf "%.s-" {1..60}
    echo
}

# Function to print header
print_header() {
    echo
    echo "Running N-body Simulation Implementations"
    print_line
}

# Function to compile and run each implementation
run_implementation() {
    local name=$1
    local cmd=$2
    local run_cmd=$3
    
    echo "Compiling $name implementation..."
    eval $cmd
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to compile $name implementation"
        return 1
    fi
    
    echo "Running $name implementation..."
    # Run 3 times and take the average
    local times=""
    local runs=3
    
    for ((i=1; i<=$runs; i++)); do
        echo "Run $i of $runs..."
        # Extract execution time from the output
        local time=$(eval $run_cmd | grep "^Execution Time: " | awk '{print $3}')
        if [[ ! -z "$time" && "$time" != "0" && "$time" != "0.0" && "$time" != "0.00" ]]; then
            times="$times $time"
        else
            echo "Warning: Invalid time value detected in run $i"
            i=$((i-1)) # Retry this run
        fi
    done
    
    # Calculate average using awk, with error checking
    if [[ ! -z "$times" ]]; then
        avg_time=$(echo $times | awk '{sum=0; n=0; for(i=1;i<=NF;i++) {sum+=$i; n++} if(n>0) print sum/n; else print "0"}')
        avg_time=$(printf "%.2f" $avg_time)
        echo "$name: $avg_time ms (average of $runs runs)"
        echo "$avg_time" > "${name}_time.txt"
    else
        echo "Error: No valid time measurements for $name"
        echo "0.01" > "${name}_time.txt" # Prevent division by zero
    fi
    print_line
}

# Print header
print_header

# Compile and run Serial version
run_implementation "Serial" \
    "g++ -O3 Serial/serial_nbody.cpp -o serial_nbody" \
    "./serial_nbody"

# Compile and run OpenMP version
run_implementation "OpenMP" \
    "g++ -O3 -fopenmp OpenMP/openmp_nbody.cpp -o openmp_nbody" \
    "./openmp_nbody"

# Compile and run Pthreads version
run_implementation "Pthreads" \
    "g++ -O3 Pthread/pthread_nbody.cpp -o pthread_nbody -pthread" \
    "./pthread_nbody"

# Compile and run MPI version
run_implementation "MPI" \
    "mpic++ -O3 MPI/mpi_nbody.cpp -o mpi_nbody" \
    "mpirun -np 4 ./mpi_nbody"

# Compile and run CUDA version (if NVIDIA GPU is available)
if command -v nvcc &> /dev/null; then
    run_implementation "CUDA" \
        "nvcc -O3 CUDA/cuda_nbody.cu -o cuda_nbody" \
        "./cuda_nbody"
fi

# Print comparison table
echo
echo "Performance Comparison"
print_line
printf "%-15s | %-15s | %-15s\n" "Implementation" "Time (ms)" "Speedup"
print_line

# Get serial time as baseline
baseline=$(cat Serial_time.txt)

# Print results for each implementation
for impl in Serial OpenMP Pthreads MPI CUDA; do
    if [ -f "${impl}_time.txt" ]; then
        time=$(cat "${impl}_time.txt")
        # Calculate speedup using awk with error checking
        speedup=$(awk -v base=$baseline -v t=$time 'BEGIN {
            if (t > 0) printf "%.2f", base/t;
            else printf "1.00"
        }')
        printf "%-15s | %-15.2f | %-15s\n" "$impl" "$time" "$speedup"
    fi
done

print_line

# Clean up temporary files and executables
rm -f *_time.txt serial_nbody openmp_nbody pthread_nbody mpi_nbody cuda_nbody

echo
echo "Note: Times are averages of 3 runs. Speedup is relative to serial implementation."
echo "MPI was run with 4 processes. Actual performance may vary based on hardware."
