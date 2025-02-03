#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <mpi.h>

using namespace std;

// Constants
const double G = 6.67430e-11;  // Gravitational constant
const double TIME_STEP = 0.01; // Time step for simulation
const int NUM_BODIES = 1000;   // Number of bodies
const int STEPS = 10000;       // Simulation steps

// Structure to represent a body (planet, particle)
struct Body {
    double x, y, vx, vy, mass;
};

// Add this structure to properly align and pack force data
struct alignas(16) Force {
    double fx;
    double fy;
};

// Custom MPI datatype for Body structure
MPI_Datatype MPI_BODY;

void initMPIType() {
    // Create MPI datatype for Body structure
    const int nitems = 5;
    int blocklengths[5] = {1, 1, 1, 1, 1};
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[5];

    offsets[0] = offsetof(Body, x);
    offsets[1] = offsetof(Body, y);
    offsets[2] = offsetof(Body, vx);
    offsets[3] = offsetof(Body, vy);
    offsets[4] = offsetof(Body, mass);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
}

// Function to compute forces for a subset of bodies
void computeForces(const vector<Body>& bodies, vector<Force>& forces,
                  int start, int end) {
    int n = bodies.size();
    
    // Reset forces for assigned bodies
    for (int i = start; i < end; i++) {
        forces[i] = {0.0, 0.0};
    }

    // Compute forces
    for (int i = start; i < end; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dist = sqrt(dx * dx + dy * dy + 1e-9);
                double force = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);

                forces[i].fx += force * dx / dist;
                forces[i].fy += force * dy / dist;
            }
        }
    }
}

// Function to update positions and velocities for a subset of bodies
void updateBodies(vector<Body>& bodies, const vector<Force>& forces,
                 int start, int end) {
    for (int i = start; i < end; i++) {
        bodies[i].vx += (forces[i].fx / bodies[i].mass) * TIME_STEP;
        bodies[i].vy += (forces[i].fy / bodies[i].mass) * TIME_STEP;
        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
    }
}

// Function to initialize bodies randomly
void initializeBodies(vector<Body>& bodies, int rank) {
    if (rank == 0) {
        srand(time(0));
        for (int i = 0; i < bodies.size(); i++) {
            bodies[i] = {
                (double)(rand() % 1000),
                (double)(rand() % 1000),
                0.0, 0.0,
                (double)(rand() % 100 + 1)
            };
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize MPI custom datatype
    initMPIType();

    vector<Body> bodies(NUM_BODIES);
    vector<Force> forces(NUM_BODIES, {0.0, 0.0});
    vector<Force> local_forces(NUM_BODIES, {0.0, 0.0});
    
    // Timing variables
    double start_time = 0.0;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    // Initialize bodies on rank 0 and broadcast to all processes
    initializeBodies(bodies, rank);
    MPI_Bcast(bodies.data(), NUM_BODIES, MPI_BODY, 0, MPI_COMM_WORLD);

    // Calculate work distribution
    int bodies_per_proc = NUM_BODIES / size;
    int start_idx = rank * bodies_per_proc;
    int end_idx = (rank == size - 1) ? NUM_BODIES : start_idx + bodies_per_proc;
    int sendcount = end_idx - start_idx;

    // Gather send counts and compute displacements for Allgatherv
    vector<int> sendcounts(size);
    MPI_Allgather(&sendcount, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> displs(size);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }

    // Main simulation loop
    for (int step = 0; step < STEPS; step++) {
        // Reset local forces
        fill(local_forces.begin(), local_forces.end(), Force{0.0, 0.0});
        
        // Compute forces for assigned bodies
        computeForces(bodies, local_forces, start_idx, end_idx);

        // Create separate arrays for fx and fy components
        vector<double> local_fx(NUM_BODIES), local_fy(NUM_BODIES);
        vector<double> global_fx(NUM_BODIES), global_fy(NUM_BODIES);

        // Copy forces to separate arrays
        for (int i = 0; i < NUM_BODIES; i++) {
            local_fx[i] = local_forces[i].fx;
            local_fy[i] = local_forces[i].fy;
        }

        // Reduce x and y components separately
        MPI_Allreduce(local_fx.data(), global_fx.data(), NUM_BODIES,
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_fy.data(), global_fy.data(), NUM_BODIES,
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Copy reduced forces back to force vector
        for (int i = 0; i < NUM_BODIES; i++) {
            forces[i].fx = global_fx[i];
            forces[i].fy = global_fy[i];
        }

        // Update positions for assigned bodies
        updateBodies(bodies, forces, start_idx, end_idx);

        // Share updated positions using Allgatherv
        MPI_Allgatherv(
            &bodies[start_idx],  // Send buffer (this process's chunk)
            sendcount,           // Number of elements to send
            MPI_BODY,            // Send type
            bodies.data(),       // Receive buffer
            sendcounts.data(),   // Array of receive counts
            displs.data(),       // Array of displacements
            MPI_BODY,            // Receive type
            MPI_COMM_WORLD
        );
    }

    // Calculate final statistics
    double local_kinetic_energy = 0.0;
    double local_distance = 0.0;
    for (int i = start_idx; i < end_idx; i++) {
        double velocity = sqrt(bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
        local_kinetic_energy += 0.5 * bodies[i].mass * velocity * velocity;
        local_distance += sqrt(bodies[i].x * bodies[i].x + bodies[i].y * bodies[i].y);
    }

    double total_kinetic_energy = 0.0;
    double total_distance = 0.0;
    MPI_Reduce(&local_kinetic_energy, &total_kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_distance, &total_distance, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print statistics on rank 0
    if (rank == 0) {
        double end_time = MPI_Wtime();
        double duration = (end_time - start_time) * 1000; // Convert to milliseconds

        cout << "=== Simulation Statistics ===\n";
        cout << "Implementation: MPI\n";
        cout << "Bodies: " << NUM_BODIES << "\n";
        cout << "Steps: " << STEPS << "\n";
        cout << "Processes: " << size << "\n";
        cout << "Execution Time: " << fixed << duration << "\n";
        cout << "Kinetic Energy: " << scientific << setprecision(3) << total_kinetic_energy << "\n";
        cout << "Avg Distance: " << fixed << setprecision(2) << total_distance/NUM_BODIES << "\n";
        cout << "=== End Statistics ===\n";
    }

    // Clean up MPI type
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    return 0;
}