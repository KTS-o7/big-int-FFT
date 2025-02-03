#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>

using namespace std;

// Constants
const double G = 6.67430e-11;  // Gravitational constant
const double TIME_STEP = 0.01; // Time step for simulation
const int NUM_BODIES = 1000;   // Number of bodies
const int STEPS = 10000;       // Simulation steps
const int BLOCK_SIZE = 256;    // CUDA thread block size

// Structure to represent a body (planet, particle)
struct Body {
    double x, y, vx, vy, mass;
};

// Structure for aligned force data
struct Force {
    double fx;
    double fy;
};

// CUDA kernel for computing forces
__global__ void computeForcesKernel(const Body* bodies, Force* forces, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        double fx = 0.0;
        double fy = 0.0;

        // Compute forces with all other bodies
        for (int j = 0; j < n; j++) {
            if (idx != j) {
                double dx = bodies[j].x - bodies[idx].x;
                double dy = bodies[j].y - bodies[idx].y;
                double dist = sqrt(dx * dx + dy * dy + 1e-9);
                double force = (G * bodies[idx].mass * bodies[j].mass) / (dist * dist);

                fx += force * dx / dist;
                fy += force * dy / dist;
            }
        }

        forces[idx].fx = fx;
        forces[idx].fy = fy;
    }
}

// CUDA kernel for updating positions
__global__ void updateBodiesKernel(Body* bodies, const Force* forces, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        bodies[idx].vx += (forces[idx].fx / bodies[idx].mass) * TIME_STEP;
        bodies[idx].vy += (forces[idx].fy / bodies[idx].mass) * TIME_STEP;
        bodies[idx].x += bodies[idx].vx * TIME_STEP;
        bodies[idx].y += bodies[idx].vy * TIME_STEP;
    }
}

// Function to initialize bodies randomly
void initializeBodies(vector<Body>& bodies) {
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

int main() {
    vector<Body> bodies(NUM_BODIES);
    vector<Force> forces(NUM_BODIES);

    // Initialize bodies on host
    initializeBodies(bodies);

    // Allocate memory on device
    Body* d_bodies;
    Force* d_forces;
    cudaMalloc(&d_bodies, NUM_BODIES * sizeof(Body));
    cudaMalloc(&d_forces, NUM_BODIES * sizeof(Force));

    // Copy initial data to device
    cudaMemcpy(d_bodies, bodies.data(), NUM_BODIES * sizeof(Body), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    int numBlocks = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    // Main simulation loop
    for (int step = 0; step < STEPS; step++) {
        // Compute forces
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_forces, NUM_BODIES);
        
        // Update positions
        updateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(d_bodies, d_forces, NUM_BODIES);
        
        // Synchronize to ensure step completion
        cudaDeviceSynchronize();
    }

    // Copy final results back to host
    cudaMemcpy(bodies.data(), d_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);

    // Calculate final statistics
    double total_kinetic_energy = 0.0;
    double total_distance = 0.0;

    for (const auto& body : bodies) {
        double velocity = sqrt(body.vx * body.vx + body.vy * body.vy);
        total_kinetic_energy += 0.5 * body.mass * velocity * velocity;
        total_distance += sqrt(body.x * body.x + body.y * body.y);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Print statistics in a consistent format
    cout << "=== Simulation Statistics ===\n";
    cout << "Implementation: CUDA\n";
    cout << "Bodies: " << NUM_BODIES << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "Block Size: " << BLOCK_SIZE << "\n";
    cout << "Blocks: " << numBlocks << "\n";
    cout << "Execution Time: " << duration.count() << "\n";
    cout << "Kinetic Energy: " << scientific << setprecision(3) << total_kinetic_energy << "\n";
    cout << "Avg Distance: " << fixed << setprecision(2) << total_distance/NUM_BODIES << "\n";
    cout << "=== End Statistics ===\n";

    // Clean up
    cudaFree(d_bodies);
    cudaFree(d_forces);
    return 0;
}
