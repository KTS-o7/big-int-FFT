#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <omp.h>

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

// Structure to store forces with padding to avoid false sharing
struct alignas(64) PaddedForce {
    double fx;
    double fy;
    char padding[48];  // Pad to 64 bytes
};

// Function to compute gravitational forces
void computeForces(vector<Body>& bodies, vector<PaddedForce>& forces) {
    int n = bodies.size();
    
    // Reset forces
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        forces[i].fx = 0.0;
        forces[i].fy = 0.0;
    }

    // Compute forces using thread-local storage to avoid atomic operations
    #pragma omp parallel
    {
        vector<PaddedForce> local_forces(n);
        
        // Initialize local forces
        for (int i = 0; i < n; i++) {
            local_forces[i].fx = 0.0;
            local_forces[i].fy = 0.0;
        }

        // Compute forces with dynamic scheduling for better load balancing
        #pragma omp for schedule(dynamic, 32)
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dist = sqrt(dx * dx + dy * dy + 1e-9);
                double force = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);

                double fx = force * dx / dist;
                double fy = force * dy / dist;

                // Update local forces
                local_forces[i].fx += fx;
                local_forces[i].fy += fy;
                local_forces[j].fx -= fx;
                local_forces[j].fy -= fy;
            }
        }

        // Reduce local forces to global forces
        #pragma omp critical
        {
            for (int i = 0; i < n; i++) {
                forces[i].fx += local_forces[i].fx;
                forces[i].fy += local_forces[i].fy;
            }
        }
    }
}

// Function to update positions and velocities
void updateBodies(vector<Body>& bodies, const vector<PaddedForce>& forces) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].vx += (forces[i].fx / bodies[i].mass) * TIME_STEP;
        bodies[i].vy += (forces[i].fy / bodies[i].mass) * TIME_STEP;
        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
    }
}

// Function to initialize bodies randomly
void initializeBodies(vector<Body>& bodies) {
    srand(time(0));
    #pragma omp parallel for
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
    // Set number of threads
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);

    vector<Body> bodies(NUM_BODIES);
    vector<PaddedForce> forces(NUM_BODIES);

    auto start_time = chrono::high_resolution_clock::now();
    double total_kinetic_energy = 0.0;
    double total_distance = 0.0;

    initializeBodies(bodies);

    // Main simulation loop
    for (int step = 0; step < STEPS; step++) {
        computeForces(bodies, forces);
        updateBodies(bodies, forces);
    }

    // Calculate final statistics
    #pragma omp parallel for reduction(+:total_kinetic_energy,total_distance)
    for (int i = 0; i < bodies.size(); i++) {
        double velocity = sqrt(bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
        total_kinetic_energy += 0.5 * bodies[i].mass * velocity * velocity;
        total_distance += sqrt(bodies[i].x * bodies[i].x + bodies[i].y * bodies[i].y);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Print statistics in a consistent format
    cout << "=== Simulation Statistics ===\n";
    cout << "Implementation: OpenMP\n";
    cout << "Bodies: " << NUM_BODIES << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "Threads: " << num_threads << "\n";
    cout << "Execution Time: " << duration.count() << "\n";
    cout << "Kinetic Energy: " << scientific << setprecision(3) << total_kinetic_energy << "\n";
    cout << "Avg Distance: " << fixed << setprecision(2) << total_distance/NUM_BODIES << "\n";
    cout << "=== End Statistics ===\n";

    return 0;
}
