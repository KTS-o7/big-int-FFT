#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <chrono>
#include <pthread.h>

using namespace std;

// Constants
const double G = 6.67430e-11;  // Gravitational constant
const double TIME_STEP = 0.01; // Time step for simulation
const int NUM_BODIES = 1000;   // Number of bodies
const int STEPS = 10000;       // Simulation steps
const int NUM_THREADS = 8;     // Number of threads to use

// Add padding to avoid false sharing
struct alignas(64) PaddedForce {
    double fx;
    double fy;
    char padding[48];  // Pad to 64 bytes
};

// Structure to represent a body (planet, particle)
struct Body {
    double x, y, vx, vy, mass;
};

// Structure to pass arguments to thread function
struct ThreadArgs {
    vector<Body>* bodies;
    vector<PaddedForce>* forces;
    int thread_id;
};

// Global mutex for thread synchronization
pthread_barrier_t barrier;

// Function to compute gravitational force for a subset of bodies
void* computeForcesThread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    vector<Body>& bodies = *(args->bodies);
    vector<PaddedForce>& forces = *(args->forces);
    int thread_id = args->thread_id;
    int n = bodies.size();

    for (int step = 0; step < STEPS; step++) {
        // Reset forces
        for (int i = 0; i < n; i++) {
            forces[i].fx = 0.0;
            forces[i].fy = 0.0;
        }

        // Improved work distribution using cyclic distribution
        for (int i = thread_id; i < n; i += NUM_THREADS) {
            for (int j = i + 1; j < n; j++) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dist = sqrt(dx * dx + dy * dy + 1e-9);
                double force = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);

                double fx = force * dx / dist;
                double fy = force * dy / dist;

                // Use atomic operations for force updates
                #pragma omp atomic
                forces[i].fx += fx;
                #pragma omp atomic
                forces[i].fy += fy;
                #pragma omp atomic
                forces[j].fx -= fx;
                #pragma omp atomic
                forces[j].fy -= fy;
            }
        }

        pthread_barrier_wait(&barrier);

        // Divide position updates among threads
        for (int i = thread_id; i < n; i += NUM_THREADS) {
            bodies[i].vx += (forces[i].fx / bodies[i].mass) * TIME_STEP;
            bodies[i].vy += (forces[i].fy / bodies[i].mass) * TIME_STEP;
            bodies[i].x += bodies[i].vx * TIME_STEP;
            bodies[i].y += bodies[i].vy * TIME_STEP;
        }

        pthread_barrier_wait(&barrier);
    }

    return nullptr;
}

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
    vector<PaddedForce> forces(NUM_BODIES);
    vector<pthread_t> threads(NUM_THREADS);
    vector<ThreadArgs> thread_args(NUM_THREADS);

    // Initialize barrier
    pthread_barrier_init(&barrier, nullptr, NUM_THREADS);

    auto start_time = chrono::high_resolution_clock::now();
    double total_kinetic_energy = 0.0;
    double total_distance = 0.0;

    initializeBodies(bodies);

    // Create threads with simpler arguments
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].bodies = &bodies;
        thread_args[i].forces = &forces;
        thread_args[i].thread_id = i;
        pthread_create(&threads[i], nullptr, computeForcesThread, &thread_args[i]);
    }

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    // Calculate final statistics
    for (const auto& body : bodies) {
        double velocity = sqrt(body.vx * body.vx + body.vy * body.vy);
        total_kinetic_energy += 0.5 * body.mass * velocity * velocity;
        total_distance += sqrt(body.x * body.x + body.y * body.y);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Clean up barrier
    pthread_barrier_destroy(&barrier);

    // Print statistics in a consistent format
    cout << "=== Simulation Statistics ===\n";
    cout << "Implementation: Pthreads\n";
    cout << "Bodies: " << NUM_BODIES << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "Threads: " << NUM_THREADS << "\n";
    cout << "Execution Time: " << duration.count() << "\n";
    cout << "Kinetic Energy: " << scientific << setprecision(3) << total_kinetic_energy << "\n";
    cout << "Avg Distance: " << fixed << setprecision(2) << total_distance/NUM_BODIES << "\n";
    cout << "=== End Statistics ===\n";

    return 0;
}
