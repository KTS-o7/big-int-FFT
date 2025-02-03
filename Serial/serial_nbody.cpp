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
const int NUM_BODIES = 1000;    // Number of bodies
const int STEPS = 10000;        // Simulation steps

// Structure to represent a body (planet, particle)
struct Body {
    double x, y, vx, vy, mass;
};

// Function to compute gravitational force
void computeForces(vector<Body>& bodies, vector<pair<double, double>>& forces) {
    int n = bodies.size();
    
    // Reset forces
    for (int i = 0; i < n; i++) {
        forces[i] = {0.0, 0.0};
    }

    // Brute-force O(n^2) force calculation
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double dx = bodies[j].x - bodies[i].x;
            double dy = bodies[j].y - bodies[i].y;
            double dist = sqrt(dx * dx + dy * dy + 1e-9); // Avoid division by zero
            double force = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);

            // Compute force components
            double fx = force * dx / dist;
            double fy = force * dy / dist;

            // Update forces
            forces[i].first += fx;
            forces[i].second += fy;
            forces[j].first -= fx;
            forces[j].second -= fy;
        }
    }
}

// Function to update positions and velocities
void updateBodies(vector<Body>& bodies, vector<pair<double, double>>& forces) {
    for (int i = 0; i < bodies.size(); i++) {
        // Update velocity using force
        bodies[i].vx += (forces[i].first / bodies[i].mass) * TIME_STEP;
        bodies[i].vy += (forces[i].second / bodies[i].mass) * TIME_STEP;

        // Update position using velocity
        bodies[i].x += bodies[i].vx * TIME_STEP;
        bodies[i].y += bodies[i].vy * TIME_STEP;
    }
}

// Function to initialize bodies randomly
void initializeBodies(vector<Body>& bodies) {
    srand(time(0));
    for (int i = 0; i < bodies.size(); i++) {
        bodies[i] = {
            (double)(rand() % 1000), // Random x
            (double)(rand() % 1000), // Random y
            0.0, 0.0,                // Initial velocity (vx, vy)
            (double)(rand() % 100 + 1) // Random mass
        };
    }
}

// Main simulation loop
int main() {
    vector<Body> bodies(NUM_BODIES);
    vector<pair<double, double>> forces(NUM_BODIES);

    // Add timing variables
    auto start_time = chrono::high_resolution_clock::now();
    double total_kinetic_energy = 0.0;
    double total_distance = 0.0;

    initializeBodies(bodies);

    // Simulation loop
    for (int step = 0; step < STEPS; step++) {
        computeForces(bodies, forces);
        updateBodies(bodies, forces);

        // Calculate statistics in final step
        if (step == STEPS - 1) {
            for (const auto& body : bodies) {
                // Calculate kinetic energy: 1/2 * m * v^2
                double velocity = sqrt(body.vx * body.vx + body.vy * body.vy);
                total_kinetic_energy += 0.5 * body.mass * velocity * velocity;
                
                // Calculate distance from origin
                total_distance += sqrt(body.x * body.x + body.y * body.y);
            }
        }

        //if (step % 100 == 0) {
          //  cout << "Step " << step << "/" << STEPS << " completed.\n";
        //}
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Print statistics in a consistent format
    cout << "=== Simulation Statistics ===\n";
    cout << "Implementation: Serial\n";
    cout << "Bodies: " << NUM_BODIES << "\n";
    cout << "Steps: " << STEPS << "\n";
    cout << "Execution Time: " << duration.count() << "\n";
    cout << "Kinetic Energy: " << scientific << setprecision(3) << total_kinetic_energy << "\n";
    cout << "Avg Distance: " << fixed << setprecision(2) << total_distance/NUM_BODIES << "\n";
    cout << "=== End Statistics ===\n";

    return 0;
}
