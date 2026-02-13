#include <bits/stdc++.h>
using namespace std;

// =============================================================================
// Point in 2D Euclidean space
// =============================================================================
struct Point {
    double x, y;
};

// =============================================================================
// Euclidean distance between two points
// =============================================================================
double distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// =============================================================================
// Compute total tour length (Hamiltonian cycle)
// =============================================================================
double tour_length(const vector<Point>& points, const vector<int>& tour) {
    double total = 0.0;
    int n = (int)tour.size();
    for (int i = 0; i < n; i++) {
        total += distance(points[tour[i]], points[tour[(i + 1) % n]]);
    }
    return total;
}

// =============================================================================
// Generate n random 2D points in [0, 1000) x [0, 1000)
// =============================================================================
vector<Point> generate_random_points(int n, unsigned seed) {
    mt19937 rng(seed);
    uniform_real_distribution<double> dist(0.0, 1000.0);
    vector<Point> points(n);
    for (auto& p : points) {
        p.x = dist(rng);
        p.y = dist(rng);
    }
    return points;
}

// =============================================================================
// Nearest Neighbor heuristic (provided — used as starting tour for 2-opt)
// =============================================================================
vector<int> nearest_neighbor(const vector<Point>& points, int start) {
    int n = (int)points.size();
    vector<bool> visited(n, false);
    vector<int> tour;
    tour.reserve(n);

    int current = start;
    visited[current] = true;
    tour.push_back(current);

    for (int step = 1; step < n; step++) {
        double best_dist = 1e18;
        int best_city = -1;
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                double d = distance(points[current], points[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best_city = j;
                }
            }
        }
        visited[best_city] = true;
        tour.push_back(best_city);
        current = best_city;
    }
    return tour;
}

// =============================================================================
// Print tour summary
// =============================================================================
void print_tour(const string& label, const vector<int>& tour, double length) {
    cout << "  " << label << ":\n";
    cout << "    Tour length: " << fixed << setprecision(2) << length << "\n";
}

// =============================================================================
// TODO(human): 2-opt single pass
//
// Try all pairs (i, j) with 0 <= i < j < n, i+1 < j (non-adjacent edges).
// For each pair, check if reversing the segment tour[i+1..j] improves the tour.
//
// The 2-opt move removes two edges and reconnects the tour:
//   Current edges: (tour[i], tour[i+1]) and (tour[j], tour[(j+1) % n])
//   New edges:     (tour[i], tour[j])   and (tour[i+1], tour[(j+1) % n])
//
// The segment tour[i+1..j] is reversed in place to maintain tour validity.
//
// Compute the change in tour length (delta):
//   delta = d(tour[i], tour[j]) + d(tour[i+1], tour[(j+1) % n])
//         - d(tour[i], tour[i+1]) - d(tour[j], tour[(j+1) % n])
//
// If delta < -1e-10 (improvement, with epsilon for floating-point):
//   Reverse tour[i+1..j]: std::reverse(tour.begin() + i + 1, tour.begin() + j + 1)
//   Return true (found an improving move)
//
// If no improving pair found after checking all pairs:
//   Return false (tour is 2-opt locally optimal)
//
// Note: use "first improvement" strategy — accept the first improving move
// found and return immediately. This is faster per pass than "best improvement"
// (checking all pairs and taking the best), and empirically converges in
// similar total time.
//
// Implementation hint:
//   - Outer loop: i from 0 to n-2
//   - Inner loop: j from i+2 to n-1 (but skip j = n-1 when i = 0,
//     because edges (0,1) and (n-1,0) are adjacent in the cyclic tour)
//   - The adjacency exception: when i=0 and j=n-1, the two edges share
//     vertex tour[0], so this is not a valid 2-opt move
// =============================================================================
bool two_opt_pass(const vector<Point>& points, vector<int>& tour) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// TODO(human): 2-opt improvement — iterate passes until convergence
//
// Repeatedly call two_opt_pass() until it returns false (no improvement found).
// This means the tour has reached a 2-opt local optimum — no single pair of
// edge swaps can improve it further.
//
// Algorithm:
//   1. Copy the input tour (or modify in place)
//   2. Set pass_count = 0
//   3. While two_opt_pass() returns true:
//      - Increment pass_count
//   4. Return the improved tour
//
// Optionally print the pass count and tour length after each pass to observe
// convergence behavior (most improvement happens in the first few passes).
//
// Typical behavior on random 100-city instances:
//   - NN tour: ~15000-20000
//   - After 2-opt: ~9000-11000 (30-45% improvement)
//   - Number of passes: 5-20 (diminishing returns)
//
// Complexity: O(n^2) per pass, O(passes * n^2) total. The number of passes
// is empirically O(n) in the worst case, so total is roughly O(n^3).
// =============================================================================
vector<int> two_opt(const vector<Point>& points, vector<int> tour) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Main — demonstrate 2-opt improvement on various instance sizes
// =============================================================================
int main() {
    cout << "=== Phase 2: TSP 2-opt Improvement ===\n\n";

    auto run_test = [](const string& name, int n, unsigned seed) {
        cout << name << ": " << n << " cities (seed=" << seed << ")\n";
        auto pts = generate_random_points(n, seed);

        // Construction: nearest neighbor from city 0
        auto nn_tour = nearest_neighbor(pts, 0);
        double nn_len = tour_length(pts, nn_tour);
        print_tour("Nearest Neighbor", nn_tour, nn_len);

        // Improvement: 2-opt
        auto start_time = chrono::high_resolution_clock::now();
        auto improved = two_opt(pts, nn_tour);
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();

        double improved_len = tour_length(pts, improved);
        print_tour("After 2-opt", improved, improved_len);

        double improvement = (1.0 - improved_len / nn_len) * 100.0;
        cout << "    Improvement: " << fixed << setprecision(1) << improvement << "%\n";
        cout << "    Time: " << fixed << setprecision(1) << elapsed_ms << " ms\n\n";
    };

    run_test("Test 1", 20, 42);
    run_test("Test 2", 50, 123);
    run_test("Test 3", 100, 999);
    run_test("Test 4", 200, 7777);
    run_test("Test 5", 500, 31415);

    cout << "Phase 2 complete.\n";
    return 0;
}
