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
// tour[i] = index of the i-th city visited; returns to tour[0] at the end.
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
// Print tour summary: city order, total length
// =============================================================================
void print_tour(const string& label, const vector<int>& tour, double length) {
    cout << "  " << label << ":\n";
    cout << "    Tour length: " << fixed << setprecision(2) << length << "\n";
    cout << "    Order: ";
    int show = min((int)tour.size(), 20);
    for (int i = 0; i < show; i++) {
        if (i > 0) cout << " -> ";
        cout << tour[i];
    }
    if ((int)tour.size() > 20) cout << " -> ...";
    cout << " -> " << tour[0] << "\n";
}

// =============================================================================
// TODO(human): Nearest Neighbor heuristic for TSP
//
// Greedy construction: start at 'start' city, always go to nearest unvisited.
//
// Algorithm:
//   1. Mark all cities as unvisited. Mark 'start' as visited. Set current = start.
//   2. Repeat (n-1) times:
//      a. Among all unvisited cities, find the one nearest to 'current'
//      b. Add that city to the tour, mark it as visited, set current = that city
//   3. Return the tour (the return-to-start edge is implicit)
//
// Complexity: O(n^2) — for each of n cities, scan all n cities to find nearest.
//
// The quality is typically 20-25% above optimal for random Euclidean instances.
// The weakness is the "last edges" problem: near the end, the remaining unvisited
// cities are scattered far apart, forcing long closing edges.
//
// Implementation hint:
//   - Use a vector<bool> visited(n, false)
//   - For the inner loop, track best_dist and best_city
//   - The tour vector should have exactly n entries
// =============================================================================
vector<int> nearest_neighbor(const vector<Point>& points, int start) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// TODO(human): Cheapest Insertion heuristic for TSP
//
// Build the tour incrementally, always inserting the city that causes the
// least increase in total tour length.
//
// Algorithm:
//   1. INITIAL TRIANGLE: Find the 3 mutually farthest cities to form the seed tour.
//      Simple approach: pick city 0, find the farthest city from it (city a),
//      find the farthest city from both 0 and a (city b — maximize min distance
//      to the existing tour). Initialize tour = {0, a, b}.
//      Mark these 3 as "in tour."
//
//   2. INSERTION LOOP: Repeat until all cities are in the tour:
//      a. For each city k NOT in the tour:
//         Find its cheapest insertion cost over all consecutive pairs (i, j) in tour:
//           insertion_cost(k) = min over edges (tour[p], tour[p+1]) of:
//             d(tour[p], k) + d(k, tour[(p+1) % tour_size]) - d(tour[p], tour[(p+1) % tour_size])
//      b. Pick the city k* with the SMALLEST cheapest insertion cost
//      c. Insert k* at its best position in the tour
//         (use tour.insert(tour.begin() + best_pos + 1, k*) or equivalent)
//
//   The insertion cost formula d(i,k) + d(k,j) - d(i,j) measures how much
//   longer the tour becomes when city k is placed between i and j. The
//   triangle inequality guarantees this is always >= 0 for metric TSP.
//
// Complexity: O(n^2) with efficient bookkeeping (or O(n^3) naively).
//
// Quality: typically 10-15% above optimal — better than NN because it builds
// the tour's global shape incrementally rather than making purely local choices.
//
// Implementation hint:
//   - Use a vector<bool> in_tour(n, false) to track which cities are inserted
//   - The "tour" is a vector<int> that grows from 3 to n entries
//   - For each candidate city, iterate over all tour edges to find best position
//   - Keep track of (best_city, best_position, best_cost) across all candidates
// =============================================================================
vector<int> cheapest_insertion(const vector<Point>& points) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Main — compare construction heuristics on sample instances
// =============================================================================
int main() {
    cout << "=== Phase 1: TSP Construction Heuristics ===\n\n";

    // --- 20-city instance ---
    {
        cout << "Instance 1: 20 cities (seed=42)\n";
        auto pts = generate_random_points(20, 42);

        // Nearest Neighbor from city 0
        auto nn_tour = nearest_neighbor(pts, 0);
        double nn_len = tour_length(pts, nn_tour);
        print_tour("Nearest Neighbor (start=0)", nn_tour, nn_len);

        // Nearest Neighbor from best starting city
        double best_nn_len = 1e18;
        vector<int> best_nn_tour;
        for (int s = 0; s < (int)pts.size(); s++) {
            auto t = nearest_neighbor(pts, s);
            double l = tour_length(pts, t);
            if (l < best_nn_len) {
                best_nn_len = l;
                best_nn_tour = t;
            }
        }
        print_tour("Nearest Neighbor (best start)", best_nn_tour, best_nn_len);

        // Cheapest Insertion
        auto ci_tour = cheapest_insertion(pts);
        double ci_len = tour_length(pts, ci_tour);
        print_tour("Cheapest Insertion", ci_tour, ci_len);

        cout << "  Improvement of CI over NN(best): "
             << fixed << setprecision(1)
             << (1.0 - ci_len / best_nn_len) * 100.0 << "%\n\n";
    }

    // --- 50-city instance ---
    {
        cout << "Instance 2: 50 cities (seed=123)\n";
        auto pts = generate_random_points(50, 123);

        auto nn_tour = nearest_neighbor(pts, 0);
        double nn_len = tour_length(pts, nn_tour);
        print_tour("Nearest Neighbor (start=0)", nn_tour, nn_len);

        // Best starting city for NN
        double best_nn_len = 1e18;
        vector<int> best_nn_tour;
        for (int s = 0; s < (int)pts.size(); s++) {
            auto t = nearest_neighbor(pts, s);
            double l = tour_length(pts, t);
            if (l < best_nn_len) {
                best_nn_len = l;
                best_nn_tour = t;
            }
        }
        print_tour("Nearest Neighbor (best start)", best_nn_tour, best_nn_len);

        auto ci_tour = cheapest_insertion(pts);
        double ci_len = tour_length(pts, ci_tour);
        print_tour("Cheapest Insertion", ci_tour, ci_len);

        cout << "  Improvement of CI over NN(best): "
             << fixed << setprecision(1)
             << (1.0 - ci_len / best_nn_len) * 100.0 << "%\n\n";
    }

    // --- 100-city instance ---
    {
        cout << "Instance 3: 100 cities (seed=999)\n";
        auto pts = generate_random_points(100, 999);

        auto nn_tour = nearest_neighbor(pts, 0);
        double nn_len = tour_length(pts, nn_tour);
        print_tour("Nearest Neighbor (start=0)", nn_tour, nn_len);

        auto ci_tour = cheapest_insertion(pts);
        double ci_len = tour_length(pts, ci_tour);
        print_tour("Cheapest Insertion", ci_tour, ci_len);

        cout << "  Improvement of CI over NN: "
             << fixed << setprecision(1)
             << (1.0 - ci_len / nn_len) * 100.0 << "%\n\n";
    }

    cout << "Phase 1 complete.\n";
    return 0;
}
