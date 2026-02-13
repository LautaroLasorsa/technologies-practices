#include <bits/stdc++.h>
using namespace std;

// =============================================================================
// Customer: location (x, y) and demand. Index 0 is the depot (demand = 0).
// =============================================================================
struct Customer {
    double x, y;
    int demand;
};

// =============================================================================
// Euclidean distance between two customers
// =============================================================================
double distance(const Customer& a, const Customer& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// =============================================================================
// Compute full distance matrix (symmetric)
// =============================================================================
vector<vector<double>> distance_matrix(const vector<Customer>& customers) {
    int n = (int)customers.size();
    vector<vector<double>> dist(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            dist[i][j] = dist[j][i] = distance(customers[i], customers[j]);
        }
    }
    return dist;
}

// =============================================================================
// VRP Solution: set of routes, each route is a sequence of customer indices
// (not including depot — depot is implicit at start and end of each route)
// =============================================================================
struct VRPSolution {
    vector<vector<int>> routes;  // routes[r] = {c1, c2, ...} (customer indices, 1-based)

    // Compute total distance (all routes, including depot-to-first and last-to-depot)
    double total_distance(const vector<vector<double>>& dist) const {
        double total = 0.0;
        for (const auto& route : routes) {
            if (route.empty()) continue;
            total += dist[0][route.front()];  // depot to first
            for (int i = 0; i + 1 < (int)route.size(); i++) {
                total += dist[route[i]][route[i + 1]];
            }
            total += dist[route.back()][0];   // last to depot
        }
        return total;
    }

    // Compute demand of a single route
    int route_demand(int r, const vector<Customer>& customers) const {
        int total = 0;
        for (int c : routes[r]) total += customers[c].demand;
        return total;
    }
};

// =============================================================================
// Print VRP solution
// =============================================================================
void print_vrp_solution(const string& label, const VRPSolution& sol,
                        const vector<Customer>& customers,
                        const vector<vector<double>>& dist) {
    cout << "  " << label << ":\n";
    cout << "    Total distance: " << fixed << setprecision(2)
         << sol.total_distance(dist) << "\n";
    cout << "    Vehicles used: " << sol.routes.size() << "\n";
    for (int r = 0; r < (int)sol.routes.size(); r++) {
        cout << "    Route " << r << " (demand="
             << sol.route_demand(r, customers) << "): depot";
        for (int c : sol.routes[r]) {
            cout << " -> " << c;
        }
        cout << " -> depot\n";
    }
}

// =============================================================================
// Generate a random VRP instance: depot at center, customers scattered around
// =============================================================================
pair<vector<Customer>, int> generate_vrp_instance(int num_customers,
                                                   int max_demand,
                                                   int capacity,
                                                   unsigned seed) {
    mt19937 rng(seed);
    uniform_real_distribution<double> pos_dist(0.0, 1000.0);
    uniform_int_distribution<int> dem_dist(1, max_demand);

    vector<Customer> customers;
    // Index 0: depot at center
    customers.push_back({500.0, 500.0, 0});

    for (int i = 0; i < num_customers; i++) {
        customers.push_back({pos_dist(rng), pos_dist(rng), dem_dist(rng)});
    }

    return {customers, capacity};
}

// =============================================================================
// TODO(human): Clarke-Wright Savings Algorithm for CVRP
//
// The most famous VRP construction heuristic (Clarke & Wright, 1964).
//
// Algorithm:
//   1. INITIAL SOLUTION: Start with n separate routes, each serving exactly
//      one customer: depot -> customer_i -> depot (for i = 1..n).
//      Total distance = 2 * sum of d(depot, i) for all customers.
//
//   2. COMPUTE SAVINGS: For each pair of customers (i, j) where i < j:
//      s(i,j) = d(0, i) + d(0, j) - d(i, j)
//      Interpretation: merging the routes serving i and j into one route
//      (depot -> ... -> i -> j -> ... -> depot) instead of two separate
//      round-trips saves s(i,j) distance.
//
//   3. SORT SAVINGS: Sort all (i, j, s(i,j)) triples by s(i,j) descending
//      (biggest savings first — greedy).
//
//   4. MERGE ROUTES: For each saving (i, j) in sorted order:
//      a. Check: i and j must be in DIFFERENT routes
//      b. Check: i must be at an ENDPOINT of its route (first or last customer)
//      c. Check: j must be at an ENDPOINT of its route (first or last customer)
//      d. Check: combined demand of both routes <= vehicle capacity
//      If all checks pass:
//        Merge the two routes by connecting the endpoint containing i
//        to the endpoint containing j.
//
//   The endpoint checks ensure we only concatenate routes end-to-end,
//   never splice into the middle of a route. This preserves route structure.
//
// Implementation hints:
//   - Store routes in a vector<vector<int>> (each route = list of customer indices)
//   - Track route_of[c] = which route index customer c belongs to
//   - To check endpoints: route.front() == c or route.back() == c
//   - When merging, you may need to reverse one or both routes so that i and j
//     are adjacent at the merge point. Four cases:
//       i = back of r1,  j = front of r2 -> append r2 to r1
//       i = back of r1,  j = back of r2  -> append reverse(r2) to r1
//       i = front of r1, j = front of r2 -> prepend reverse(r2) to r1 (or reverse r1, append r2)
//       i = front of r1, j = back of r2  -> prepend r2 to r1 (or append r1 to r2)
//   - After merging, update route_of[] for all moved customers
//   - Remove the empty route (or mark it empty and skip later)
//
// Parameters:
//   customers: vector of Customer (index 0 = depot)
//   capacity:  vehicle capacity Q
//   dist:      precomputed distance matrix
//
// Returns: VRPSolution with the constructed routes
// =============================================================================
VRPSolution clarke_wright_savings(const vector<Customer>& customers,
                                   int capacity,
                                   const vector<vector<double>>& dist) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Main — run Clarke-Wright on sample VRP instances
// =============================================================================
int main() {
    cout << "=== Phase 3: VRP Construction (Clarke-Wright Savings) ===\n\n";

    // --- Instance 1: Small (15 customers) ---
    {
        cout << "Instance 1: 15 customers, capacity=50, seed=42\n";
        auto [customers, capacity] = generate_vrp_instance(15, 15, 50, 42);

        cout << "  Depot: (500, 500)\n";
        cout << "  Customers:\n";
        for (int i = 1; i < (int)customers.size(); i++) {
            cout << "    " << i << ": (" << fixed << setprecision(0)
                 << customers[i].x << ", " << customers[i].y
                 << ") demand=" << customers[i].demand << "\n";
        }

        auto dist = distance_matrix(customers);

        // Naive solution: one route per customer
        double naive_dist = 0.0;
        for (int i = 1; i < (int)customers.size(); i++) {
            naive_dist += 2.0 * dist[0][i];
        }
        cout << "  Naive (one route per customer): " << fixed << setprecision(2)
             << naive_dist << "\n";

        auto sol = clarke_wright_savings(customers, capacity, dist);
        print_vrp_solution("Clarke-Wright", sol, customers, dist);

        double savings_pct = (1.0 - sol.total_distance(dist) / naive_dist) * 100.0;
        cout << "    Savings vs naive: " << fixed << setprecision(1)
             << savings_pct << "%\n\n";
    }

    // --- Instance 2: Medium (30 customers) ---
    {
        cout << "Instance 2: 30 customers, capacity=80, seed=123\n";
        auto [customers, capacity] = generate_vrp_instance(30, 20, 80, 123);
        auto dist = distance_matrix(customers);

        double naive_dist = 0.0;
        for (int i = 1; i < (int)customers.size(); i++) {
            naive_dist += 2.0 * dist[0][i];
        }
        cout << "  Naive distance: " << fixed << setprecision(2) << naive_dist << "\n";

        auto sol = clarke_wright_savings(customers, capacity, dist);
        print_vrp_solution("Clarke-Wright", sol, customers, dist);

        double savings_pct = (1.0 - sol.total_distance(dist) / naive_dist) * 100.0;
        cout << "    Savings vs naive: " << fixed << setprecision(1)
             << savings_pct << "%\n\n";
    }

    // --- Instance 3: Tight capacity (15 customers, small vehicles) ---
    {
        cout << "Instance 3: 15 customers, capacity=25 (tight), seed=777\n";
        auto [customers, capacity] = generate_vrp_instance(15, 12, 25, 777);
        auto dist = distance_matrix(customers);

        double naive_dist = 0.0;
        for (int i = 1; i < (int)customers.size(); i++) {
            naive_dist += 2.0 * dist[0][i];
        }
        cout << "  Naive distance: " << fixed << setprecision(2) << naive_dist << "\n";

        auto sol = clarke_wright_savings(customers, capacity, dist);
        print_vrp_solution("Clarke-Wright", sol, customers, dist);

        double savings_pct = (1.0 - sol.total_distance(dist) / naive_dist) * 100.0;
        cout << "    Savings vs naive: " << fixed << setprecision(1)
             << savings_pct << "%\n\n";
    }

    cout << "Phase 3 complete.\n";
    return 0;
}
