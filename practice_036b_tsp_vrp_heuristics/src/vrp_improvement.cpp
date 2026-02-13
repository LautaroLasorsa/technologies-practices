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
// Euclidean distance
// =============================================================================
double distance(const Customer& a, const Customer& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// =============================================================================
// Full distance matrix
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
// VRP Solution
// =============================================================================
struct VRPSolution {
    vector<vector<int>> routes;

    double total_distance(const vector<vector<double>>& dist) const {
        double total = 0.0;
        for (const auto& route : routes) {
            if (route.empty()) continue;
            total += dist[0][route.front()];
            for (int i = 0; i + 1 < (int)route.size(); i++) {
                total += dist[route[i]][route[i + 1]];
            }
            total += dist[route.back()][0];
        }
        return total;
    }

    int route_demand(int r, const vector<Customer>& customers) const {
        int total = 0;
        for (int c : routes[r]) total += customers[c].demand;
        return total;
    }

    // Remove empty routes
    void compact() {
        routes.erase(
            remove_if(routes.begin(), routes.end(),
                       [](const vector<int>& r) { return r.empty(); }),
            routes.end());
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
// Generate random VRP instance
// =============================================================================
pair<vector<Customer>, int> generate_vrp_instance(int num_customers,
                                                   int max_demand,
                                                   int capacity,
                                                   unsigned seed) {
    mt19937 rng(seed);
    uniform_real_distribution<double> pos_dist(0.0, 1000.0);
    uniform_int_distribution<int> dem_dist(1, max_demand);

    vector<Customer> customers;
    customers.push_back({500.0, 500.0, 0});  // depot

    for (int i = 0; i < num_customers; i++) {
        customers.push_back({pos_dist(rng), pos_dist(rng), dem_dist(rng)});
    }

    return {customers, capacity};
}

// =============================================================================
// Clarke-Wright Savings (provided — used as starting solution for improvement)
// =============================================================================
VRPSolution clarke_wright_savings(const vector<Customer>& customers,
                                   int capacity,
                                   const vector<vector<double>>& dist) {
    int n = (int)customers.size();  // includes depot at index 0

    // Start with one route per customer
    VRPSolution sol;
    sol.routes.resize(n - 1);
    vector<int> route_of(n, -1);  // route_of[customer] = route index
    for (int i = 1; i < n; i++) {
        sol.routes[i - 1] = {i};
        route_of[i] = i - 1;
    }

    // Compute savings
    struct Saving {
        int i, j;
        double value;
    };
    vector<Saving> savings;
    for (int i = 1; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double s = dist[0][i] + dist[0][j] - dist[i][j];
            if (s > 0) savings.push_back({i, j, s});
        }
    }
    sort(savings.begin(), savings.end(),
         [](const Saving& a, const Saving& b) { return a.value > b.value; });

    // Merge routes greedily
    for (const auto& [ci, cj, sval] : savings) {
        int ri = route_of[ci];
        int rj = route_of[cj];
        if (ri == rj) continue;  // same route
        if (sol.routes[ri].empty() || sol.routes[rj].empty()) continue;

        // Check endpoints
        bool i_front = (sol.routes[ri].front() == ci);
        bool i_back = (sol.routes[ri].back() == ci);
        bool j_front = (sol.routes[rj].front() == cj);
        bool j_back = (sol.routes[rj].back() == cj);

        if (!i_front && !i_back) continue;
        if (!j_front && !j_back) continue;

        // Check capacity
        int demand_ri = sol.route_demand(ri, customers);
        int demand_rj = sol.route_demand(rj, customers);
        if (demand_ri + demand_rj > capacity) continue;

        // Merge: orient routes so ci is at back of ri and cj is at front of rj
        if (i_front) reverse(sol.routes[ri].begin(), sol.routes[ri].end());
        if (j_back) reverse(sol.routes[rj].begin(), sol.routes[rj].end());

        // Append rj to ri
        for (int c : sol.routes[rj]) {
            sol.routes[ri].push_back(c);
            route_of[c] = ri;
        }
        sol.routes[rj].clear();
    }

    sol.compact();
    return sol;
}

// =============================================================================
// Compute distance of a single route (depot -> customers -> depot)
// =============================================================================
double route_distance(const vector<int>& route, const vector<vector<double>>& dist) {
    if (route.empty()) return 0.0;
    double total = dist[0][route.front()];
    for (int i = 0; i + 1 < (int)route.size(); i++) {
        total += dist[route[i]][route[i + 1]];
    }
    total += dist[route.back()][0];
    return total;
}

// =============================================================================
// TODO(human): Relocate pass — move a customer from one route to another
//
// Try moving each customer from its current route to the best position in
// every other route. Accept the first improving move found.
//
// Algorithm:
//   For each route r1 (index r1_idx) in the solution:
//     For each position p in route r1 (customer c = r1[p]):
//       Compute removal_saving: the distance saved by removing c from r1.
//         If p == 0 (first customer):
//           removal_saving = dist[0][c] + dist[c][r1[1]] - dist[0][r1[1]]
//         If p == last:
//           removal_saving = dist[r1[p-1]][c] + dist[c][0] - dist[r1[p-1]][0]
//         Otherwise:
//           removal_saving = dist[r1[p-1]][c] + dist[c][r1[p+1]] - dist[r1[p-1]][r1[p+1]]
//         (Handle the edge case where r1 has only 1 customer separately:
//          removal_saving = dist[0][c] + dist[c][0])
//
//       For each other route r2 (index r2_idx != r1_idx):
//         If r2's demand + c's demand > capacity: skip (infeasible)
//
//         Find the best insertion position q in r2:
//           For q = 0 (before first customer of r2):
//             insertion_cost = dist[0][c] + dist[c][r2[0]] - dist[0][r2[0]]
//           For q = r2.size() (after last customer):
//             insertion_cost = dist[r2.back()][c] + dist[c][0] - dist[r2.back()][0]
//           For q between consecutive customers r2[q-1] and r2[q]:
//             insertion_cost = dist[r2[q-1]][c] + dist[c][r2[q]] - dist[r2[q-1]][r2[q]]
//           (Also handle r2 being empty: insertion_cost = dist[0][c] + dist[c][0])
//
//         delta = insertion_cost - removal_saving
//         (negative delta = improvement)
//
//         If delta < -1e-10:
//           Perform the move:
//             1. Remove c from r1 at position p
//             2. Insert c into r2 at position q
//             3. Remove empty routes if r1 became empty
//           Return true
//
//   Return false (no improving relocate found — local optimum for relocate)
// =============================================================================
bool relocate_pass(VRPSolution& sol,
                   const vector<Customer>& customers,
                   int capacity,
                   const vector<vector<double>>& dist) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// TODO(human): 2-opt* pass — inter-route 2-opt (swap tails between routes)
//
// For each pair of routes (r1, r2), try swapping their tails at every pair
// of cut points. Accept the first improving move found.
//
// Algorithm:
//   For each pair of routes (r1_idx, r2_idx) with r1_idx < r2_idx:
//     Let r1 = sol.routes[r1_idx], r2 = sol.routes[r2_idx]
//
//     For each cut point i in r1 (i = 0, 1, ..., r1.size()-1):
//       For each cut point j in r2 (j = 0, 1, ..., r2.size()-1):
//         Proposed new routes:
//           new_r1 = r1[0..i] + r2[j+1..end]  (first i+1 of r1, then tail of r2)
//           new_r2 = r2[0..j] + r1[i+1..end]  (first j+1 of r2, then tail of r1)
//
//         Check capacity feasibility:
//           demand(new_r1) = sum of demands of customers in new_r1
//           demand(new_r2) = sum of demands of customers in new_r2
//           Both must be <= capacity
//
//         Compute delta = route_distance(new_r1) + route_distance(new_r2)
//                       - route_distance(r1) - route_distance(r2)
//
//         Optimization: instead of recomputing full routes, delta can be
//         computed from just the 4 changed edges:
//           old edges: (r1[i], r1[i+1]) and (r2[j], r2[j+1])
//             where r1[i+1] = r1[i+1] if i+1 < r1.size(), else depot (0)
//             and   r2[j+1] = r2[j+1] if j+1 < r2.size(), else depot (0)
//           new edges: (r1[i], r2[j+1]) and (r2[j], r1[i+1])
//           delta = dist[r1[i]][r2_next] + dist[r2[j]][r1_next]
//                 - dist[r1[i]][r1_next] - dist[r2[j]][r2_next]
//           (where r1_next, r2_next account for depot at route ends)
//
//         If delta < -1e-10 AND both new routes are capacity-feasible:
//           Perform the swap:
//             tail1 = r1[i+1..end] (save before modifying)
//             tail2 = r2[j+1..end]
//             r1 = r1[0..i] + tail2
//             r2 = r2[0..j] + tail1
//           Remove empty routes if any
//           Return true
//
//   Return false (no improving 2-opt* move found)
//
// Note: 2-opt* differs from intra-route 2-opt — it does NOT reverse segments.
// It swaps tails between two routes, which changes the assignment of customers
// to vehicles. This is why capacity must be re-checked.
// =============================================================================
bool two_opt_star_pass(VRPSolution& sol,
                       const vector<vector<double>>& dist,
                       const vector<Customer>& customers,
                       int capacity) {
    throw runtime_error("TODO(human): not implemented");
}

// =============================================================================
// VRP local search: alternate relocate and 2-opt* until no improvement
// =============================================================================
VRPSolution vrp_local_search(VRPSolution sol,
                              const vector<Customer>& customers,
                              int capacity,
                              const vector<vector<double>>& dist) {
    int iteration = 0;
    bool improved = true;
    while (improved) {
        improved = false;

        // Relocate passes
        while (relocate_pass(sol, customers, capacity, dist)) {
            improved = true;
            sol.compact();
        }

        // 2-opt* passes
        while (two_opt_star_pass(sol, dist, customers, capacity)) {
            improved = true;
            sol.compact();
        }

        iteration++;
        if (iteration > 100) break;  // safety limit
    }
    cout << "    Local search iterations: " << iteration << "\n";
    return sol;
}

// =============================================================================
// Main — Clarke-Wright + local search improvement
// =============================================================================
int main() {
    cout << "=== Phase 4: VRP Improvement (Relocate + 2-opt*) ===\n\n";

    auto run_test = [](const string& name, int num_customers, int max_demand,
                       int capacity, unsigned seed) {
        cout << name << ": " << num_customers << " customers, capacity="
             << capacity << ", seed=" << seed << "\n";

        auto [customers, cap] = generate_vrp_instance(num_customers, max_demand,
                                                       capacity, seed);
        auto dist = distance_matrix(customers);

        // Naive: one route per customer
        double naive_dist = 0.0;
        for (int i = 1; i < (int)customers.size(); i++) {
            naive_dist += 2.0 * dist[0][i];
        }
        cout << "  Naive distance: " << fixed << setprecision(2) << naive_dist << "\n";

        // Clarke-Wright construction
        auto cw_sol = clarke_wright_savings(customers, cap, dist);
        double cw_dist = cw_sol.total_distance(dist);
        print_vrp_solution("Clarke-Wright (construction)", cw_sol, customers, dist);

        // Local search improvement
        auto start_time = chrono::high_resolution_clock::now();
        auto improved_sol = vrp_local_search(cw_sol, customers, cap, dist);
        auto end_time = chrono::high_resolution_clock::now();
        double elapsed_ms = chrono::duration<double, milli>(end_time - start_time).count();

        double improved_dist = improved_sol.total_distance(dist);
        print_vrp_solution("After local search", improved_sol, customers, dist);

        cout << "    Improvement over CW: " << fixed << setprecision(1)
             << (1.0 - improved_dist / cw_dist) * 100.0 << "%\n";
        cout << "    Improvement over naive: " << fixed << setprecision(1)
             << (1.0 - improved_dist / naive_dist) * 100.0 << "%\n";
        cout << "    Time: " << fixed << setprecision(1) << elapsed_ms << " ms\n\n";
    };

    run_test("Test 1", 15, 15, 50, 42);
    run_test("Test 2", 25, 20, 60, 123);
    run_test("Test 3", 30, 20, 80, 999);
    run_test("Test 4", 40, 15, 70, 7777);

    cout << "Phase 4 complete.\n";
    return 0;
}
