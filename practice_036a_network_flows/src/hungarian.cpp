#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

// =============================================================================
// Print cost matrix
// =============================================================================
void print_matrix(const string& label, const vector<vector<int>>& cost) {
    int n = (int)cost.size();
    cout << "  " << label << " (" << n << "x" << n << "):\n";
    // Header
    cout << "         ";
    for (int j = 0; j < n; j++) {
        cout << "Job" << j << "  ";
    }
    cout << "\n";
    for (int i = 0; i < n; i++) {
        cout << "  W" << i << "  [";
        for (int j = 0; j < n; j++) {
            cout << setw(5) << cost[i][j];
            if (j + 1 < n) cout << ",";
        }
        cout << " ]\n";
    }
}

// =============================================================================
// Print assignment result
// =============================================================================
void print_assignment(const string& label, const vector<vector<int>>& cost,
                      const vector<int>& assignment) {
    int n = (int)assignment.size();
    int total = 0;
    cout << "  " << label << ":\n";
    for (int i = 0; i < n; i++) {
        int j = assignment[i];
        cout << "    Worker " << i << " -> Job " << j
             << " (cost = " << cost[i][j] << ")\n";
        total += cost[i][j];
    }
    cout << "  Total cost: " << total << "\n";
}

// =============================================================================
// TODO(human): Hungarian Algorithm (Kuhn-Munkres) — O(n^3)
//
// Input: n x n cost matrix C where C[i][j] = cost of assigning worker i to job j
// Output: assignment[i] = job assigned to worker i, minimizing total cost
//
// The algorithm maintains:
//   u[i]: potential for worker i (row)
//   v[j]: potential for job j (column)
//   p[j]: which worker is matched to job j (or 0 if unmatched)
//   way[j]: the job through which the shortest augmenting path reaches j
//
// For each worker i (1 to n):
//   1. Create a virtual "unmatched" column 0 and set p[0] = i
//   2. Find shortest augmenting path from i to an unmatched job:
//      - minv[j] = min reduced cost to reach job j
//      - used[j] = whether job j is in the current tree
//      - Repeat: pick unvisited j with minimum minv[j], mark visited
//        For matched worker p[j], update minv for all unvisited jobs
//   3. Update potentials u, v along the augmenting path
//   4. Augment: follow way[] to update matching p[]
//
// Complexity: O(n^3) -- n workers, each requires O(n^2) shortest path
//
// This is the "competitive programming" version of the Hungarian algorithm,
// adapted from the standard CP template. The potential variables u[], v[]
// correspond to dual variables in the LP dual of the assignment problem.
//
// Reference: https://cp-algorithms.com/graph/hungarian-algorithm.html
//
// Implementation notes:
//   - Uses 1-indexed workers and jobs internally (0 is the virtual column)
//   - u[] and v[] are sized (n+1), p[] and way[] are sized (n+1)
//   - The cost matrix is 0-indexed: cost[i][j] for i,j in [0,n)
//   - Return 0-indexed assignment: result[i] = j means worker i -> job j
// =============================================================================
vector<int> hungarian(const vector<vector<int>>& cost) {
    throw std::runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Sample cost matrices
// =============================================================================

// 5x5 assignment problem
// Classic example with varied costs
vector<vector<int>> make_5x5_cost() {
    return {
        { 82, 83, 69, 92, 65 },
        { 77, 37, 49, 92, 88 },
        { 11, 69, 5,  86, 68 },
        { 8,  9,  98, 23, 78 },
        { 56, 98, 35, 68, 0  },
    };
}

// 8x8 assignment problem
// Larger test case for correctness verification
vector<vector<int>> make_8x8_cost() {
    return {
        { 20, 25, 22, 28, 24, 30, 21, 27 },
        { 15, 18, 23, 17, 19, 22, 25, 20 },
        { 30, 27, 25, 20, 32, 28, 26, 24 },
        { 22, 20, 18, 25, 23, 19, 28, 21 },
        { 28, 32, 30, 22, 20, 25, 24, 26 },
        { 18, 22, 20, 30, 28, 15, 19, 23 },
        { 25, 19, 27, 24, 21, 23, 16, 29 },
        { 23, 28, 24, 19, 26, 21, 22, 17 },
    };
}

// Trivial 3x3 for debugging
vector<vector<int>> make_3x3_cost() {
    return {
        { 1, 2, 3 },
        { 4, 5, 6 },
        { 7, 8, 9 },
    };
    // Optimal: W0->J0(1), W1->J1(5), W2->J2(9) = 15?
    // Actually: W0->J2(3), W1->J1(5), W2->J0(7) = 15 also.
    // Or: W0->J0(1), W1->J2(6), W2->J1(8) = 15. Multiple optima, all cost=15.
}

// =============================================================================
// Main — solve assignment problems
// =============================================================================
int main() {
    cout << "=== Phase 3: Hungarian Algorithm (Optimal Assignment) ===\n\n";

    // Test 1: Trivial 3x3
    {
        cout << "Test 1: 3x3 assignment\n";
        auto cost = make_3x3_cost();
        print_matrix("Cost matrix", cost);
        auto assignment = hungarian(cost);
        print_assignment("Optimal assignment", cost, assignment);
        cout << "  (Expected total cost: 15)\n\n";
    }

    // Test 2: 5x5 assignment
    {
        cout << "Test 2: 5x5 assignment\n";
        auto cost = make_5x5_cost();
        print_matrix("Cost matrix", cost);
        auto assignment = hungarian(cost);
        print_assignment("Optimal assignment", cost, assignment);
        // Verify: compute brute force for small n
        cout << "\n";
    }

    // Test 3: 8x8 assignment
    {
        cout << "Test 3: 8x8 assignment\n";
        auto cost = make_8x8_cost();
        print_matrix("Cost matrix", cost);
        auto assignment = hungarian(cost);
        print_assignment("Optimal assignment", cost, assignment);
        cout << "\n";
    }

    // Test 4: Verify with symmetric costs (should assign diagonal)
    {
        cout << "Test 4: Identity-like cost (off-diagonal expensive)\n";
        int n = 4;
        vector<vector<int>> cost(n, vector<int>(n, 100));
        for (int i = 0; i < n; i++) cost[i][i] = 1;
        print_matrix("Cost matrix", cost);
        auto assignment = hungarian(cost);
        print_assignment("Optimal assignment", cost, assignment);
        cout << "  (Expected total cost: " << n << ")\n\n";
    }

    cout << "Phase 3 complete.\n";
    return 0;
}
