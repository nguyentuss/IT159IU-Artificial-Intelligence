"""
TSP Optimal Solver Utility.

Computes optimal or near-optimal tour length for TSP instances.
- Held-Karp (exact) for small instances (≤20 cities)
- 2-opt heuristic for larger instances
"""

import numpy as np
from itertools import combinations


def load_tsp_coordinates(file_path):
    """Load TSP coordinates from CSV file."""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                x, y = map(float, line.strip().split(','))
                coords.append([x, y])
    return np.array(coords)


def compute_distance_matrix(coords):
    """Compute Euclidean distance matrix."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = coords[i] - coords[j]
            dist[i, j] = np.sqrt(np.sum(diff ** 2))
    return dist


def held_karp(dist):
    """
    Held-Karp algorithm for exact TSP solution.
    Time complexity: O(n² * 2^n)
    Only feasible for n ≤ 20.
    
    Returns: (optimal_tour_length, tour)
    """
    n = len(dist)
    
    # dp[mask][i] = minimum distance to reach city i, having visited cities in mask
    # mask is a bitmask where bit j is set if city j has been visited
    INF = float('inf')
    
    # Initialize DP table
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Start from city 0
    dp[1][0] = 0  # mask=1 means only city 0 visited, at city 0
    
    # Iterate through all subsets
    for mask in range(1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask][last] == INF:
                continue
                
            # Try to extend to each unvisited city
            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                    
                new_mask = mask | (1 << next_city)
                new_dist = dp[mask][last] + dist[last][next_city]
                
                if new_dist < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_dist
                    parent[new_mask][next_city] = last
    
    # Find minimum tour: all cities visited, return to start
    full_mask = (1 << n) - 1
    min_tour = INF
    last_city = -1
    
    for i in range(1, n):
        tour_length = dp[full_mask][i] + dist[i][0]
        if tour_length < min_tour:
            min_tour = tour_length
            last_city = i
    
    # Reconstruct tour
    tour = []
    mask = full_mask
    current = last_city
    
    while current != -1:
        tour.append(current)
        prev = parent[mask][current]
        mask ^= (1 << current)
        current = prev
    
    tour = tour[::-1]  # Reverse to get correct order
    tour.append(0)  # Return to start
    
    return min_tour, tour


def nearest_neighbor(dist, start=0):
    """Nearest neighbor heuristic for TSP."""
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    total = 0
    
    current = start
    for _ in range(n - 1):
        best_next = -1
        best_dist = float('inf')
        
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        
        visited[best_next] = True
        tour.append(best_next)
        total += best_dist
        current = best_next
    
    # Return to start
    total += dist[current][start]
    tour.append(start)
    
    return total, tour


def two_opt(dist, tour):
    """
    2-opt improvement heuristic.
    Repeatedly reverses segments to reduce tour length.
    """
    n = len(tour) - 1  # Exclude the return to start
    improved = True
    
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Calculate change in tour length if we reverse segment [i, j]
                delta = (
                    dist[tour[i-1]][tour[j]] + dist[tour[i]][tour[j+1]]
                    - dist[tour[i-1]][tour[i]] - dist[tour[j]][tour[j+1]]
                )
                
                if delta < -1e-10:  # Improvement found
                    # Reverse segment
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    
    # Recalculate tour length
    total = sum(dist[tour[i]][tour[i+1]] for i in range(len(tour) - 1))
    
    return total, tour


def solve_tsp_optimal(file_path, max_exact=20):
    """
    Solve TSP instance optimally or near-optimally.
    
    Args:
        file_path: Path to CSV file with coordinates
        max_exact: Maximum cities for exact Held-Karp algorithm
        
    Returns: (tour_length, method_used)
    """
    coords = load_tsp_coordinates(file_path)
    n = len(coords)
    dist = compute_distance_matrix(coords)
    
    if n <= max_exact:
        # Use exact Held-Karp algorithm
        tour_length, tour = held_karp(dist)
        return tour_length, 'held_karp'
    else:
        # Use nearest neighbor + 2-opt heuristic
        # Try multiple starting points
        best_length = float('inf')
        best_tour = None
        
        for start in range(min(n, 10)):  # Try first 10 starting points
            length, tour = nearest_neighbor(dist, start)
            length, tour = two_opt(dist, tour)
            
            if length < best_length:
                best_length = length
                best_tour = tour
        
        return best_length, '2opt_heuristic'


def main():
    """Compute optimal tours for all TSP datasets."""
    import os
    
    data_dir = 'data/tsp'
    files = ['tiny.csv', 'small-1.csv', 'small-2.csv', 'small.csv']
    
    print("TSP Optimal Tour Lengths")
    print("=" * 50)
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            coords = load_tsp_coordinates(file_path)
            n = len(coords)
            tour_length, method = solve_tsp_optimal(file_path)
            print(f"{filename}: {n} cities -> {tour_length:.4f} ({method})")
        else:
            print(f"{filename}: File not found")


if __name__ == '__main__':
    main()
