from math import sqrt
import numpy as np
from typing import Dict, List, Optional, Tuple, TextIO
from functools import lru_cache
import os
import time
import tsplib95

# Import local solvers
from basic_solvers.Lin_kerninghan import *
from basic_solvers.Bruteforce import solve_tsp_brute_force
from basic_solvers.initial_solution import setup_initial_solution
from basic_solvers.permutation_distance import compute_permutation_distance


# =============================================================================
# TSP FILE LOADING FUNCTIONS
# =============================================================================

def rotate_to_start(tour, start):
    idx = tour.index(start)
    return tour[idx:] + tour[:idx]

def _load_tsp_file(filepath: str) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Load a TSP file and build distance matrix.
    
    Parameters
    ----------
    filepath : str
        Path to the .tsp file
        
    Returns
    -------
    Tuple
        (distance_matrix, node_ids, metadata)
    """
    problem = tsplib95.load(filepath)
    
    # Get nodes
    nodes = list(problem.get_nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    
    # Build distance matrix based on problem type
    mat = np.zeros((n, n))
    
    # Try different methods to get distances
    if hasattr(problem, 'node_coords') and problem.node_coords:
        # Euclidean 2D coordinates
        coords = problem.node_coords
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i == j:
                    mat[i, j] = 0
                else:
                    x1, y1 = coords[u]
                    x2, y2 = coords[v]
                    mat[i, j] = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    elif hasattr(problem, 'get_weight'):
        # Use get_weight method
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                mat[i, j] = 0 if i == j else problem.get_weight(u, v)
    else:
        # Try edge weights
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                mat[i, j] = 0 if i == j else problem.get_edge_weight(u, v)
    
    # Extract metadata
    metadata = {
        'name': problem.name,
        'comment': problem.comment,
        'dimension': n,
        'edge_weight_type': getattr(problem, 'edge_weight_type', 'UNKNOWN')
    }
    
    return mat, nodes, metadata

def rotate_to_start(tour, start):
    idx = tour.index(start)
    return tour[idx:] + tour[:idx]


def _load_all_instances(instances_dir: str = "instances") -> List[Tuple[str, np.ndarray, List[int], Dict]]:
    """
    Load all TSP instances from a directory.
    
    Parameters
    ----------
    instances_dir : str
        Directory containing .tsp files
        
    Returns
    -------
    List of tuples
        Each tuple: (filename, distance_matrix, node_ids, metadata)
    """
    instances = []
    
    if not os.path.exists(instances_dir):
        print(f"[warning] instances directory '{instances_dir}' not found")
        return instances
    
    # Get all .tsp files
    tsp_files = [f for f in os.listdir(instances_dir) if f.endswith('.tsp')]
    tsp_files.sort()
    
    if VERBOSE:
        print(f"[loader] found {len(tsp_files)} TSP files in '{instances_dir}'")
    
    for filename in tsp_files:
        filepath = os.path.join(instances_dir, filename)
        try:
            dm, nodes, metadata = _load_tsp_file(filepath)
            instances.append((filename, dm, nodes, metadata))
            if VERBOSE:
                print(f"[loader] loaded {filename}: {metadata['dimension']} nodes")
        except Exception as e:
            print(f"[error] failed to load {filename}: {e}")
    
    return instances


# =============================================================================
# DYNAMIC PROGRAMMING TSP SOLVER
# =============================================================================

def solve_tsp_dynamic_programming(
    distance_matrix: np.ndarray,
    maxsize: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Solve TSP to optimality with dynamic programming (Held-Karp algorithm).
    
    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n)
    maxsize
        Maximum cache size for lru_cache
        
    Returns
    -------
    Tuple
        (optimal_cycle, total_distance)
    """
    n = distance_matrix.shape[0]
    
    # Handle small cases directly
    if n == 1:
        return [0], 0.0
    if n == 2:
        dist = distance_matrix[0, 1] + distance_matrix[1, 0]
        return [0, 1], dist
    
    # Get initial set {1, 2, ..., n-1} as frozenset
    N = frozenset(range(1, n))
    memo: Dict[Tuple, int] = {}
    
    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]
        
        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin
        
        return min_cost
    
    best_distance = dist(0, N)
    
    # Reconstruct path
    ni = 0
    solution = [0]
    
    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})
    
    return solution, best_distance


# =============================================================================
# LIN-KERNIGHAN HELPER FUNCTIONS
# =============================================================================

def _cycle_to_successors(cycle: List[int]) -> List[int]:
    """Convert cycle to successors representation."""
    successors = cycle[:]
    n = len(cycle)
    for i, _ in enumerate(cycle):
        successors[cycle[i]] = cycle[(i + 1) % n]
    return successors


def _successors_to_cycle(successors: List[int]) -> List[int]:
    """Convert successors to cycle representation."""
    cycle = successors[:]
    j = 0
    for i, _ in enumerate(successors):
        cycle[i] = j
        j = successors[j]
    return cycle


def _minimizes_hamiltonian_path_distance(
    tabu: np.ndarray,
    iteration: int,
    successors: List[int],
    ejected_edge: Tuple[int, int],
    distance_matrix: np.ndarray,
    hamiltonian_path_distance: float,
    hamiltonian_cycle_distance: float,
) -> Tuple[int, int, float]:
    """Find edge that minimizes Hamiltonian path distance after ejection."""
    a, b = ejected_edge
    best_c = c = last_c = successors[b]
    path_cb_distance = distance_matrix[c, b]
    path_bc_distance = distance_matrix[b, c]
    hamiltonian_path_distance_found = hamiltonian_cycle_distance
    
    while successors[c] != a:
        d = successors[c]
        path_cb_distance += distance_matrix[c, last_c]
        path_bc_distance += distance_matrix[last_c, c]
        new_hamiltonian_path_distance_found = (
            hamiltonian_path_distance
            + distance_matrix[b, d]
            - distance_matrix[c, d]
            + path_cb_distance
            - path_bc_distance
        )
        
        if new_hamiltonian_path_distance_found + distance_matrix[a, c] < hamiltonian_cycle_distance:
            return c, d, new_hamiltonian_path_distance_found
        
        if (tabu[c, d] != iteration and 
            new_hamiltonian_path_distance_found < hamiltonian_path_distance_found):
            hamiltonian_path_distance_found = new_hamiltonian_path_distance_found
            best_c = c
        
        last_c = c
        c = d
    
    return best_c, successors[best_c], hamiltonian_path_distance_found


def _print_message(msg: str, verbose: bool, log_file_handler: Optional[TextIO]) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)
    if verbose:
        print(msg)


# =============================================================================
# CLUSTERING FUNCTIONS
# =============================================================================

def _partition_vertices_by_pivot(
    vertices: List[int], 
    distance_matrix: np.ndarray, 
    max_size: int
) -> List[List[int]]:
    """
    Recursively partition vertices into clusters of size <= max_size.
    Uses geographical clustering based on distance.
    """
    if len(vertices) <= max_size:
        return [vertices]
    
    pivot = vertices[0]
    others = [v for v in vertices if v != pivot]
    others.sort(key=lambda v: distance_matrix[pivot, v])
    split = len(others) // 2
    left = [pivot] + others[:split]
    right = others[split:]
    
    clusters = []
    clusters.extend(_partition_vertices_by_pivot(left, distance_matrix, max_size))
    clusters.extend(_partition_vertices_by_pivot(right, distance_matrix, max_size))
    return clusters


def _partition_kmeans_style(
    vertices: List[int],
    distance_matrix: np.ndarray,
    max_size: int
) -> List[List[int]]:
    """
    Partition vertices using a greedy k-means-like approach.
    Better for geographical data.
    """
    if len(vertices) <= max_size:
        return [vertices]
    
    n = len(vertices)
    k = (n + max_size - 1) // max_size  # Number of clusters needed
    
    # Initialize centroids using greedy selection
    centroids = [vertices[0]]
    for _ in range(1, k):
        # Find vertex farthest from current centroids
        best_v = None
        best_dist = -1
        for v in vertices:
            if v in centroids:
                continue
            min_dist = min(distance_matrix[v, c] for c in centroids)
            if min_dist > best_dist:
                best_dist = min_dist
                best_v = v
        if best_v:
            centroids.append(best_v)
    
    # Assign vertices to nearest centroid
    clusters: Dict[int, List[int]] = {c: [] for c in centroids}
    for v in vertices:
        if v in centroids:
            clusters[v].append(v)
        else:
            nearest = min(centroids, key=lambda c: distance_matrix[v, c])
            clusters[nearest].append(v)
    
    # If any cluster is too large, recursively partition
    result = []
    for cluster_vertices in clusters.values():
        if len(cluster_vertices) > max_size:
            result.extend(_partition_kmeans_style(cluster_vertices, distance_matrix, max_size))
        else:
            result.append(cluster_vertices)
    
    return result


def _rotate_cycle_to_start(cycle: List[int], start_node: int) -> List[int]:
    """Rotate cycle to start from a specific node."""
    if start_node not in cycle:
        return cycle[:]
    idx = cycle.index(start_node)
    return cycle[idx:] + cycle[:idx]


def _tour_distance(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a tour."""
    if not tour:
        return float('inf')
    n = len(tour)
    return sum(distance_matrix[tour[i], tour[(i + 1) % n]] for i in range(n))


def _compute_fixed_nodes_by_cluster(
    clusters: List[List[int]], 
    tour: List[int]
) -> List[int]:
    """Compute vertices with both neighbors in same cluster."""
    if not tour:
        return []
    
    vertex_cluster: Dict[int, int] = {}
    for ci, cluster in enumerate(clusters):
        for v in cluster:
            vertex_cluster[v] = ci
    
    n = len(tour)
    fixed = []
    for i, v in enumerate(tour):
        prev_v = tour[(i - 1) % n]
        next_v = tour[(i + 1) % n]
        if (vertex_cluster.get(prev_v, -1) == vertex_cluster.get(v, -1) and
            vertex_cluster.get(next_v, -1) == vertex_cluster.get(v, -1)):
            fixed.append(v)
    
    return fixed


# =============================================================================
# COMBINED SOLVER (DP + LIN-KERNIGHAN)
# =============================================================================

VERBOSE = False

def solve_tsp_combined(
    distance_matrix: np.ndarray,
    max_cluster_size: int = 18,
    use_kmeans: bool = True,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float, List[int]]:
    """
    Combined solver: splits graph into clusters of max_cluster_size,
    solves each with DP, then joins with Lin-Kernighan.
    
    Parameters
    ----------
    distance_matrix
        Distance matrix (n x n)
    max_cluster_size
        Maximum size for DP-solved clusters (default 25)
    use_kmeans
        Use k-means style clustering (better for geographical data)
    log_file
        Optional log file path
    verbose
        Print progress messages
        
    Returns
    -------
    Tuple
        (final_cycle, total_distance, fixed_nodes)
    """
    start_time = time.time()
    num_vertices = distance_matrix.shape[0]
    vertices = list(range(num_vertices))
    
    if VERBOSE:
        print(f"[combined] start: n={num_vertices}, max_cluster_size={max_cluster_size}")
    
    # Small instance: use exact DP directly
    if num_vertices <= max_cluster_size:
        t0 = time.time()
        cycle, dist = solve_tsp_dynamic_programming(distance_matrix)
        t1 = time.time()
        if verbose:
            print(f"[combined] small instance solved by DP in {t1 - t0:.3f}s")
        return cycle, dist, list(range(num_vertices))
    
    # Partition vertices into clusters
    t0 = time.time()
    if use_kmeans:
        clusters = _partition_kmeans_style(vertices, distance_matrix, max_cluster_size)
    else:
        clusters = _partition_vertices_by_pivot(vertices, distance_matrix, max_cluster_size)
    t1 = time.time()
    if verbose:
        print(f"[combined] partitioned into {len(clusters)} clusters in {t1 - t0:.3f}s")
        for i, c in enumerate(clusters):
            print(f"  cluster {i}: {len(c)} vertices")
    
    # Solve each cluster with DP
    cluster_cycles: List[List[int]] = []
    for ci, cluster in enumerate(clusters):
        t0 = time.time()
        
        if len(cluster) <= max_cluster_size:
            # Use DP for small clusters
            submat = distance_matrix[np.ix_(cluster, cluster)]
            local_cycle, _ = solve_tsp_dynamic_programming(submat)
            global_cycle = [cluster[i] for i in local_cycle]
        else:
            # Use combined solver recursively for large clusters
            submat = distance_matrix[np.ix_(cluster, cluster)]
            local_cycle, _, _ = solve_tsp_combined(submat, max_cluster_size, use_kmeans)
            global_cycle = [cluster[i] for i in local_cycle]
        
        t1 = time.time()
        cluster_cycles.append(global_cycle)
        if verbose:
            print(f"[combined] cluster {ci} size={len(cluster)} solved in {t1 - t0:.3f}s")
    
    # Build contracted distance matrix between clusters
    k = len(clusters)
    contracted = np.zeros((k, k))
    best_pairs: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    for i in range(k):
        for j in range(k):
            if i == j:
                contracted[i, j] = 0.0
                continue
            
            # Find minimum distance between clusters
            min_d = float('inf')
            best_pair = (clusters[i][0], clusters[j][0])
            for u in clusters[i]:
                for v in clusters[j]:
                    d = distance_matrix[u, v]
                    if d < min_d:
                        min_d = d
                        best_pair = (u, v)
            contracted[i, j] = min_d
            best_pairs[(i, j)] = best_pair
    
    # Solve contracted TSP with Lin-Kernighan
    t0 = time.time()
    cluster_candidates = build_candidate_sets(contracted, k=5)
    contracted_order, _ = solve_tsp_lin_kernighan(contracted,candidates=cluster_candidates)
    t1 = time.time()
    if verbose:
        print(f"[combined] contracted TSP (k={k}) solved by LK in {t1 - t0:.3f}s")
        print(f"[combined] contracted order: {contracted_order}")
    
    # Build initial global tour by connecting clusters
    ordered_clusters = contracted_order
    global_tour: List[int] = []
    
    for idx_pos in range(len(ordered_clusters)):
        ci = ordered_clusters[idx_pos]
        cj = ordered_clusters[(idx_pos + 1) % len(ordered_clusters)]
        u, v = best_pairs[(ci, cj)]
        
        entry_node = cluster_tour[0]  # ou melhor escolha
        rotated = rotate_to_start(cluster_tour, entry_node)
        global_tour += rotated[:-1]
    
    # Remove duplicates while preserving order
    seen = set()
    final_tour = []
    for v in global_tour:
        if v not in seen:
            final_tour.append(v)
            seen.add(v)
    
    # Add any missing vertices
    for v in vertices:
        if v not in seen:
            final_tour.append(v)
    
    # Compute fixed nodes
    t0 = time.time()
    fixed_nodes = _compute_fixed_nodes_by_cluster(clusters, final_tour)
    t1 = time.time()
    if verbose:
        print(f"[combined] computed {len(fixed_nodes)} fixed nodes in {t1 - t0:.3f}s")
    node_to_cluster = {}
    for cid, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cid
    # Refine with Lin-Kernighan (only joining clusters, not modifying DP-optimized parts)
    if len(final_tour) == num_vertices:
        if verbose:
            print(f"[combined] starting LK refinement with {len(fixed_nodes)} protected nodes")
        t0 = time.time()
        node_candidates = build_candidate_sets(distance_matrix, k=15)
        improved_cycle, improved_dist = solve_tsp_lin_kernighan(
            distance_matrix,
            x0=final_tour,
            candidates=node_candidates,
            fixed_nodes=fixed_nodes,
            node_to_cluster=node_to_cluster
        )
        t1 = time.time()
        if verbose:
            print(f"[combined] LK refinement finished in {t1 - t0:.3f}s")
            print(f"[combined] total time: {time.time() - start_time:.3f}s")
        return improved_cycle, improved_dist, fixed_nodes
    
    # Fallback
    if verbose:
        print("[combined] fallback: running LK without initial tour")
    t0 = time.time()
    cycle, dist = solve_tsp_lin_kernighan(distance_matrix, log_file=log_file, verbose=verbose)
    t1 = time.time()
    if verbose:
        print(f"[combined] total time: {time.time() - start_time:.3f}s")
    return cycle, dist, fixed_nodes


# =============================================================================
# MULTI-INSTANCE TSP SOLVER (KEY FEATURE)
# =============================================================================

def solve_single_tsp(
    filepath: str,
    max_cluster_size: int = 16,
    use_kmeans: bool = True,
    verbose: bool = False,
) -> Tuple[List[int], float, Dict]:

    start_time = time.time()
    
    if verbose:
        print("=" * 60)
        print("[multi-instance] Starting Multi-Instance TSP Solver")
        print("=" * 60)
    
    dm, nodes, metadata = _load_tsp_file(filepath)
    
    if dm.shape[0] == 0:
        raise ValueError(f"No nodes found in '{filepath}'")

    
    if verbose:
        print(f"[multi-instance] Loaded {len(instances)} instances")
    
    # Solve each instance
    instance_solutions: List[Dict] = []
    
    for filename, dm, nodes, metadata in instances:
        t0 = time.time()
        n = dm.shape[0]
        
        if verbose:
            print(f"\n[multi-instance] Processing {filename} ({n} nodes)")
        
        # Choose solver based on size
        if n <= max_cluster_size:
            # Use exact DP for small instances
            cycle, dist = solve_tsp_dynamic_programming(dm)
            solver_used = "DP"
        else:
            # Use combined solver for larger instances
            cycle, dist, _ = solve_tsp_combined(dm, max_cluster_size, use_kmeans, verbose=verbose)
            solver_used = "Combined"
        
        t1 = time.time()
        
        # Store solution
        solution = {
            'filename': filename,
            'nodes': nodes,
            'cycle': cycle,
            'distance': dist,
            'solver': solver_used,
            'time': t1 - t0,
            'metadata': metadata
        }
        instance_solutions.append(solution)
        
        if verbose:
            print(f"  -> {solver_used} solution: distance = {dist:.2f}, time = {t1-t0:.3f}s")
    
    # Build meta-graph connecting instances
    if verbose:
        print(f"\n[multi-instance] Building meta-graph connecting {len(instances)} instances")
    
    # Collect representative nodes from each instance (for meta-graph)
    # Use the optimal tour nodes as representatives
    meta_nodes: List[Tuple[int, int]] = []  # (instance_idx, local_node_idx)
    instance_start_indices: List[int] = []
    
    current_idx = 0
    for sol in instance_solutions:
        instance_start_indices.append(current_idx)
        # Use all nodes from each instance
        for local_node in sol['cycle']:
            meta_nodes.append((len(instance_solutions) - 1, local_node))
        current_idx += len(sol['cycle'])
    
    # Actually, better approach: concatenate all instance solutions
    # and build a meta-tour that visits each instance optimally
    
    # Build concatenated initial tour
    global_tour: List[int] = []  # (instance_idx, local_node_idx)
    instance_global_ranges: List[Tuple[int, int]] = []  # (start, end) indices in global tour
    
    for inst_idx, sol in enumerate(instance_solutions):
        start_idx = len(global_tour)
        for local_node in sol['cycle']:
            global_tour.append((inst_idx, local_node))
        end_idx = len(global_tour)
        instance_global_ranges.append((start_idx, end_idx))
    
    # Build distance matrix for meta-tour
    # This connects nodes across instances
    total_nodes = len(global_tour)
    meta_distance = np.zeros((total_nodes, total_nodes))
    
    # For intra-instance distances, use original instance distances
    # For inter-instance distances, use minimum distance between instances
    
    # Precompute inter-instance minimum distances
    n_instances = len(instance_solutions)
    inter_inst_dist = np.zeros((n_instances, n_instances))
    inter_inst_pairs: Dict[Tuple[int, int], Tuple[int, int]] = {}
    
    for i in range(n_instances):
        for j in range(n_instances):
            if i == j:
                inter_inst_dist[i, j] = 0.0
            else:
                dm_i = instances[i][1]  # distance matrix
                dm_j = instances[j][1]
                nodes_i = instance_solutions[i]['cycle']
                nodes_j = instance_solutions[j]['cycle']
                
# Find minimum distance between instances
                # Use range(min(len(dm_i), len(dm_j))) to ensure valid indices
                n_i = min(len(dm_i), len(nodes_i))
                n_j = min(len(dm_j), len(nodes_j))
                min_d = float('inf')
                best_pair = (0, 0)
                for idx_i, ni in enumerate(nodes_i[:n_i]):
                    for idx_j, nj in enumerate(nodes_j[:n_j]):
                        if ni < len(dm_i) and nj < len(dm_i):
                            d = dm_i[ni, nj]
                            if d < min_d:
                                min_d = d
                                best_pair = (ni, nj)
                inter_inst_dist[i, j] = min_d
                inter_inst_pairs[(i, j)] = best_pair
    
    # Build full meta distance matrix
    for i in range(total_nodes):
        inst_i, local_i = global_tour[i]
        for j in range(total_nodes):
            inst_j, local_j = global_tour[j]
            
            if inst_i == inst_j:
                # Intra-instance: use original distance
                meta_distance[i, j] = instances[inst_i][1][local_i, local_j]
            else:
                # Inter-instance: use precomputed minimum
                meta_distance[i, j] = inter_inst_dist[inst_i, inst_j]
    
    # Solve meta-tour with Lin-Kernighan
    if verbose:
        print(f"[multi-instance] Solving meta-tour (n={total_nodes}) with Lin-Kernighan")
    
    t0 = time.time()
    improved_cycle, improved_dist = solve_tsp_lin_kernighan(
        meta_distance, 
        x0=list(range(total_nodes)),  # Use concatenated order as initial
        verbose=verbose
    )
    t1 = time.time()
    
    if verbose:
        print(f"[multi-instance] Meta-tour optimization: {improved_dist:.2f} in {t1-t0:.3f}s")
    
    # Convert back to original node IDs
    final_tour_node_ids = []
    for meta_idx in improved_cycle:
        inst_idx, local_node = global_tour[meta_idx]
        original_node_id = instances[inst_idx][2][local_node]  # node_ids[local_node]
        final_tour_node_ids.append(original_node_id)
    
    total_time = time.time() - start_time
    
    # Build results summary
    results = {
        'final_tour': final_tour_node_ids,
        'final_distance': improved_dist,
        'instance_solutions': instance_solutions,
        'inter_instance_distances': inter_inst_dist.tolist(),
        'total_time': total_time,
        'num_instances': len(instances),
        'max_cluster_size': max_cluster_size
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("[multi-instance] FINAL RESULTS")
        print("=" * 60)
        print(f"Final tour: {final_tour_node_ids}")
        print(f"Final distance: {improved_dist:.2f}")
        print(f"Total time: {total_time:.3f}s")
        
        # Print per-instance summary
        print("\nPer-instance results:")
        for sol in instance_solutions:
            print(f"  {sol['filename']}: {sol['solver']} -> {sol['distance']:.2f}")
    
    return final_tour_node_ids, improved_dist, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Lin-Kernighan TSP Solver")
    parser.add_argument("--instances", default="instances", help="Instances directory")
    parser.add_argument("--max-cluster", type=int, default=18, help="Max cluster size for DP")
    parser.add_argument("--no-kmeans", action="store_true", help="Disable k-means clustering")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()
    
    # Run single instance solver - first file in instances/
    instances_dir = args.instances
    tsp_files = [f for f in os.listdir(instances_dir) if f.endswith('.tsp')]
    if not tsp_files:
        print(f"No TSP files found in {instances_dir}")
        exit(1)
    
    filepath = os.path.join(instances_dir, tsp_files[0])
    tour, dist, results = solve_tsp_combined(
        _load_tsp_file(filepath)[0],  # just distance matrix
        max_cluster_size=args.max_cluster,
        use_kmeans=not args.no_kmeans,
        verbose=args.verbose
    )

    
    print("\n" + "=" * 60)
    print("FINAL SOLUTION")
    print("=" * 60)
    print(f"Optimal tour: {tour}")
    print(f"Total distance: {dist:.2f}")
    print(f"Total computation time: {results['total_time']:.3f}s")

