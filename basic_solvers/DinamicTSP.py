from math import *
import numpy as np
from python_tsp.exact import solve_tsp_brute_force
from python_tsp.utils import setup_initial_solution
from functools import lru_cache
from python_tsp.heuristics import solve_tsp_lin_kernighan
from typing import Dict, List, Optional, Tuple, TextIO
import tsplib95
import time


n = 0
start = 0
visited = []

problem = tsplib95.load('wi29.tsp')

print(problem.render())


def solve_tsp_dynamic_programming(
    distance_matrix: np.ndarray,
    maxsize: Optional[int] = None,
) -> Tuple[List, float]:
    """
    Solve TSP to optimality with dynamic programming

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j. It does not need to be symmetric

    maxsize
        Parameter passed to ``lru_cache`` decorator. Used to define the maximum
        size for the recursion tree. Defaults to `None`, which essentially
        means "take as much space as needed".

    Returns
    -------
    permutation
        A permutation of nodes from 0 to n that produces the least total
        distance

    distance
        The total distance the optimal permutation produces

    Notes
    -----
    Algorithm: cost of the optimal path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Consider a TSP instance with 3 nodes: {0, 1, 2}. Let dist(0, {1, 2}) be the
    distance from 0, visiting all nodes in {1, 2} and going back to 0. This can
    be computed recursively as:

        dist(0, {1, 2}) = min(
            c_{0, 1} + dist(1, {2}),
            c_{0, 2} + dist(2, {1}),
        )

    wherein c_{0, 1} is the cost from going from 0 to 1 in the distance matrix.
    The inner dist(1, {2}) is computed as:

        dist(1, {2}) = min(
            c_{1, 2} + dist(2, {}),
        )

    and similarly for dist(2, {1}). The stopping point in the recursion is:

        dist(2, {}) = c_{2, 0}.

    This process can be generalized as:

        dist(ni, N) =   min   ( c_{ni, nj} + dist(nj, N - {nj}) )
                      nj in N

    and

        dist(ni, {}) = c_{ni, 0}

    With starting point as dist(0, {1, 2, ..., tsp_size}). The notation
    N - {nj} is the difference operator, meaning set N without node nj.


    Algorithm: compute the optimal path
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The previous process returns the distance of the optimal path. To find the
    actual path, we need to store in a memory the following key/values:

        memo[(ni, N)] = nj_min

    with nj_min the node in N that provided the smallest value of dist(ni, N).
    Then, the process goes backwards starting from
    memo[(0, {1, 2, ..., tsp_size})].

    In the previous example, suppose memo[(0, {1, 2})] = 1.
    Then, look for memo[(1, {2})] = 2.
    Then, since the next step would be memo[2, {}], stop there. The optimal
    path would be 0 -> 1 -> 2 -> 0.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm#cite_note-5
    """
    # Get initial set {1, 2, ..., tsp_size} as a frozenset because @lru_cache
    # requires a hashable type
    N = frozenset(range(1, distance_matrix.shape[0]))
    memo: Dict[Tuple, int] = {}

    # Step 1: get minimum distance
    @lru_cache(maxsize=maxsize)
    def dist(ni: int, N: frozenset) -> float:
        if not N:
            return distance_matrix[ni, 0]

        # Store the costs in the form (nj, dist(nj, N))
        costs = [
            (nj, distance_matrix[ni, nj] + dist(nj, N.difference({nj})))
            for nj in N
        ]
        nmin, min_cost = min(costs, key=lambda x: x[1])
        memo[(ni, N)] = nmin

        return min_cost

    best_distance = dist(0, N)

    # Step 2: get path with the minimum distance
    ni = 0  # start at the origin
    solution = [0]

    while N:
        ni = memo[(ni, N)]
        solution.append(ni)
        N = N.difference({ni})

    return solution, best_distance



def _cycle_to_successors(cycle: List[int]) -> List[int]:
    """
    Convert a cycle representation to successors representation.

    Parameters
    ----------
    cycle
        A list representing a cycle.

    Returns
    -------
    List
        A list representing successors.
    """
    successors = cycle[:]
    n = len(cycle)
    for i, _ in enumerate(cycle):
        successors[cycle[i]] = cycle[(i + 1) % n]
    return successors


def _successors_to_cycle(successors: List[int]) -> List[int]:
    """
    Convert a successors representation to a cycle representation.

    Parameters
    ----------
    successors
        A list representing successors.

    Returns
    -------
    List
        A list representing a cycle.
    """
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
    """
    Minimize the Hamiltonian path distance after ejecting an edge.

    Parameters
    ----------
    tabu
        A NumPy array for tabu management.

    iteration
        The current iteration.

    successors
        A list representing successors.

    ejected_edge
        The edge that was ejected.

    distance_matrix
        A NumPy array representing the distance matrix.

    hamiltonian_path_distance
        The Hamiltonian path distance.

    hamiltonian_cycle_distance
        The Hamiltonian cycle distance.

    Returns
    -------
    Tuple
        The best c, d, and the new Hamiltonian path distance found.
    """
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

        if (
            new_hamiltonian_path_distance_found + distance_matrix[a, c]
            < hamiltonian_cycle_distance
        ):
            return c, d, new_hamiltonian_path_distance_found

        if (
            tabu[c, d] != iteration
            and new_hamiltonian_path_distance_found
            < hamiltonian_path_distance_found
        ):
            hamiltonian_path_distance_found = (
                new_hamiltonian_path_distance_found
            )
            best_c = c

        last_c = c
        c = d

    return best_c, successors[best_c], hamiltonian_path_distance_found


def _print_message(
    msg: str, verbose: bool, log_file_handler: Optional[TextIO]
) -> None:
    if log_file_handler:
        print(msg, file=log_file_handler)

    if verbose:
        print(msg)


def _solve_tsp_brute_force(
    distance_matrix: np.ndarray,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    x, fx = solve_tsp_brute_force(distance_matrix)
    x = x or []

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )
    msg = (
        "Few nodes to use Lin-Kernighan heuristics, "
        "using Brute Force instead. "
    )
    if not x:
        msg += "No solution found."
    else:
        msg += f"Found value: {fx}"
    _print_message(msg, verbose, log_file_handler)

    if log_file_handler:
        log_file_handler.close()

    return x, fx


def solve_tsp_lin_kernighan(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesperson Problem using the Lin-Kernighan algorithm.

    Parameters
    ----------
    distance_matrix
        Distance matrix of shape (n x n) with the (i, j) entry indicating the
        distance from node i to j

    x0
        Initial permutation. If not provided, it starts with a random path.

    log_file
        If not `None`, creates a log file with details about the whole
        execution.

    verbose
        If true, prints algorithm status every iteration.

    Returns
    -------
    Tuple
        A tuple containing the Hamiltonian cycle and its distance.

    References
    ----------
    Éric D. Taillard, "Design of Heuristic Algorithms for Hard Optimization,"
    Chapter 5, Section 5.3.2.1: Lin-Kernighan Neighborhood, Springer, 2023.
    """
    num_vertices = distance_matrix.shape[0]
    if num_vertices < 4:
        return _solve_tsp_brute_force(distance_matrix, log_file, verbose)

    hamiltonian_cycle, hamiltonian_cycle_distance = setup_initial_solution(
        distance_matrix=distance_matrix, x0=x0
    )
    vertices = list(range(num_vertices))
    iteration = 0
    improvement = True
    tabu = np.zeros(shape=(num_vertices, num_vertices), dtype=int)

    log_file_handler = (
        open(log_file, "w", encoding="utf-8") if log_file else None
    )

    while improvement:
        iteration += 1
        improvement = False
        successors = _cycle_to_successors(hamiltonian_cycle)

        # Eject edge [a, b] to start the chain and compute the Hamiltonian
        # path distance obtained by ejecting edge [a, b] from the cycle
        # as reference.
        a = int(distance_matrix[vertices, successors].argmax())
        b = successors[a]
        hamiltonian_path_distance = (
            hamiltonian_cycle_distance - distance_matrix[a, b]
        )

        while True:
            ejected_edge = a, b

            # Find the edge [c, d] that minimizes the Hamiltonian path obtained
            # by removing edge [c, d] and adding edge [b, d], with [c, d] not
            # removed in the current ejection chain.
            (
                c,
                d,
                hamiltonian_path_distance_found,
            ) = _minimizes_hamiltonian_path_distance(
                tabu,
                iteration,
                successors,
                ejected_edge,
                distance_matrix,
                hamiltonian_path_distance,
                hamiltonian_cycle_distance,
            )

            # If the Hamiltonian cycle cannot be improved, return
            # to the solution and try another ejection.
            if hamiltonian_path_distance_found >= hamiltonian_cycle_distance:
                break

            # Update Hamiltonian path distance reference
            hamiltonian_path_distance = hamiltonian_path_distance_found

            # Reverse the direction of the path from b to c
            i, si, successors[b] = b, successors[b], d
            while i != c:
                successors[si], i, si = i, si, successors[si]

            # Don't remove again the minimal edge found
            tabu[c, d] = tabu[d, c] = iteration

            # c plays the role of b in the next iteration
            b = c

            msg = (
                f"Current value: {hamiltonian_cycle_distance}; "
                f"Ejection chain: {iteration}"
            )
            _print_message(msg, verbose, log_file_handler)

            # If the Hamiltonian cycle improves, update the solution
            if (
                hamiltonian_path_distance + distance_matrix[a, b]
                < hamiltonian_cycle_distance
            ):
                improvement = True
                successors[a] = b
                hamiltonian_cycle = _successors_to_cycle(successors)
                hamiltonian_cycle_distance = (
                    hamiltonian_path_distance + distance_matrix[a, b]
                )

    if log_file_handler:
        log_file_handler.close()

    return hamiltonian_cycle, hamiltonian_cycle_distance


def _partition_vertices_by_pivot(
    vertices: List[int], distance_matrix: np.ndarray, max_size: int
) -> List[List[int]]:
    """
    Recursively partition vertices into clusters of size <= max_size.
    Uses a simple pivot + distance sorting split to keep nearby nodes together.
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


def _rotate_cycle_to_start(cycle: List[int], start_node: int) -> List[int]:
    if start_node not in cycle:
        return cycle[:]
    idx = cycle.index(start_node)
    return cycle[idx:] + cycle[:idx]


def _tour_distance(tour: List[int], distance_matrix: np.ndarray) -> float:
    if not tour:
        return float('inf')
    n = len(tour)
    dist = 0.0
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        dist += distance_matrix[a, b]
    return dist


def _compute_fixed_nodes_by_cluster(
    clusters: List[List[int]], tour: List[int]
) -> List[int]:
    """
    Compute vertices that have both neighbors in the tour belonging
    to the same cluster as the vertex itself. These vertices are
    considered "fixed" (do not need to move when starting Lin-Kernighan).
    Returns the list of fixed vertex indices.
    """
    if not tour:
        return []
    # Map vertex -> cluster index
    vertex_cluster: Dict[int, int] = {}
    for ci, cluster in enumerate(clusters):
        for v in cluster:
            vertex_cluster[v] = ci

    n = len(tour)
    fixed = []
    for i, v in enumerate(tour):
        prev_v = tour[(i - 1) % n]
        next_v = tour[(i + 1) % n]
        if (
            vertex_cluster.get(prev_v, -1) == vertex_cluster.get(v, -1)
            and vertex_cluster.get(next_v, -1) == vertex_cluster.get(v, -1)
        ):
            fixed.append(v)
    return fixed


def solve_tsp_combined(
    distance_matrix: np.ndarray,
    max_cluster_size: int = 25,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float, List[int]]:
    """
    Combined solver: splits the graph until clusters have at most
    `max_cluster_size` vertices, solves each cluster with the dynamic
    programming solver, then connects clusters using a contracted-graph
    Lin-Kernighan run and refines the final tour with Lin-Kernighan.
    """
    start_time = time.time()
    num_vertices = distance_matrix.shape[0]
    vertices = list(range(num_vertices))

    print(f"[combined] start: n={num_vertices}, max_cluster_size={max_cluster_size}")

    # Small instance: use exact DP
    if num_vertices <= max_cluster_size:
        t0 = time.time()
        cycle, dist = solve_tsp_dynamic_programming(distance_matrix)
        t1 = time.time()
        print(f"[combined] small instance solved by DP in {t1 - t0:.3f}s")
        # all nodes are effectively fixed in a small optimal solution
        fixed_nodes = list(range(num_vertices))
        print(f"[combined] finished total in {time.time() - start_time:.3f}s")
        return cycle, dist, fixed_nodes


    def _build_distance_matrix_from_problem(problem) -> Tuple[np.ndarray, List[int]]:
            """Build a distance matrix (NxN) and return the original node id order.

            Tries multiple tsplib95 APIs (`get_weight`, `get_edge_weight`, `node_coords`,
            `node_coordinates`, or `get_graph`) to construct the matrix.
            """
            nodes = list(problem.get_nodes())
            n = len(nodes)
            idx = {node: i for i, node in enumerate(nodes)}
            mat = np.zeros((n, n))

            # Try get_weight
            if hasattr(problem, "get_weight"):
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        mat[i, j] = 0 if i == j else problem.get_weight(u, v)
                return mat, nodes

            # Try get_edge_weight
            if hasattr(problem, "get_edge_weight"):
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        mat[i, j] = 0 if i == j else problem.get_edge_weight(u, v)
                return mat, nodes

            # Try node coordinates (Euclidean fallback)
            coords = None
            if hasattr(problem, "node_coords"):
                coords = problem.node_coords
            elif hasattr(problem, "node_coordinates"):
                coords = problem.node_coordinates

            if coords:
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        if i == j:
                            mat[i, j] = 0
                            continue
                        x1, y1 = coords[u]
                        x2, y2 = coords[v]
                        mat[i, j] = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                return mat, nodes

            # Try graph representation
            try:
                g = problem.get_graph()
                for i, u in enumerate(nodes):
                    for j, v in enumerate(nodes):
                        if i == j:
                            mat[i, j] = 0
                            continue
                        w = g[u][v].get("weight", g[u][v].get("cost", 0))
                        mat[i, j] = w
                return mat, nodes
            except Exception:
                pass

            raise RuntimeError("Unable to construct distance matrix from `problem` instance")


    if __name__ == "__main__":
            # Build distance matrix from the loaded TSPLIB problem
            print("[main] building distance matrix from `problem`")
            dm, node_order = _build_distance_matrix_from_problem(problem)

            # Run the combined solver (clusters <= 25 by default)
            print("[main] running solve_tsp_combined...")
            t0 = time.time()
            cycle, dist, fixed_nodes = solve_tsp_combined(dm, max_cluster_size=25, verbose=True)
            t1 = time.time()
            print(f"[main] solve_tsp_combined took {t1 - t0:.3f}s")

            # Map cycle indices back to original TSPLIB node ids
            tour_node_ids = [node_order[i] for i in cycle]
            fixed_node_ids = [node_order[i] for i in fixed_nodes]

            print("Combined solver result:")
            print("Tour (node ids):", tour_node_ids)
            print("Distance:", dist)
            print("Fixed nodes (node ids):", fixed_node_ids)

    # Partition vertices into clusters
    t0 = time.time()
    clusters = _partition_vertices_by_pivot(vertices, distance_matrix, max_cluster_size)
    t1 = time.time()
    print(f"[combined] partitioned into {len(clusters)} clusters in {t1 - t0:.3f}s")

    # Solve each cluster optimally using dynamic programming
    cluster_cycles: List[List[int]] = []
    for ci, cluster in enumerate(clusters):
        t0 = time.time()
        submat = distance_matrix[np.ix_(cluster, cluster)]
        local_cycle, _ = solve_tsp_dynamic_programming(submat)
        t1 = time.time()
        # map local indices to global vertex indices
        global_cycle = [cluster[i] for i in local_cycle]
        cluster_cycles.append(global_cycle)
        print(f"[combined] cluster {ci} size={len(cluster)} solved by DP in {t1 - t0:.3f}s")

    # Build contracted distance matrix between clusters (min pairwise)
    k = len(clusters)
    contracted = np.zeros((k, k))
    best_pairs: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i in range(k):
        for j in range(k):
            if i == j:
                contracted[i, j] = 0.0
                continue
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

    # Solve contracted TSP using Lin-Kernighan
    t0 = time.time()
    contracted_order, _ = solve_tsp_lin_kernighan(contracted)
    t1 = time.time()
    print(f"[combined] contracted TSP (k={k}) solved by LK in {t1 - t0:.3f}s")
    print(f"[combined] contracted order: {contracted_order}")

    # Build an initial global tour by ordering clusters and rotating
    # each cluster cycle so it starts at the node that connects best to
    # the next cluster in the contracted order.
    ordered_clusters = contracted_order
    global_tour: List[int] = []
    for idx_pos in range(len(ordered_clusters)):
        ci = ordered_clusters[idx_pos]
        cj = ordered_clusters[(idx_pos + 1) % len(ordered_clusters)]
        u, v = best_pairs[(ci, cj)]

        # rotate cluster ci's cycle so it starts at u
        rotated = _rotate_cycle_to_start(cluster_cycles[ci], u)
        # append rotated cycle nodes
        global_tour.extend(rotated)

    # Ensure we have a valid permutation (may contain duplicates due to naive concat)
    # We'll create final tour by keeping first occurrence of each vertex in order,
    # then appending any missing vertices (shouldn't happen but safe).
    seen = set()
    final_tour = []
    for v in global_tour:
        if v not in seen:
            final_tour.append(v)
            seen.add(v)
    for v in vertices:
        if v not in seen:
            final_tour.append(v)

    # Compute fixed nodes according to cluster membership
    t0 = time.time()
    fixed_nodes = _compute_fixed_nodes_by_cluster(clusters, final_tour)
    t1 = time.time()
    print(f"[combined] computed fixed nodes count={len(fixed_nodes)} in {t1 - t0:.3f}s")

    # If final_tour has length n, we can refine with Lin-Kernighan starting from this tour
    if len(final_tour) == num_vertices:
        print(f"[combined] starting LK refinement from initial tour (n={num_vertices})")
        t0 = time.time()
        improved_cycle, improved_dist = solve_tsp_lin_kernighan(
            distance_matrix, x0=final_tour, log_file=log_file, verbose=verbose
        )
        t1 = time.time()
        print(f"[combined] LK refinement finished in {t1 - t0:.3f}s")
        print(f"[combined] total time: {time.time() - start_time:.3f}s")
        return improved_cycle, improved_dist, fixed_nodes

    # Fallback: run Lin-Kernighan without initial solution
    print("[combined] final fallback: running LK without initial tour")
    t0 = time.time()
    cycle, dist = solve_tsp_lin_kernighan(distance_matrix, log_file=log_file, verbose=verbose)
    t1 = time.time()
    print(f"[combined] fallback LK finished in {t1 - t0:.3f}s")
    print(f"[combined] total time: {time.time() - start_time:.3f}s")
    return cycle, dist, fixed_nodes
