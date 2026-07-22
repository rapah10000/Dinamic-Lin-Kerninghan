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


VERBOSE = False


# =============================================================================
# TSP FILE LOADING FUNCTIONS
# =============================================================================

def rotate_to_start(tour: List[int], start: int) -> List[int]:
    """Rotate a cycle so that it starts at `start`, preserving order."""
    idx = tour.index(start)
    return tour[idx:] + tour[:idx]


def _load_tsp_file(filepath: str) -> Tuple[np.ndarray, List[int], Dict]:
    """
    Load a TSP file and build distance matrix.

    Parameters
    ----------
    filepath : str
        Path to the .tsp file.

    Returns
    -------
    Tuple
        (distance_matrix, node_ids, metadata)
    """
    problem = tsplib95.load(filepath)

    # Get nodes
    nodes = list(problem.get_nodes())
    n = len(nodes)

    # Initialize distance matrix
    mat = np.zeros((n, n), dtype=float)

    # Detect edge weight type
    edge_type = getattr(problem, "edge_weight_type", "").upper()

    if edge_type == "EUC_2D" and hasattr(problem, "node_coords") and problem.node_coords:
        # Fast vectorized computation for Euclidean instances
        coords = problem.node_coords
        xy = np.array([coords[u] for u in nodes], dtype=float)

        diff = xy[:, None, :] - xy[None, :, :]
        mat = np.rint(np.sqrt((diff ** 2).sum(axis=-1))).astype(float)

    else:
        # Use TSPLIB's official distance function
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    mat[i, j] = problem.get_weight(u, v)

    # Extract metadata
    metadata = {
        "name": problem.name,
        "comment": problem.comment,
        "dimension": n,
        "edge_weight_type": edge_type,
    }

    return mat, nodes, metadata



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

from typing import List, Tuple
import numpy as np


def solve_tsp_dynamic_programming(
    distance_matrix: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Solve TSP exactly using the Held-Karp algorithm
    with bitmask dynamic programming.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix (n x n)

    Returns
    -------
    Tuple[List[int], float]
        (optimal tour, optimal distance)
    """

    n = distance_matrix.shape[0]

    if n == 0:
        return [], 0.0

    if n == 1:
        return [0], 0.0

    if n == 2:
        dist = (
            distance_matrix[0, 1]
            + distance_matrix[1, 0]
        )
        return [0, 1], float(dist)

    INF = float("inf")

    #
    # dp[(mask,last)] = minimum cost
    #
    dp = {}

    #
    # parent[(mask,last)] = predecessor
    #
    parent = {}

    #
    # Initial states
    #
    for k in range(1, n):
        mask = 1 << (k - 1)
        dp[(mask, k)] = distance_matrix[0, k]

    #
    # Iterate over subset sizes
    #
    full_mask = (1 << (n - 1)) - 1

    for mask in range(full_mask + 1):

        for last in range(1, n):

            if not (mask & (1 << (last - 1))):
                continue

            state = (mask, last)

            if state not in dp:
                continue

            current_cost = dp[state]

            #
            # Try every next city
            #
            remaining = full_mask ^ mask

            r = remaining

            while r:

                bit = r & -r
                nxt = bit.bit_length()

                new_mask = mask | bit

                new_cost = (
                    current_cost
                    + distance_matrix[last, nxt]
                )

                new_state = (new_mask, nxt)

                if (
                    new_state not in dp
                    or new_cost < dp[new_state]
                ):
                    dp[new_state] = new_cost
                    parent[new_state] = last

                r ^= bit

    #
    # Close the tour
    #
    best_cost = INF
    last_city = -1

    for last in range(1, n):

        state = (full_mask, last)

        if state not in dp:
            continue

        cost = (
            dp[state]
            + distance_matrix[last, 0]
        )

        if cost < best_cost:
            best_cost = cost
            last_city = last

    #
    # Reconstruct tour
    #
    mask = full_mask
    current = last_city

    tour = [0]

    reverse_path = []

    while current != 0:

        reverse_path.append(current)

        previous = parent.get((mask, current), 0)

        mask ^= 1 << (current - 1)

        current = previous

    tour.extend(reversed(reverse_path))

    return tour, float(best_cost)

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

    # Initialize centroids using greedy (farthest-point) selection
    centroids = [vertices[0]]
    for _ in range(1, k):
        best_v = None
        best_dist = -1.0
        for v in vertices:
            if v in centroids:
                continue
            min_dist = min(distance_matrix[v, c] for c in centroids)
            if min_dist > best_dist:
                best_dist = min_dist
                best_v = v
        # BUGFIX: `if best_v:` falhava quando o vértice escolhido era o índice 0
        # (0 é "falsy" em Python), descartando silenciosamente esse centróide.
        if best_v is not None:
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
# CLUSTER JOINING (STITCHING) — núcleo do algoritmo híbrido
# =============================================================================
#
# O DP resolve cada cluster como um CICLO fechado ótimo. Para encadear os
# clusters em um único tour global, é preciso "abrir" cada ciclo em algum
# ponto. A escolha desse ponto de corte impacta diretamente a qualidade do
# tour inicial entregue ao Lin-Kernighan: um corte ruim gera arestas de
# junção caras que o LK precisa gastar tempo consertando (ou pode nem
# conseguir, caindo em ótimo local). As funções abaixo escolhem o ponto de
# corte que minimiza o custo das duas arestas de conexão (entrada vinda do
# cluster anterior + saída para o próximo cluster).

def _best_cut_point_both_ends(
    cycle: List[int],
    distance_matrix: np.ndarray,
    entry_target: int,
    exit_target: int,
) -> List[int]:
    """
    Dado um ciclo fechado (ótimo internamente), testa todos os pontos de
    corte possíveis e devolve a rotação do ciclo que minimiza:

        dist(entry_target, primeiro_no) + dist(ultimo_no, exit_target)

    Como o ciclo é fechado, todas as rotações têm o mesmo custo interno
    (a soma das arestas internas não muda) — a única coisa que varia é o
    custo das duas arestas de conexão com os clusters vizinhos.
    """
    n = len(cycle)
    if n == 1:
        return cycle[:]

    best_cost = float('inf')
    best_rotation = cycle
    for i in range(n):
        rotated = cycle[i:] + cycle[:i]
        first_node = rotated[0]
        last_node = rotated[-1]
        cost = distance_matrix[entry_target, first_node] + distance_matrix[last_node, exit_target]
        if cost < best_cost:
            best_cost = cost
            best_rotation = rotated

    return best_rotation


def _local_2opt_on_joints(
    tour: List[int],
    distance_matrix: np.ndarray,
    joint_indices: List[int],
    window: int = 3,
    max_passes: int = 10,
) -> List[int]:
    """
    Aplica 2-opt restrito à vizinhança das junções entre clusters.

    Só considera trocas de arestas cujos índices estão a até `window`
    posições de uma junção (`joint_indices`) — os nós internos dos clusters,
    que já são ótimos pelo DP, não são tocados. Isso é muito mais barato do
    que rodar Lin-Kernighan completo sobre o tour inteiro, e já elimina boa
    parte da ineficiência da concatenação ingênua dos clusters.
    """
    n = len(tour)
    if n < 4:
        return tour

    tour = tour[:]
    for _ in range(max_passes):
        improved = False
        candidate_positions = set()
        for idx in joint_indices:
            for delta in range(-window, window + 1):
                candidate_positions.add((idx + delta) % n)

        candidates = sorted(candidate_positions)
        for i in candidates:
            for j in candidates:
                if i >= j:
                    continue
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                if a == c or a == d or b == c or b == d:
                    continue
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                if new_cost < old_cost - 1e-9:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True

        if not improved:
            break

    return tour


# =============================================================================
# COMBINED SOLVER (DP + LIN-KERNIGHAN)
# =============================================================================

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
        Maximum size for DP-solved clusters
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

    if verbose:
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

    # Solve each cluster with DP (or recursively, for oversized clusters)
    cluster_cycles: List[List[int]] = []
    for ci, cluster in enumerate(clusters):
        t0 = time.time()

        if len(cluster) <= max_cluster_size:
            submat = distance_matrix[np.ix_(cluster, cluster)]
            local_cycle, _ = solve_tsp_dynamic_programming(submat)
            global_cycle = [cluster[i] for i in local_cycle]
        else:
            submat = distance_matrix[np.ix_(cluster, cluster)]
            local_cycle, _, _ = solve_tsp_combined(submat, max_cluster_size, use_kmeans)
            global_cycle = [cluster[i] for i in local_cycle]

        t1 = time.time()
        cluster_cycles.append(global_cycle)
        if verbose:
            print(f"[combined] cluster {ci} size={len(cluster)} solved in {t1 - t0:.3f}s")

    # Build contracted distance matrix between clusters (best connecting pair per pair of clusters)
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

    # Solve contracted TSP with Lin-Kernighan (order in which clusters are visited)
    t0 = time.time()
    cluster_candidates = build_candidate_sets(contracted, k=5)
    contracted_order, _ = solve_tsp_lin_kernighan(contracted, candidates=cluster_candidates)
    t1 = time.time()
    if verbose:
        print(f"[combined] contracted TSP (k={k}) solved by LK in {t1 - t0:.3f}s")
        print(f"[combined] contracted order: {contracted_order}")

    # Build initial global tour by connecting clusters at their best cut points
    # (BUGFIX: versão anterior referenciava uma variável `cluster_tour` inexistente
    #  e descartava o par ótimo (u, v) calculado em `best_pairs`.)
    ordered_clusters = contracted_order
    global_tour: List[int] = []
    joint_indices: List[int] = []  # posições no tour global onde há uma junção entre clusters

    num_clusters = len(ordered_clusters)
    for idx_pos in range(num_clusters):
        ci = ordered_clusters[idx_pos]
        cj = ordered_clusters[(idx_pos + 1) % num_clusters]
        prev_ci = ordered_clusters[(idx_pos - 1) % num_clusters]

        # ponto de entrada vindo do cluster anterior, ponto de saída para o próximo
        entry_from_prev, _ = best_pairs[(prev_ci, ci)]
        _, exit_to_next = best_pairs[(ci, cj)]

        best_rotation = _best_cut_point_both_ends(
            cluster_cycles[ci],
            distance_matrix,
            entry_target=entry_from_prev,
            exit_target=exit_to_next,
        )

        joint_indices.append(len(global_tour))
        global_tour += best_rotation

    # Remove duplicates while preserving order (defensivo: não deveria haver
    # duplicados se cada nó pertence a exatamente um cluster, mas protege
    # contra inconsistências no particionamento)
    seen = set()
    final_tour = []
    for v in global_tour:
        if v not in seen:
            final_tour.append(v)
            seen.add(v)

    # Add any missing vertices (defensivo, mesma lógica acima)
    for v in vertices:
        if v not in seen:
            final_tour.append(v)

    # Refine the joints with a cheap, localized 2-opt before handing off to LK
    t0 = time.time()
    final_tour = _local_2opt_on_joints(final_tour, distance_matrix, joint_indices)
    t1 = time.time()
    if verbose:
        print(f"[combined] local 2-opt on joints finished in {t1 - t0:.3f}s")

    # Compute fixed nodes (both neighbors already in same cluster -> protected from LK)
    t0 = time.time()
    fixed_nodes = _compute_fixed_nodes_by_cluster(clusters, final_tour)
    t1 = time.time()
    if verbose:
        print(f"[combined] computed {len(fixed_nodes)} fixed nodes in {t1 - t0:.3f}s")

    node_to_cluster: Dict[int, int] = {}
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

    # Fallback (não deveria ser atingido em condições normais; mantido por segurança)
    if verbose:
        print("[combined] fallback: running LK without initial tour "
              f"(final_tour tinha {len(final_tour)} de {num_vertices} nós esperados)")
    t0 = time.time()
    cycle, dist = solve_tsp_lin_kernighan(distance_matrix, log_file=log_file, verbose=verbose)
    t1 = time.time()
    if verbose:
        print(f"[combined] total time: {time.time() - start_time:.3f}s")
    return cycle, dist, fixed_nodes


# =============================================================================
# SINGLE INSTANCE SOLVER
# =============================================================================

def solve_single_tsp(
    filepath: str,
    max_cluster_size: int = 64,
    use_kmeans: bool = True,
    verbose: bool = False,
) -> Tuple[List[int], float, Dict]:
    """
    Resolve uma única instância .tsp, escolhendo DP puro (instâncias pequenas)
    ou o solver combinado DP+LK (instâncias maiores que max_cluster_size).

    BUGFIX: a versão anterior desta função misturava a lógica de uma única
    instância com a de múltiplas instâncias (referenciava uma variável
    `instances` que nunca era definida). Esta versão resolve apenas o
    arquivo indicado; para múltiplas instâncias, use `solve_multi_instance_tsp`.

    Parameters
    ----------
    filepath
        Caminho para o arquivo .tsp
    max_cluster_size
        Tamanho máximo de cluster resolvido por DP
    use_kmeans
        Usar particionamento k-means style (recomendado para dados geográficos)
    verbose
        Imprimir mensagens de progresso

    Returns
    -------
    Tuple
        (tour com IDs originais dos nós, distância total, metadata/estatísticas)
    """
    start_time = time.time()

    dm, nodes, metadata = _load_tsp_file(filepath)
    n = dm.shape[0]

    if n == 0:
        raise ValueError(f"No nodes found in '{filepath}'")

    if verbose:
        print(f"[single] Loaded '{filepath}': {n} nodes")

    if n <= max_cluster_size:
        cycle, dist = solve_tsp_dynamic_programming(dm)
        solver_used = "DP"
        fixed_nodes = list(range(n))
    else:
        cycle, dist, fixed_nodes = solve_tsp_combined(
            dm, max_cluster_size=max_cluster_size, use_kmeans=use_kmeans, verbose=verbose
        )
        solver_used = "Combined"

    total_time = time.time() - start_time

    tour_node_ids = [nodes[i] for i in cycle]

    results = {
        'filename': os.path.basename(filepath),
        'solver': solver_used,
        'distance': dist,
        'time': total_time,
        'metadata': metadata,
        'fixed_nodes': fixed_nodes,
    }

    if verbose:
        print(f"[single] {solver_used} solution: distance = {dist:.2f}, time = {total_time:.3f}s")

    return tour_node_ids, dist, results


# =============================================================================
# MULTI-INSTANCE TSP SOLVER
# =============================================================================

def solve_multi_instance_tsp(
    instances_dir: str = "instances",
    max_cluster_size: int = 64,
    use_kmeans: bool = True,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int]], float, Dict]:
    """
    Resolve múltiplas instâncias .tsp de um diretório e as encadeia em um
    único meta-tour, conectando-as pelos pares de nós mais próximos entre
    instâncias e refinando com Lin-Kernighan.

    Parameters
    ----------
    instances_dir
        Diretório contendo os arquivos .tsp
    max_cluster_size
        Tamanho máximo de cluster resolvido por DP em cada instância
    use_kmeans
        Usar particionamento k-means style
    verbose
        Imprimir mensagens de progresso

    Returns
    -------
    Tuple
        (final_tour como lista de (instance_idx, node_id_original),
         distância final do meta-tour, dicionário de resultados)
    """
    start_time = time.time()

    if verbose:
        print("=" * 60)
        print("[multi-instance] Starting Multi-Instance TSP Solver")
        print("=" * 60)

    instances = _load_all_instances(instances_dir)
    if not instances:
        raise ValueError(f"No .tsp instances found in '{instances_dir}'")

    if verbose:
        print(f"[multi-instance] Loaded {len(instances)} instances")

    # Solve each instance
    instance_solutions: List[Dict] = []
    for filename, dm, nodes, metadata in instances:
        t0 = time.time()
        n = dm.shape[0]

        if verbose:
            print(f"\n[multi-instance] Processing {filename} ({n} nodes)")

        if n <= max_cluster_size:
            cycle, dist = solve_tsp_dynamic_programming(dm)
            solver_used = "DP"
        else:
            cycle, dist, _ = solve_tsp_combined(dm, max_cluster_size, use_kmeans, verbose=verbose)
            solver_used = "Combined"

        t1 = time.time()

        instance_solutions.append({
            'filename': filename,
            'nodes': nodes,
            'cycle': cycle,
            'distance': dist,
            'solver': solver_used,
            'time': t1 - t0,
            'metadata': metadata,
        })

        if verbose:
            print(f"  -> {solver_used} solution: distance = {dist:.2f}, time = {t1 - t0:.3f}s")

    if verbose:
        print(f"\n[multi-instance] Building meta-graph connecting {len(instances)} instances")

    # Concatenate all instance solutions into a single initial tour
    global_tour: List[Tuple[int, int]] = []  # (instance_idx, local_node_idx)
    for inst_idx, sol in enumerate(instance_solutions):
        for local_node in sol['cycle']:
            global_tour.append((inst_idx, local_node))

    total_nodes = len(global_tour)

    # Precompute inter-instance minimum distances (used for edges crossing instances)
    n_instances = len(instance_solutions)
    inter_inst_dist = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(n_instances):
            if i == j:
                inter_inst_dist[i, j] = 0.0
                continue

            dm_i = instances[i][1]
            nodes_i = instance_solutions[i]['cycle']
            nodes_j = instance_solutions[j]['cycle']

            min_d = float('inf')
            for ni in nodes_i:
                for nj in nodes_j:
                    # BUGFIX: a versão anterior comparava `nj < len(dm_i)`, mas os
                    # índices de nj pertencem à instância j, não à i — deveria ser
                    # validado contra o tamanho da própria instância de origem de nj.
                    if ni < dm_i.shape[0] and nj < dm_i.shape[0]:
                        d = dm_i[ni, nj]
                        if d < min_d:
                            min_d = d
            inter_inst_dist[i, j] = min_d

    # Build full meta distance matrix
    meta_distance = np.zeros((total_nodes, total_nodes))
    for i in range(total_nodes):
        inst_i, local_i = global_tour[i]
        for j in range(total_nodes):
            inst_j, local_j = global_tour[j]
            if inst_i == inst_j:
                meta_distance[i, j] = instances[inst_i][1][local_i, local_j]
            else:
                meta_distance[i, j] = inter_inst_dist[inst_i, inst_j]

    if verbose:
        print(f"[multi-instance] Solving meta-tour (n={total_nodes}) with Lin-Kernighan")

    t0 = time.time()
    improved_cycle, improved_dist = solve_tsp_lin_kernighan(
        meta_distance,
        x0=list(range(total_nodes)),
        verbose=verbose,
    )
    t1 = time.time()

    if verbose:
        print(f"[multi-instance] Meta-tour optimization: {improved_dist:.2f} in {t1 - t0:.3f}s")

    # Convert back to (instance_idx, original_node_id)
    final_tour = []
    for meta_idx in improved_cycle:
        inst_idx, local_node = global_tour[meta_idx]
        original_node_id = instances[inst_idx][2][local_node]
        final_tour.append((inst_idx, original_node_id))

    total_time = time.time() - start_time

    results = {
        'final_tour': final_tour,
        'final_distance': improved_dist,
        'instance_solutions': instance_solutions,
        'inter_instance_distances': inter_inst_dist.tolist(),
        'total_time': total_time,
        'num_instances': len(instances),
        'max_cluster_size': max_cluster_size,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("[multi-instance] FINAL RESULTS")
        print("=" * 60)
        print(f"Final distance: {improved_dist:.2f}")
        print(f"Total time: {total_time:.3f}s")
        print("\nPer-instance results:")
        for sol in instance_solutions:
            print(f"  {sol['filename']}: {sol['solver']} -> {sol['distance']:.2f}")

    return final_tour, improved_dist, results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Programming + Lin-Kernighan Hybrid TSP Solver")
    parser.add_argument("--instances", default="instances", help="Instances directory")
    parser.add_argument("--max-cluster", type=int, default=18, help="Max cluster size for DP")
    parser.add_argument("--no-kmeans", action="store_true", help="Disable k-means clustering")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--multi", action="store_true", help="Run multi-instance mode instead of single-file mode")
    args = parser.parse_args()

    if args.multi:
        final_tour, dist, results = solve_multi_instance_tsp(
            instances_dir=args.instances,
            max_cluster_size=args.max_cluster,
            use_kmeans=not args.no_kmeans,
            verbose=args.verbose,
        )
        print("\n" + "=" * 60)
        print("FINAL SOLUTION (multi-instance)")
        print("=" * 60)
        print(f"Total distance: {dist:.2f}")
        print(f"Total computation time: {results['total_time']:.3f}s")
    else:
        instances_dir = args.instances
        tsp_files = [f for f in os.listdir(instances_dir) if f.endswith('.tsp')]
        if not tsp_files:
            print(f"No TSP files found in {instances_dir}")
            exit(1)

        filepath = os.path.join(instances_dir, tsp_files[0])
        # BUGFIX: a versão anterior chamava solve_tsp_combined diretamente e depois
        # tentava acessar results['total_time'], mas solve_tsp_combined retorna
        # (cycle, dist, fixed_nodes) — não um dict. Usamos solve_single_tsp, que
        # já decide entre DP puro / Combined e devolve um dict de resultados.
        tour, dist, results = solve_single_tsp(
            filepath,
            max_cluster_size=args.max_cluster,
            use_kmeans=not args.no_kmeans,
            verbose=args.verbose,
        )

        print("\n" + "=" * 60)
        print("FINAL SOLUTION")
        print("=" * 60)
        print(f"Solver used: {results['solver']}")
        print(f"Optimal tour: {tour}")
        print(f"Total distance: {dist:.2f}")
        print(f"Total computation time: {results['time']:.3f}s")