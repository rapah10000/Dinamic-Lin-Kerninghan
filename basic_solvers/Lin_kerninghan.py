import random
import time


def compute_tour_cost(tour, dist):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += dist[tour[i]][tour[(i + 1) % n]]
    return cost


def build_candidate_sets(distance_matrix, k=15):
    n = len(distance_matrix)
    candidates = []

    for i in range(n):
        nearest = sorted(range(n), key=lambda j: distance_matrix[i][j])
        nearest.remove(i)
        candidates.append(nearest[:k])

    return candidates


def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]


def double_bridge(tour):
    n = len(tour)
    a, b, c, d = sorted(random.sample(range(n), 4))
    return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]


def solve_tsp_lin_kernighan(
    distance_matrix,
    x0=None,
    candidates=None,
    fixed_nodes=None,
    node_to_cluster=None,
    max_no_improve=50,
    verbose=False
):
    import random

    n = len(distance_matrix)

    # Initial solution
    if x0 is None:
        tour = list(range(n))
        random.shuffle(tour)
    else:
        tour = x0[:]

    def compute_cost(tour):
        return sum(distance_matrix[tour[i]][tour[(i+1) % n]] for i in range(n))

    # Candidate sets
    if candidates is None:
        k = min(15, n-1)
        candidates = []
        for i in range(n):
            nearest = sorted(range(n), key=lambda j: distance_matrix[i][j])
            nearest.remove(i)
            candidates.append(nearest[:k])

    # Position array (🔥 performance)
    position = [0] * n
    for i in range(n):
        position[tour[i]] = i

    best_tour = tour[:]
    best_cost = compute_cost(tour)

    dont_look = [False] * n
    no_improve_count = 0

    while no_improve_count < max_no_improve:
        improvement_found = False

        for t1_idx in range(n):
            if dont_look[t1_idx]:
                continue

            t1 = tour[t1_idx]

            neighbors = candidates[t1] if candidates else range(n)

            for t2 in neighbors:
                if t2 == t1:
                    continue

                t2_idx = position[t2]
                i, j = sorted([t1_idx, t2_idx])

                # 🔥 PROTEÇÃO COMPLETA DE CLUSTERS
                if node_to_cluster:
                    segment = tour[i:j+1]
                    clusters_in_segment = {node_to_cluster[node] for node in segment}

                    # evita mexer dentro de um único cluster
                    if len(clusters_in_segment) == 1:
                        continue

                # 2-opt
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_cost = compute_cost(new_tour)

                if new_cost < best_cost:
                    tour = new_tour
                    best_tour = new_tour[:]
                    best_cost = new_cost

                    # 🔥 atualizar posição SEMPRE
                    for k in range(n):
                        position[tour[k]] = k

                    dont_look[t1_idx] = False
                    improvement_found = True
                    break

            # 🔥 fallback (exploração completa)
            if not improvement_found:
                for t2 in range(n):
                    if t2 == t1:
                        continue

                    t2_idx = position[t2]
                    i, j = sorted([t1_idx, t2_idx])

                    if node_to_cluster:
                        segment = tour[i:j+1]
                        clusters_in_segment = {node_to_cluster[node] for node in segment}

                        if len(clusters_in_segment) == 1:
                            continue

                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_cost = compute_cost(new_tour)

                    if new_cost < best_cost:
                        tour = new_tour
                        best_tour = new_tour[:]
                        best_cost = new_cost

                        for k in range(n):
                            position[tour[k]] = k

                        dont_look[t1_idx] = False
                        improvement_found = True
                        break

            if not improvement_found:
                dont_look[t1_idx] = True
            else:
                break

        if improvement_found:
            no_improve_count = 0
        else:
            no_improve_count += 1

    if verbose:
        print(f"[LK FINAL] Cost: {best_cost:.2f}")

    return best_tour, best_cost