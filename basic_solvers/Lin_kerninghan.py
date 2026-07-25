import random


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
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]


def double_bridge(tour, allowed_positions=None, rng=None):
    r = rng if rng is not None else random
    n = len(tour)
    if n < 8:
        return tour[:]
    if allowed_positions is not None:
        candidates = sorted({p for p in allowed_positions if 1 <= p <= n - 1})
        if len(candidates) < 3:
            return tour[:]
        a, b, c = sorted(r.sample(candidates, 3))
    else:
        a, b, c = sorted(r.sample(range(1, n), 3))
    return tour[:a] + tour[c:] + tour[b:c] + tour[a:b]


def solve_tsp_lin_kernighan(
    distance_matrix,
    x0=None,
    candidates=None,
    fixed_nodes=None,
    node_to_cluster=None,
    joint_indices=None,
    joint_window=3,
    max_no_improve=50,
    max_perturbations=100,
    seed=None,
    verbose=False,
    log_file=None,
):
    n = len(distance_matrix)
    dist = distance_matrix
    rng = random.Random(seed) if seed is not None else random

    if n <= 3:
        tour = x0[:] if x0 is not None else list(range(n))
        return tour, compute_tour_cost(tour, dist)

    if x0 is None:
        tour = list(range(n))
        rng.shuffle(tour)
    else:
        tour = x0[:]

    if candidates is None:
        k = min(15, n - 1)
        candidates = build_candidate_sets(distance_matrix, k)

    fixed_set = set(fixed_nodes) if fixed_nodes else set()

    position = [0] * n
    for idx, node in enumerate(tour):
        position[node] = idx

    current_cost = compute_tour_cost(tour, dist)

    def reverse_segment(lo, hi):
        i, j = lo + 1, hi
        while i < j:
            tour[i], tour[j] = tour[j], tour[i]
            position[tour[i]] = i
            position[tour[j]] = j
            i += 1
            j -= 1
        if i == j:
            position[tour[i]] = i

    def move_is_protected(t1, t1_next, t2, t2_next, lo, hi):
        if t1 in fixed_set or t1_next in fixed_set or t2 in fixed_set or t2_next in fixed_set:
            return True
        if node_to_cluster is not None:
            segment_clusters = {node_to_cluster[node] for node in tour[lo + 1:hi + 1]}
            if len(segment_clusters) <= 1:
                return True
        return False

    def try_improve_from(t1_idx):
        nonlocal current_cost
        t1 = tour[t1_idx]
        if t1 in fixed_set:
            return False, None
        i = t1_idx
        t1_next = tour[(i + 1) % n]
        d_removed_1 = dist[t1][t1_next]
        for t2 in candidates[t1]:
            d_new_1 = dist[t1][t2]
            if d_new_1 >= d_removed_1:
                break
            j = position[t2]
            if j == i or j == (i + 1) % n:
                continue
            t2_next = tour[(j + 1) % n]
            if t2_next == t1:
                continue
            delta = (
                d_new_1
                + dist[t1_next][t2_next]
                - d_removed_1
                - dist[t2][t2_next]
            )
            if delta < -1e-9:
                lo, hi = (i, j) if i < j else (j, i)
                if move_is_protected(t1, t1_next, t2, t2_next, lo, hi):
                    continue
                reverse_segment(lo, hi)
                current_cost += delta
                return True, {t1, t1_next, t2, t2_next}
        return False, None

    def try_or_opt_from(t1_idx, block_size=1):
        nonlocal current_cost
        if block_size != 1:
            raise NotImplementedError("Somente Or-opt(1) implementado.")

        a_idx = t1_idx
        a = tour[a_idx]

        if a in fixed_set:
            return False, None

        prev_idx = (a_idx - 1) % n
        next_idx = (a_idx + 1) % n
        prev_node = tour[prev_idx]
        next_node = tour[next_idx]

        if (
            prev_node in fixed_set
            or next_node in fixed_set
        ):
            return False, None

        for target in candidates[a]:
            insert_idx = position[target]
            if (
                insert_idx == a_idx
                or insert_idx == prev_idx
                or insert_idx == next_idx
            ):
                continue
            after_idx = (insert_idx + 1) % n
            after_node = tour[after_idx]
            if after_node == a:
                continue

            # BUGFIX: faltava proteger o PONTO DE INSERÇÃO. Sem isso, o nó `a`
            # podia ser inserido bem no meio de uma aresta já otimizada pelo
            # DP (target -> after_node), quebrando exatamente a garantia que
            # fixed_nodes deveria dar.
            if target in fixed_set or after_node in fixed_set:
                continue

            delta = (
                - dist[prev_node][a]
                - dist[a][next_node]
                - dist[target][after_node]
                + dist[prev_node][next_node]
                + dist[target][a]
                + dist[a][after_node]
            )
            if delta >= -1e-9:
                continue

            node = tour.pop(a_idx)
            if insert_idx > a_idx:
                insert_idx -= 1
            tour.insert(insert_idx + 1, node)
            for idx, node in enumerate(tour):
                position[node] = idx
            current_cost += delta
            return True, {prev_node, a, next_node, target, after_node}
        return False, None

    def try_or_opt2_from(a_idx):
        nonlocal current_cost
        if a_idx >= n - 1:
            return False, None
        a = tour[a_idx]
        b = tour[a_idx + 1]
        # BUGFIX: faltava checar `b` (o segundo nó do bloco) contra
        # fixed_set — só `a` era verificado, então um nó fixo podia ser
        # realocado junto no bloco sem nenhuma proteção.
        if a in fixed_set or b in fixed_set:
            return False, None
        prev_idx = (a_idx - 1) % n
        next_idx = (a_idx + 2) % n
        prev_node = tour[prev_idx]
        next_node = tour[next_idx]
        if (
            prev_node in fixed_set
            or next_node in fixed_set
        ):
            return False, None

        for target in candidates[a]:
            insert_idx = position[target]
            if (
                insert_idx >= a_idx + 1
                and insert_idx <= a_idx + 1
            ):
                continue
            after_idx = (insert_idx + 1) % n
            after_node = tour[after_idx]
            if after_node in (a, b):
                continue

            # BUGFIX: mesma falha do Or-opt(1) — o ponto de inserção
            # (target -> after_node) nunca era checado contra fixed_set.
            if target in fixed_set or after_node in fixed_set:
                continue

            delta = (
                - dist[prev_node][a]
                - dist[b][next_node]
                - dist[target][after_node]
                + dist[prev_node][next_node]
                + dist[target][a]
                + dist[b][after_node]
            )
            if delta >= -1e-9:
                continue

            block = tour[a_idx:a_idx + 2]
            del tour[a_idx:a_idx + 2]
            if insert_idx > a_idx:
                insert_idx -= 2
            tour[insert_idx + 1:insert_idx + 1] = block
            for idx, node in enumerate(tour):
                position[node] = idx
            current_cost += delta
            return True, {prev_node, a, b, next_node, target, after_node}
        return False, None

    def local_search():
        nonlocal current_cost
        dont_look = [False] * n
        for node in fixed_set:
            dont_look[node] = True
        active = [i for i in range(n) if not dont_look[tour[i]]]
        no_improve_streak = 0
        while active and no_improve_streak < max_no_improve:
            improved_any = False
            next_active = []
            for t1_idx in active:
                t1 = tour[t1_idx]
                if dont_look[t1]:
                    continue
                result, touched_nodes = try_improve_from(position[t1])
                if not result:
                    result, touched_nodes = try_or_opt_from(position[t1], block_size=1)
                if not result:
                    result, touched_nodes = try_or_opt2_from(position[t1])
                if result:
                    improved_any = True
                    for node in touched_nodes:
                        dont_look[node] = False
                    next_active.extend(position[node] for node in touched_nodes)
                else:
                    dont_look[t1] = True
            active = list(set(next_active)) if improved_any else []
            no_improve_streak = 0 if improved_any else no_improve_streak + 1
            if not improved_any:
                break

    def bidirectional_local_search():
        nonlocal tour, current_cost
        local_search()
        forward_tour = tour[:]
        forward_cost = current_cost
        reverse_tour = [forward_tour[0], *reversed(forward_tour[1:])]
        tour[:] = reverse_tour
        for idx, node in enumerate(tour):
            position[node] = idx
        current_cost = compute_tour_cost(tour, dist)
        local_search()
        reverse_cost = current_cost
        if reverse_cost < forward_cost:
            return tour[:], reverse_cost
        tour[:] = forward_tour
        for idx, node in enumerate(tour):
            position[node] = idx
        current_cost = forward_cost
        return forward_tour, forward_cost

    best_tour, best_cost = bidirectional_local_search()
    tour[:] = best_tour
    for idx, node in enumerate(tour):
        position[node] = idx
    current_cost = best_cost

    allowed_positions = None
    enable_perturbation = False
    if not fixed_set:
        enable_perturbation = n >= 8
    elif joint_indices:
        allowed_positions = set()
        for idx in joint_indices:
            for delta in range(-joint_window, joint_window + 1):
                allowed_positions.add((idx + delta) % n)
        enable_perturbation = n >= 8

    if enable_perturbation:
        no_improve_perturbations = 0
        while no_improve_perturbations < max_perturbations:
            candidate_tour = double_bridge(best_tour, allowed_positions=allowed_positions, rng=rng)
            tour[:] = candidate_tour
            for idx, node in enumerate(tour):
                position[node] = idx
            current_cost = compute_tour_cost(tour, dist)
            candidate_best_tour, candidate_best_cost = bidirectional_local_search()
            tour[:] = candidate_best_tour
            for idx, node in enumerate(tour):
                position[node] = idx
            current_cost = candidate_best_cost
            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_tour = tour[:]
                no_improve_perturbations = 0
            else:
                no_improve_perturbations += 1

    if verbose:
        print(f"[LK FINAL] Cost: {best_cost:.2f}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"[LK FINAL] Cost: {best_cost:.2f}\n")

    return best_tour, best_cost