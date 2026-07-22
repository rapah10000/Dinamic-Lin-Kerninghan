import random


def compute_tour_cost(tour, dist):
    cost = 0
    n = len(tour)
    for i in range(n):
        cost += dist[tour[i]][tour[(i + 1) % n]]
    return cost


def build_candidate_sets(distance_matrix, k=15):
    """
    Para cada nó, retorna seus k vizinhos mais próximos, ordenados por
    distância crescente. A ordenação é essencial: o loop principal usa isso
    para podar a busca antecipadamente (ver `_improve_from_node`).
    """
    n = len(distance_matrix)
    candidates = []

    for i in range(n):
        nearest = sorted(range(n), key=lambda j: distance_matrix[i][j])
        nearest.remove(i)
        candidates.append(nearest[:k])

    return candidates


def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]


def double_bridge(tour, allowed_positions=None):
    """
    Perturbação 4-opt clássica usada em Iterated Local Search.

    Parameters
    ----------
    tour
        Tour atual.
    allowed_positions
        Se fornecido, os 3 pontos de corte só são sorteados dentre essas
        posições (em vez de qualquer posição de 1 a n-1). Use isso para
        restringir a perturbação a regiões específicas do tour — por
        exemplo, apenas ao redor das junções entre clusters, preservando o
        interior de segmentos já otimizados (por DP, por exemplo) intactos.
        Se houver menos de 3 posições válidas disponíveis, a perturbação é
        pulada (retorna o tour inalterado).
    """
    n = len(tour)
    if n < 8:
        return tour[:]

    if allowed_positions is not None:
        candidates = sorted({p for p in allowed_positions if 1 <= p <= n - 1})
        if len(candidates) < 3:
            return tour[:]
        a, b, c = sorted(random.sample(candidates, 3))
    else:
        a, b, c = sorted(random.sample(range(1, n), 3))

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
    max_perturbations=30,
    verbose=False,
    log_file=None,
):
    """
    2-opt com listas de candidatos + Iterated Local Search (double bridge),
    respeitando nós/arestas protegidos (fixed_nodes / node_to_cluster).

    Parameters
    ----------
    distance_matrix
        Matriz de distâncias (n x n).
    x0
        Tour inicial (lista de índices). Se None, gera um tour aleatório.
    candidates
        Listas de candidatos por nó (ver build_candidate_sets). Se None, são
        calculadas aqui.
    fixed_nodes
        Nós cujas duas arestas já são conhecidas como ótimas (por exemplo,
        resolvidas exatamente por DP dentro de um cluster). Esses nós nunca
        iniciam uma troca, e movimentos que romperiam suas arestas são
        bloqueados.
    node_to_cluster
        Mapa nó -> id do cluster. Usado para impedir que uma troca reverta um
        segmento inteiramente contido em um único cluster (o que seria
        redundante, já que o DP já resolveu esse cluster de forma ótima).
    joint_indices
        Posições no tour inicial (`x0`) onde há uma junção entre clusters.
        Quando fornecido junto com `fixed_nodes`, habilita uma perturbação
        double-bridge restrita a essas posições (± `joint_window`), em vez de
        pular a perturbação inteiramente. Isso permite ao ILS reorganizar
        como os clusters se conectam sem nunca tocar o interior de um
        cluster já resolvido de forma ótima pelo DP.
    joint_window
        Quantas posições ao redor de cada junção ficam liberadas para os
        cortes do double-bridge restrito (só tem efeito quando
        `joint_indices` é fornecido).
    max_no_improve
        Número de iterações sem melhoria no loop de busca local (2-opt) antes
        de considerar convergido.
    max_perturbations
        Número de perturbações (double bridge) sem melhoria no ILS antes de
        parar. Só é usado quando `fixed_nodes` é None — na fase de junção de
        clusters, perturbar aleatoriamente poderia desfazer partes já ótimas
        do DP, então preferimos apenas a busca local determinística.
    verbose
        Imprime o progresso.
    """
    n = len(distance_matrix)
    dist = distance_matrix

    if n <= 3:
        tour = x0[:] if x0 is not None else list(range(n))
        return tour, compute_tour_cost(tour, dist)

    if x0 is None:
        tour = list(range(n))
        random.shuffle(tour)
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
        """Reverte tour[lo+1:hi+1] (inclusive) e atualiza o array de posições."""
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
        # Nunca tocar arestas incidentes a nós já otimizados pelo DP
        if t1 in fixed_set or t1_next in fixed_set or t2 in fixed_set or t2_next in fixed_set:
            return True
        # Nunca reverter um segmento inteiramente contido em um único cluster
        if node_to_cluster is not None:
            segment_clusters = {node_to_cluster[node] for node in tour[lo + 1:hi + 1]}
            if len(segment_clusters) <= 1:
                return True
        return False

    def try_improve_from(t1_idx):
        """
        Tenta melhorar quebrando a aresta (t1, sucessor(t1)) por uma troca
        2-opt com algum candidato de t1. Usa o delta de custo (O(1)) em vez
        de recomputar o tour inteiro, e poda a busca assim que a distância
        candidata deixa de compensar (pois `candidates[t1]` está ordenado).
        """
        nonlocal current_cost

        t1 = tour[t1_idx]
        if t1 in fixed_set:
            return False

        i = t1_idx
        t1_next = tour[(i + 1) % n]
        d_removed_1 = dist[t1][t1_next]

        for t2 in candidates[t1]:
            d_new_1 = dist[t1][t2]
            # Poda: se a nova aresta já é mais longa que a removida, nenhuma
            # troca subsequente (candidatos mais distantes) pode compensar.
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

    def local_search():
        """Roda 2-opt até convergir para um ótimo local, respeitando fixed_nodes."""
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

    # Fase 1: busca local até o ótimo (2-opt puro, respeitando fixed_nodes)
    local_search()

    best_tour = tour[:]
    best_cost = current_cost

    # Fase 2: Iterated Local Search com double bridge.
    #
    # - Sem nós protegidos: perturbação irrestrita (qualquer ponto de corte).
    # - Com nós protegidos (fixed_nodes) mas sem joint_indices: pula a
    #   perturbação inteiramente, por segurança (comportamento anterior).
    # - Com nós protegidos E joint_indices: perturbação restrita às
    #   vizinhanças das junções entre clusters, permitindo ao ILS explorar
    #   diferentes formas de conectar os clusters sem nunca desfazer o
    #   interior de um cluster já resolvido de forma ótima pelo DP.
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
            candidate_tour = double_bridge(best_tour, allowed_positions=allowed_positions)

            tour[:] = candidate_tour
            for idx, node in enumerate(tour):
                position[node] = idx
            current_cost = compute_tour_cost(tour, dist)

            local_search()

            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_tour = tour[:]
                no_improve_perturbations = 0
                if verbose:
                    print(f"[LK ILS] melhoria encontrada: {best_cost:.2f}")
            else:
                no_improve_perturbations += 1

    if verbose:
        print(f"[LK FINAL] Cost: {best_cost:.2f}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"[LK FINAL] Cost: {best_cost:.2f}\n")

    return best_tour, best_cost