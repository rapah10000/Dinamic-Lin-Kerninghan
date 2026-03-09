from basic_solvers.Lin_kerninghan import solve_tsp_lin_kernighan,_minimizes_hamiltonian_path_distance,_cycle_to_successors,_successors_to_cycle,_print_message
from basic_solvers.permutation_distance import compute_permutation_distance
import numpy as np
from typing import List, Optional, TextIO, Tuple
from basic_solvers.Bruteforce import solve_tsp_brute_force
from basic_solvers.initial_solution import setup_initial_solution

def solve_tsp_lin_kernighan(
    distance_matrix: np.ndarray,
    x0: Optional[List[int]] = None,
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[int], float]:
    num_vertices = distance_matrix.shape[0]
    if num_vertices < 4:
        return solve_tsp_brute_force(distance_matrix, log_file, verbose)

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
