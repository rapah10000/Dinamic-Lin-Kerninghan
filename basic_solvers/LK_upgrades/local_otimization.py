from typing import List
import numpy as np


def compute_tour_cost(
    tour: List[int],
    distance_matrix: np.ndarray,
) -> float:
    """
    Compute the total cost of a Hamiltonian cycle.

    Parameters
    ----------
    tour
        Tour represented by a list of city indices.

    distance_matrix
        Distance matrix.

    Returns
    -------
    float
        Total tour length.
    """

    n = len(tour)

    if n == 0:
        return 0.0

    cost = 0.0

    for i in range(n - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]

    cost += distance_matrix[tour[-1], tour[0]]

    return float(cost)


def two_opt_gain(
    tour: List[int],
    i: int,
    j: int,
    distance_matrix: np.ndarray,
) -> float:
    """
    Compute the gain obtained by performing a 2-opt move.

    Positive value -> improvement.

    Parameters
    ----------
    tour
        Current tour.

    i, j
        Indices of the segment to reverse.

    distance_matrix
        Distance matrix.

    Returns
    -------
    float
        Gain produced by the move.
    """

    n = len(tour)

    a = tour[i - 1]
    b = tour[i]

    c = tour[j]
    d = tour[(j + 1) % n]

    old_cost = (
        distance_matrix[a, b]
        + distance_matrix[c, d]
    )

    new_cost = (
        distance_matrix[a, c]
        + distance_matrix[b, d]
    )

    return old_cost - new_cost



def apply_two_opt(
    tour: List[int],
    i: int,
    j: int,
) -> List[int]:
    """
    Apply a 2-opt move.

    Parameters
    ----------
    tour
        Current tour.

    i, j
        Segment to reverse.

    Returns
    -------
    List[int]
        New tour.
    """

    new_tour = tour.copy()

    new_tour[i : j + 1] = reversed(
        new_tour[i : j + 1]
    )

    return new_tour


from typing import List, Tuple
import numpy as np


def two_opt_best(
    tour: List[int],
    distance_matrix: np.ndarray,
) -> Tuple[List[int], float]:
    """
    Best-improvement 2-opt local search.

    Parameters
    ----------
    tour
        Initial solution.

    distance_matrix
        Distance matrix.

    Returns
    -------
    Tuple
        Improved tour and its cost.
    """

    best_tour = tour.copy()

    best_cost = compute_tour_cost(
        best_tour,
        distance_matrix,
    )

    n = len(best_tour)

    improved = True

    while improved:

        improved = False

        best_gain = 0.0
        best_move = None

        for i in range(1, n - 2):

            for j in range(i + 1, n - 1):

                gain = two_opt_gain(
                    best_tour,
                    i,
                    j,
                    distance_matrix,
                )

                if gain > best_gain:

                    best_gain = gain
                    best_move = (i, j)

        if best_move is not None:

            i, j = best_move

            best_tour = apply_two_opt(
                best_tour,
                i,
                j,
            )

            best_cost -= best_gain

            improved = True

    return best_tour, best_cost