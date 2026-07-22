import numpy as np
from typing import List, Optional, Tuple
from random import sample
from basic_solvers.permutation_distance import compute_permutation_distance

def setup_initial_solution(
    distance_matrix: np.ndarray, x0: Optional[List] = None
) -> Tuple[List[int], float]:
    if not x0:
        n = distance_matrix.shape[0]  # number of nodes
        x0 = [0] + sample(range(1, n), n - 1)  # ensure 0 is the first node

    fx0 = compute_permutation_distance(distance_matrix, x0)
    return x0, fx0