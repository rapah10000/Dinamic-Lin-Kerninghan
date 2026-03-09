from typing import List

import numpy as np


def compute_permutation_distance(
    distance_matrix: np.ndarray, permutation: List[int]
) -> float:
    ind1 = permutation
    ind2 = permutation[1:] + permutation[:1]
    return distance_matrix[ind1, ind2].sum()
