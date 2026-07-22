"""
Quick test for TSPLIB instances.
"""

import os
import shutil
import time

from DinamicLinKerninghan import solve_tsp_combined, _load_tsp_file


# ============================================================
# Known optimal values (TSPLIB)
# ============================================================

OPTIMAL = {
    "burma14.tsp": 3323,
    "wi29.tsp": 27603,
    "dj38.tsp": 6656,
    "qa194.tsp": 9352,
    "uy734.tsp": 79114,
    "lu980.tsp": 11340,
}


# ============================================================
# Instances to test
# ============================================================

INSTANCES = [
    "burma14.tsp",
    "wi29.tsp",
    "dj38.tsp",
    "qa194.tsp",
    #"uy734.tsp",
    # "lu980.tsp",
]


# ============================================================
# Validate solution
# ============================================================

def validate_solution(distance_matrix, tour, reported_cost):
    """
    Validate a TSP solution.
    """

    n = len(distance_matrix)

    print("\nValidation")

    # Check repeated cities
    if len(set(tour)) != n:
        print("❌ Repeated cities detected.")
    else:
        print("✅ No repeated cities.")

    # Check number of cities
    if len(tour) != n:
        print(
            f"❌ Tour contains {len(tour)} cities, expected {n}."
        )
    else:
        print("✅ Correct number of cities.")

    # Compute tour cost
    cost = 0.0

    for i in range(n - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]

    cost += distance_matrix[tour[-1], tour[0]]

    print(f"Reported cost : {reported_cost:.2f}")
    print(f"Computed cost : {cost:.2f}")

    if abs(cost - reported_cost) < 1e-6:
        print("✅ Cost validated.")
    else:
        print("❌ Cost mismatch!")

    return cost


# ============================================================
# Prepare temporary folder
# ============================================================

os.makedirs("instances_test", exist_ok=True)

for instance in INSTANCES:

    src = os.path.join("instances", instance)

    if os.path.exists(src):
        shutil.copy(src, "instances_test")


# ============================================================
# Execute tests
# ============================================================

results = []

for instance in INSTANCES:

    print("\n")
    print("=" * 80)
    print(f"Testing {instance}")
    print("=" * 80)

    distance_matrix, nodes, metadata = _load_tsp_file(
        f"instances_test/{instance}"
    )

    print(f"Dimension        : {metadata['dimension']}")
    print(f"Edge weight type : {metadata['edge_weight_type']}")

    start = time.perf_counter()

    tour, distance, _ = solve_tsp_combined(
        distance_matrix,
        verbose=False,
    )

    elapsed = time.perf_counter() - start

    print(f"\nDistance : {distance:.2f}")
    print(f"Time     : {elapsed:.4f} s")
    print(f"Tour     : {tour}")

    computed_cost = validate_solution(
        distance_matrix,
        tour,
        distance,
    )

    optimal = OPTIMAL.get(instance)

    if optimal is not None:

        gap = (
            100.0 * (distance - optimal)
        ) / optimal

        print(f"\nOptimal cost : {optimal}")
        print(f"GAP          : {gap:.4f}%")

    results.append(
        (
            instance,
            metadata["dimension"],
            distance,
            elapsed,
            optimal,
        )
    )


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 100)
print("FINAL RESULTS")
print("=" * 100)

header = (
    f"{'Instance':15}"
    f"{'Cities':>10}"
    f"{'Distance':>15}"
    f"{'Optimal':>15}"
    f"{'Gap (%)':>12}"
    f"{'Time (s)':>12}"
)

print(header)
print("-" * len(header))

for instance, n, dist, t, optimal in results:

    if optimal is None:
        gap_str = "-"
        optimal_str = "-"
    else:
        gap = 100 * (dist - optimal) / optimal
        gap_str = f"{gap:.4f}"
        optimal_str = f"{optimal}"

    print(
        f"{instance:15}"
        f"{n:10d}"
        f"{dist:15.2f}"
        f"{optimal_str:>15}"
        f"{gap_str:>12}"
        f"{t:12.4f}"
    )

print("=" * 100)

print("\nAll tests completed successfully!")

# ============================================================
# Cleanup
# ============================================================

shutil.rmtree("instances_test")