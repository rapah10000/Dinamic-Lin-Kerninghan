"""Quick test with small instances separately"""
import os
import shutil
from DinamicLinKerninghan import solve_tsp_combined, _load_tsp_file

# Create temp directory
os.makedirs('instances_test', exist_ok=True)

# Copy small instances
for name in ['wi29.tsp', 'dj38.tsp']:
    src = f'instances/{name}'
    if os.path.exists(src):
        shutil.copy(src, 'instances_test/')

# Test each instance separately
print("Testing wi29.tsp...")
dm_wi29, _, _ = _load_tsp_file('instances_test/wi29.tsp')
tour_wi29, dist_wi29, _ = solve_tsp_combined(dm_wi29, verbose=False)

print("Testing dj38.tsp...")
dm_dj38, _, _ = _load_tsp_file('instances_test/dj38.tsp')
tour_dj38, dist_dj38, _ = solve_tsp_combined(dm_dj38, verbose=False)

print(f"\n=== RESULTS ===")
print(f"wi29.tsp: distance = {dist_wi29:.2f}")
print(f"dj38.tsp: distance = {dist_dj38:.2f}")
print("\nTest completed successfully! Instances run separately.")

# Cleanup
shutil.rmtree('instances_test')

