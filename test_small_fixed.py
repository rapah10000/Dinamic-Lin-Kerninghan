"""Quick test with small instances separately"""
import os
import shutil
from DinamicLinKerninghan import solve_tsp_combined, _load_tsp_file

# Create temp directory
os.makedirs('instances_test', exist_ok=True)

# Copy small instances
for name in ['burma14.tsp','wi29.tsp', 'dj38.tsp', 'qa194.tsp', 'uy734.tsp', 'lu980.tsp']:
    src = f'instances/{name}'
    if os.path.exists(src):
        shutil.copy(src, 'instances_test/')

# Test each instance separately
print("Testing burma14.tsp...")
dm_burma14, _, _ = _load_tsp_file('instances_test/burma14.tsp')
tour_burma14, dist_burma14, _ = solve_tsp_combined(dm_burma14, verbose=False)
print(f"burma14.tsp: distance = {dist_burma14:.2f}")

print("Testing wi29.tsp...")
dm_wi29, _, _ = _load_tsp_file('instances_test/wi29.tsp')
tour_wi29, dist_wi29, _ = solve_tsp_combined(dm_wi29, verbose=False)
print(f"wi29.tsp: distance = {dist_wi29:.2f}")


print("Testing dj38.tsp...")
dm_dj38, _, _ = _load_tsp_file('instances_test/dj38.tsp')
tour_dj38, dist_dj38, _ = solve_tsp_combined(dm_dj38, verbose=False)
print(f"dj38.tsp: distance = {dist_dj38:.2f}")

print("Testing qa194.tsp...")
dm_qa194, _, _ = _load_tsp_file('instances_test/qa194.tsp')
tour_qa194, dist_qa194, _ = solve_tsp_combined(dm_qa194, verbose=False)
print(f"qa194.tsp: distance = {dist_qa194:.2f}")

#print("Testing uy734.tsp...")
#dm_uy734, _, _ = _load_tsp_file('instances_test/uy734.tsp')
#tour_uy734, dist_uy734, _ = solve_tsp_combined(dm_uy734, verbose=False)
#
#print("Testing lu980.tsp...")
#dm_lu980, _, _ = _load_tsp_file('instances_test/lu980.tsp')
#tour_lu980, dist_lu980, _ = solve_tsp_combined(dm_lu980, verbose=False)

print(f"\n=== RESULTS ===")
print(f"wi29.tsp: distance = {dist_wi29:.2f}")
print(f"dj38.tsp: distance = {dist_dj38:.2f}")
print(f"qa194.tsp: distance = {dist_qa194:.2f}")
#print(f"uy734.tsp: distance = {dist_uy734:.2f}")
#print(f"lu980.tsp: distance = {dist_lu980:.2f}")
print("\nTest completed successfully! Instances run separately.")

# Cleanup
shutil.rmtree('instances_test')

