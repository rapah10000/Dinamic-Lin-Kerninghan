"""Quick test with small instances only"""
import os
import shutil

# Create temp directory for small instances only
os.makedirs('instances_test', exist_ok=True)

# Copy only small instances
for name in ['wi29.tsp', 'dj38.tsp']:
    src = f'instances/{name}'
    if os.path.exists(src):
        shutil.copy(src, 'instances_test/')

'''# Temporarily rename large files in main directory
if os.path.exists('instances/lu980.tsp'):
    os.rename('instances/lu980.tsp', 'instances/lu980.tsp.bak')
if os.path.exists('instances/qa194.tsp'):
    os.rename('instances/qa194.tsp', 'instances/qa194.tsp.bak')
if os.path.exists('instances/uy734.tsp'):
    os.rename('instances/uy734.tsp', 'instances/uy734.tsp.bak')'''

# Run test
from DinamicLinKerninghan import solve_tsp_combined, _load_tsp_file

print("Testing wi29.tsp...")
dm_wi29, _, _ = _load_tsp_file('instances_test/wi29.tsp')
tour_wi29, dist_wi29, _ = solve_tsp_combined(dm_wi29, verbose=False)

print("Testing dj38.tsp...")
dm_dj38, _, _ = _load_tsp_file('instances_test/dj38.tsp')
tour_dj38, dist_dj38, _ = solve_tsp_combined(dm_dj38, verbose=False)

print(f"wi29: distance = {dist_wi29:.2f}")
print(f"dj38: distance = {dist_dj38:.2f}")


print("\n=== FINAL RESULT ===")
print("Final Tour:", tour)
print("Total Distance:", dist)
print("Total Time:", results['total_time'], "seconds")
print("Number of Instances:", results['num_instances'])

# Print per-instance results
print("\nPer-instance breakdown:")
for sol in results['instance_solutions']:
    print(f"  {sol['filename']}: {sol['solver']} -> distance={sol['distance']:.2f}, time={sol['time']:.3f}s")

# Restore large files
'''if os.path.exists('instances/lu980.tsp.bak'):
    os.rename('instances/lu980.tsp.bak', 'instances/lu980.tsp')
if os.path.exists('instances/qa194.tsp.bak'):
    os.rename('instances/qa194.tsp.bak', 'instances/qa194.tsp')
if os.path.exists('instances/uy734.tsp.bak'):
    os.rename('instances/uy734.tsp.bak', 'instances/uy734.tsp')'''

# Cleanup
shutil.rmtree('instances_test')
print("\nTest completed successfully!")

