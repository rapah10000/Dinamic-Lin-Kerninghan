# TODO: Modify to Run Instances Separately

**Goal**: Disable multi-instance connection, run each TSP file independently.

**Steps**:
- [x] 1. Comment out `solve_multi_instance_tsp()` function in DinamicLinKerninghan.py
- [x] 2. Add `solve_single_tsp()` wrapper using `solve_tsp_combined()`
- [x] 3. Update `test_small.py` to test single instances separately
- [x] 4. Update `__main__` to solve first instance or take filename arg
- [x] 5. Test execution
- [x] 6. Complete

**Current Status**: Multi-instance ready, single-instance ready.

