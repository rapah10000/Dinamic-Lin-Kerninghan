# TODO: Modify to Run Instances Separately

**Goal**: Disable multi-instance connection, run each TSP file independently.

**Steps**:
- [ ] 1. Comment out `solve_multi_instance_tsp()` function in DinamicLinKerninghan.py
- [ ] 2. Add `solve_single_tsp()` wrapper using `solve_tsp_combined()`
- [ ] 3. Update `test_small.py` to test single instances separately
- [ ] 4. Update `__main__` to solve first instance or take filename arg
- [ ] 5. Test execution
- [ ] 6. Complete

**Current Status**: Multi-instance disabled, single-instance ready.

