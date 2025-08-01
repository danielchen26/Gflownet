# Test Directory Structure

The test suite is organized hierarchically to reflect the different aspects of the GFlowNet.jl package.

## Directory Organization

### `core/`
Core functionality tests for fundamental GFlowNet components.

- **`test_core_functions.jl`** - Basic GFlowNet functions
- **`test_core_interface.jl`** - High-level interface functions
- **`test_utilities.jl`** - Utility functions

#### `core/detailed_balance/`
Tests for the detailed balance mathematical foundation and implementation.
- **`test_detailed_balance.jl`** - Basic detailed balance loss computation
- **`test_detailed_balance_comprehensive.jl`** - Comprehensive verification
- **`test_detailed_balance_summary.jl`** - Summary tests and validation

#### `core/flow_computation/`
Tests for flow computation and conservation.
- **`test_flow_functions.jl`** - Flow computation, caching, and conservation

#### `core/policies/`
Tests for forward and backward policies.
- **`test_backward_policy.jl`** - Backward policy with joint representation

#### `core/neural_networks/`
Tests for neural network components.
- **`test_neural_networks.jl`** - Lux network creation and functionality

### `objectives/`
Tests for different training objectives.

#### `objectives/trajectory_balance/`
Tests for trajectory balance objective (currently empty - TB tests are in integration).

#### `objectives/detailed_balance/`
- **`test_training.jl`** - DETAILED_BALANCE training tests

#### `objectives/learnable_z/`
Tests for learnable partition function.
- **`test_learnable_z.jl`** - Basic learnable Z functionality
- **`test_perfect_z_learning.jl`** - Perfect Z learning verification

### `applications/`
Tests for specific domain applications.

#### `applications/grid_world/`
- **`test_grid_world.jl`** - Grid world implementation
- **`test_grid_world_versions.jl`** - Comparison of different implementations

#### `applications/supply_chain/`
- **`test_supply_chain.jl`** - Supply chain optimization application

### `integration/`
Integration tests that combine multiple components.
- **`test_training.jl`** - End-to-end training tests

### `debugging/`
Diagnostic and debugging tests.

#### `debugging/zygote_issues/`
Tests related to Zygote compatibility and gradient computation.
- **`test_mutation_trace.jl`** - Traces mutation issues
- **`test_zygote_compatibility.jl`** - Zygote compatibility checks

#### `debugging/diagnostics/`
- **`test_detailed_balance_debug.jl`** - Detailed balance debugging
- **`test_feature_status.jl`** - Working vs broken feature comparison

## Running Tests

To run all tests:
```bash
julia --project=. test/runtests.jl
```

To run specific test categories:
```bash
# Run core tests
julia --project=. test/core/test_core_functions.jl

# Run detailed balance tests
julia --project=. test/core/detailed_balance/test_detailed_balance.jl

# Run application tests
julia --project=. test/applications/grid_world/test_grid_world.jl
```