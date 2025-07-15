# GFNDSS

### Code modules

- `GFNs.py` – core GFlowNet logic and trainer implementation.
- `gfn_async.py` – multiprocessing helpers for asynchronous training.
- `cost_computer.py` – collection of cost functions and the `CostComputer` helper.
- `energy_estimator.py` – GPU-accelerated energy estimation using Clifford maps.
- `gfn_objectives.py` – modular training objectives for the GFlowNet.
- `gf2_ops.py` – GF(2) linear algebra routines used by the Clifford tableau code.
- `models.py` – neural network architectures for the GFlowNet policy and sampler.
- `clifford_map.py` – vectorized stabilizer tableau simulation.
- `pauli_hamiltonian_helper.py` – parser and utilities for Pauli Hamiltonian files.
- `quantum_action_mapping.py` – mapping from action indices to quantum gate tuples.
- `run_exp.py` – command-line interface for launching experiments from JSON configs.
- `main.py` – high level experiment runner combining training and energy estimation.
