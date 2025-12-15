# GFNDSS

GFlowNet-based Derandomized Shadow Sampling for quantum circuit optimization and energy estimation.

## Overview

This project implements a GFlowNet (Generative Flow Network) approach to learn optimal Clifford measurement circuits for quantum Hamiltonian energy estimation. The method combines:

- **GFlowNet Training**: Learns to sample shallow Clifford circuits that maximize Pauli operator coverage
- **Derandomized Shadow Sampling (DSS)**: Efficient energy estimation using learned measurement circuits
- **GPU-Accelerated Simulation**: Vectorized Clifford tableau operations for fast training

## Installation

### Requirements

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.5.0
```

For quantum hardware experiments (optional):
```
qiskit>=1.0.0
qiskit-ibm-runtime>=0.20.0
```

## Quick Start

### Running an Experiment

```bash
# From project root
python code/run_config.py --config path/to/config.json
```

### Example Configuration

```json
{
    "hamiltonian_path": "Hamiltonians/H2_STO3g_4qubits/jw.txt",
    "n_updates": 10000,
    "eval_every": 1000,
    "n_measurements": 1000,
    "max_depth": 8,
    "hidden_dim": 1024,
    "num_hidden_layers": 3,
    "lr": 1e-3,
    "model_type": "clifford_mlp",
    "cost_type": "exponential",
    "objective_type": "tb",
    "async_eval": true
}
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hamiltonian_path` | (required) | Path to Pauli Hamiltonian file |
| `n_updates` | 10000 | Total training updates |
| `eval_every` | 1000 | Evaluation frequency |
| `n_measurements` | 1000 | Circuits per batch element |
| `max_depth` | 8 | Maximum circuit depth |
| `hidden_dim` | 1024 | Neural network hidden dimension |
| `num_hidden_layers` | 3 | Number of hidden layers |
| `lr` | 1e-3 | Learning rate |
| `model_type` | "clifford_mlp" | Model architecture |
| `cost_type` | "exponential" | Cost function type |
| `objective_type` | "tb" | GFlowNet objective (tb, db, subtb, fl) |
| `async_eval` | false | Enable asynchronous evaluation |
| `resume` | true | Resume from checkpoint if available |

### Model Types
- `clifford_mlp`: MLP with Clifford tableau encoding
- `clifford_deepsets`: DeepSets architecture for permutation invariance
- `attention_mlp`: Attention-based architecture
- `quantum_aware_mlp`: Quantum-structure-aware MLP

### Cost Functions
- `exponential`: Exponential weight on Pauli costs
- `linear`: Linear Pauli weight cost
- `ogm`: Operator growth metric
- `l1`: L1 norm cost

### Objectives
- `tb`: Trajectory Balance
- `db`: Detailed Balance
- `subtb`: Sub-Trajectory Balance
- `fl`: Forward-Looking

## Project Structure

```
GFNDSS/
├── code/                      # Core implementation
│   ├── main.py               # Experiment runner
│   ├── run_config.py         # Config-based launcher
│   ├── GFNs.py               # GFlowNet implementation
│   ├── models.py             # Neural network architectures
│   ├── clifford_map.py       # Clifford tableau simulation
│   ├── energy_estimator.py   # Energy estimation
│   ├── cost_computer.py      # Cost functions
│   ├── gfn_objectives.py     # Training objectives
│   ├── gf2_ops.py            # GF(2) linear algebra
│   ├── masking_engine.py     # Action masking
│   ├── quantum_action_mapping.py  # Gate definitions
│   └── pauli_hamiltonian_helper.py # Hamiltonian parser
│
├── quantum_hardware_exp/      # Hardware experiment tools
│   ├── circuits/             # Circuit loading
│   ├── estimation/           # DSS estimator
│   ├── runner/               # Experiment runners
│   └── data/                 # Example data
│
└── Hamiltonians/             # Hamiltonian files (not included)
```

## Code Modules

### Core (`code/`)

| Module | Description |
|--------|-------------|
| `GFNs.py` | GFlowNet implementation with trajectory sampling, loss computation, and training loop |
| `models.py` | Neural network architectures for policy and backward sampler |
| `clifford_map.py` | Vectorized Clifford tableau simulation using Heisenberg representation |
| `energy_estimator.py` | GPU-accelerated energy estimation with derandomized measurements |
| `cost_computer.py` | Cost functions for circuit quality evaluation |
| `gfn_objectives.py` | Modular training objectives (TB, DB, SubTB, FL) |
| `gf2_ops.py` | GF(2) linear algebra for Clifford operations |
| `masking_engine.py` | Valid action mask computation |
| `quantum_action_mapping.py` | Mapping between action indices and quantum gates |
| `pauli_hamiltonian_helper.py` | Hamiltonian file parsing and ground state computation |
| `main.py` | High-level experiment orchestration |
| `run_config.py` | JSON config loader with cluster support |

### Quantum Hardware (`quantum_hardware_exp/`)

| Module | Description |
|--------|-------------|
| `circuits/dss_loader.py` | Load circuits from checkpoints |
| `circuits/hit_detection.py` | Pauli hit detection for measurements |
| `estimation/dss_estimator.py` | DSS energy estimation |
| `runner/snapshot_runner.py` | Execute snapshots on simulator/hardware |
| `runner/compare_dss.py` | Compare Flow-Shadow vs baseline |
| `energy_estimator_statevector.py` | Qiskit statevector estimation |
| `state_preparation.py` | Ground state preparation |
| `hamiltonian_loader.py` | Hamiltonian loading utilities |
| `phase_tracker.py` | Phase tracking for Clifford gates |

## Hamiltonian File Format

Supported formats:

**Two-line format** (`.txt`):
```
ZIZI
(0.17+0j)
IZIZ
(-0.22+0j)
```

**CSV format**:
```
pauli,coefficient
ZIZI,0.17
IZIZ,-0.22
```

**JSON format**:
```json
{
    "paulis": [
        {"label": "ZIZI", "coeff": 0.17},
        {"label": "IZIZ", "coeff": -0.22}
    ]
}
```

## Output

Results are saved to `results_dir/molecule_name/experiment_TIMESTAMP/`:

- `config.json`: Experiment configuration
- `checkpoint_update.pth`: Model checkpoint
- `metrics.jsonl`: Training metrics over time
- `results.jsonl`: Evaluation results
- `training_progress.png`: Training visualization

## License


