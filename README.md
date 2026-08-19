# FlowMeas

Generative learning of shallow Clifford measurement ensembles for quantum energy estimation.

FlowMeas recasts resource-constrained measurement design as a generative learning
problem: a GFlowNet policy samples a finite ensemble of shallow Clifford measurement
circuits subject to a hardware budget (CNOT depth, nearest-neighbour connectivity,
total number of measurements), trained against a state-independent proxy cost.

Reference implementation for:

> **Generative Learning for Quantum Measurement Design**
> Jun Dai, Olivier Nahman-Lévesque, Guillaume Rabusseau, Hong-Ye Hu, Cunlu Zhou
> arXiv:2608.11396 — https://arxiv.org/abs/2608.11396

## Method at a glance

A Hamiltonian is decomposed as `H = c₀I + Σₖ cₖPₖ`. An ensemble of `N` Clifford
circuits `U = (U₁,…,U_N)` covers term `Pₖ` whenever `UⱼPₖUⱼ†` is diagonal in the `Z`
basis; the number of covering circuits is the hit count `hₖ(U)`. A GFlowNet forward
policy builds all `N` circuits gate by gate from the action set

```
{H, S, HS, SH, HSH}_q  ∪  {CNOT_{q,q±1}}  ∪  {⊥}
```

under a cap of `d_max` CNOT layers, and is trained with an ensemble-level trajectory
balance objective against a reward `R(U) = Φ(C(U))`, where `C` is a state-independent
proxy cost computed from the coefficients `cₖ` and the hit counts alone.

## Status of this release

| Component | Status |
|---|---|
| GFlowNet training stack (policy, objectives, masking, sampling) | included |
| Clifford tableau simulation (CPU/PyTorch reference path) | included |
| Proxy cost functions and energy estimation | included |
| DMRG reference states for large-system benchmarks | included |
| **Batched Clifford-tableau GPU core** | **to be announced shortly** |

The GPU-accelerated Clifford-tableau core used for the large-scale runs in the paper
lives in a separate package and is **to be announced shortly**. It is an *optional*
dependency: `code/measurement_adapter/` resolves it lazily and falls back to the
bundled PyTorch tableau path in `code/clifford_map.py`, so everything in this
repository runs without it. Once the package is released, installing it will
transparently enable the batched CUDA backend — no changes to configs or training
code are required.

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Optional GPU backend (CuPy; see the note above about the tableau core):

```bash
pip install -r requirements-measurement-adapter.txt
```

## Quick start

```bash
# From project root
python code/run_config.py --config path/to/config.json
```

### Example configuration

```json
{
    "hamiltonian_path": "Hamiltonians/H2_STO3g_4qubits/jw.txt",
    "n_updates": 10000,
    "eval_every": 1000,
    "n_measurements": 1000,
    "max_depth": 2,
    "hidden_dim": 1024,
    "num_hidden_layers": 3,
    "lr": 1e-3,
    "model_type": "clifford_mlp",
    "cost_type": "confidence",
    "objective_type": "tb",
    "async_eval": true
}
```

## Configuration options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hamiltonian_path` | (required) | Path to Pauli Hamiltonian file |
| `n_updates` | 10000 | Total training updates |
| `eval_every` | 1000 | Evaluation frequency |
| `n_measurements` | 1000 | Ensemble size `N` (circuits per batch element) |
| `max_depth` | 8 | Maximum CNOT-layer depth `d_max` |
| `hidden_dim` | 1024 | Neural network hidden dimension |
| `num_hidden_layers` | 3 | Number of hidden layers |
| `lr` | 1e-3 | Learning rate |
| `model_type` | "clifford_mlp" | Policy architecture |
| `cost_type` | "exponential" | State-independent proxy cost |
| `objective_type` | "tb" | GFlowNet objective |
| `async_eval` | false | Enable asynchronous evaluation |
| `resume` | true | Resume from checkpoint if available |

### Proxy cost functions (`cost_type`)

The three proxies studied in the paper map to these names:

| `cost_type` | Paper | Description |
|---|---|---|
| `linear_bias` | `C_VB` | Variance-plus-bias: `Σ cₖ²/hₖ` over covered terms plus `(Σ\|cₖ\|)²` over uncovered terms |
| `confidence` | `C_DSS` | DSS confidence cost: `Σₖ wₖ exp(−ε²hₖ/2)` |
| `ogm` | `C_OGM` | Overlapped-grouping diagonal-variance objective |

Also available: `exponential`, `logarithmic`, `l1`.

### Policy architectures (`model_type`)

`clifford_mlp` — an MLP over the flattened `(2n)²` Clifford tableau, used as the
GFlowNet forward policy `P_F`. The backward policy `P_B` is a fixed uniform
distribution over valid actions (`uniform`) and is not configurable.

### Objectives (`objective_type`)

`tb` (trajectory balance), `db` (detailed balance), `subtb` (sub-trajectory
balance), `fl` (forward-looking), `entropy`, `multi`.

## Project structure

```
FlowMeas/
├── code/
│   ├── main.py                      # Experiment orchestration
│   ├── run_config.py                # JSON config launcher (entry point)
│   ├── config.py                    # ExperimentConfig + config coercion
│   ├── eval_restricted_depths.py    # Depth-restricted (d_max) evaluation sweep
│   │
│   ├── GFNs.py                      # Public GFlowNet import surface
│   ├── gfn_core.py                  # Construction / update / checkpointing
│   ├── gfn_flows.py                 # Forward / backward flow computation
│   ├── gfn_objectives.py            # TB, DB, SubTB, FL objectives
│   ├── gfn_sampling/                # Trajectory sampling + fused-kernel gates
│   ├── gfn_trainer.py               # Training loop
│   ├── gfn_trajectory.py            # Trajectory batch container
│   ├── gfn_runtime.py               # Runtime knobs, tableau protocol
│   ├── gfn_async.py                 # Asynchronous learner / sampler
│   ├── bucketed_sampler.py          # Bucketed static-capacity sampler
│   ├── models.py                    # CliffordMLP forward policy, uniform backward policy
│   ├── masking_engine.py            # Depth + connectivity action masks
│   ├── quantum_action_mapping.py    # Action index <-> gate mapping
│   │
│   ├── clifford_map.py              # PyTorch Clifford tableau (CPU path)
│   ├── gf2_ops.py                   # GF(2) linear algebra
│   ├── pauli_tracker.py             # Pauli conjugation / phase tracking
│   ├── measurement_adapter/         # Bridge to the GPU tableau core (optional)
│   │
│   ├── cost_computer.py             # State-independent proxy costs
│   ├── energy_estimator.py          # Energy estimation from ensembles
│   ├── pauli_hamiltonian_helper.py  # Hamiltonian parsing + ground states
│   ├── hubbard_loader.py            # Hubbard / compact-encoding workloads
│   ├── fci_solver.py                # FCI reference energies
│   ├── post_hf_solver.py            # MP2 / CISD / CCSD references
│   ├── dmrg_reference.py            # DMRG reference sidecars
│   ├── tenpy_dmrg.py                # TeNPy DMRG backend
│   ├── pauli_mpo_dmrg.py            # Pauli-sum MPO construction
│   ├── full_state_guard.py          # 26-qubit exact-full-state guardrail
│   ├── validation_tier.py           # Evaluation-tier contracts
│   │
│   ├── result_types.py              # Result dataclasses + JSON codec
│   ├── reporting.py                 # Async result export + summary stats
│   ├── energy_reporting.py          # Per-batch energy result assembly
│   ├── rmse_reporting.py            # Canonical RMSE reporting schema
│   │
│   ├── build_molecule_hamiltonian.py    # Molecular Hamiltonian generation
│   ├── generate_hubbard_hamiltonians.py # Hubbard Hamiltonian generation
│   ├── generate_h2o_grid.py             # H2O geometry grid
│   └── prepare_dmrg_ground_states.py    # DMRG ground-state precompute
│
└── Hamiltonians/                    # Hamiltonian files (not included)
```

## Hamiltonian file format

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

Results are written to `results_dir/molecule_name/experiment_TIMESTAMP/`:

- `config.json` — experiment configuration
- `checkpoint_update.pth` — model checkpoint
- `metrics_history.jsonl` — training metrics over time
- `results.jsonl` — evaluation results
- `evaluation_results.json` — exported evaluation results
- `summary_statistics.json` — per-update energy/RMSE/coverage summary

This release emits data only; no figures are produced.

## Citation

```bibtex
@article{dai2026flowmeas,
  title   = {Generative Learning for Quantum Measurement Design},
  author  = {Dai, Jun and Nahman-L{\'e}vesque, Olivier and Rabusseau, Guillaume
             and Hu, Hong-Ye and Zhou, Cunlu},
  journal = {arXiv preprint arXiv:2608.11396},
  year    = {2026}
}
```

## License

To be announced.
