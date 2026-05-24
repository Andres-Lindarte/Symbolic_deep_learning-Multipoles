# GNN-PySR: Symbolic Discovery of Electrostatic Laws

> Rediscovering the laws of electrostatics from data — using Graph Neural Networks and Symbolic Regression.

---

## Overview

This project implements the **GNN + Symbolic Regression** pipeline introduced by [Cranmer et al. (2020)](https://arxiv.org/abs/2006.11287) and applies it to classical electrostatics. A Graph Neural Network (GNN) is trained on synthetic data of electric monopoles and dipoles. Once trained, the raw outputs of its message-passing layer — which encode the learned per-edge interaction — are extracted and fed into [PySR](https://github.com/MilesCranmer/PySR) to recover a closed-form analytic expression. [SymPy](https://www.sympy.org/) is then used to simplify, verify, and cross-check the result against the known physical law.

The key insight is that `aggr='add'` in the GNN directly encodes the **superposition principle**: the total field at the observer is the sum of individual charge contributions. This means the network can generalize from monopoles to dipoles without any architectural change — and PySR only needs to find the per-edge law, not the full composite expression.

---

## Scientific Goal

Given a set of point charges and an observer position, the GNN should learn:

| Mode | Target | Known law |
|---|---|---|
| `potential` | Scalar V at observer | $V = k\,q / r$ |
| `efield_vector` | Vector **E** at observer (monopole) | $E_i = k\,q\,\Delta_i / r^3$ |
| `dipole_potential` | Scalar V at observer (dipole) | $V = k\,q/r_+ - k\,q/r_-$ |
| `dipole_efield` | Vector **E** at observer (dipole) | $\mathbf{E} = k\,q\,\hat{r}_+/r_+^2 - k\,q\,\hat{r}_-/r_-^2$ |

PySR then recovers the symbolic expression for the per-edge message and SymPy verifies that the three Cartesian components (Ex, Ey, Ez) are the same functional form under index permutation, confirming that a single universal law was learned.

---

## Repository Structure

```
GNN-PYSR-APPROACH/
│
├── generate_data.py        # Synthetic dataset generation (monopole & dipole)
├── model.py                # MultipoleGNN — message-passing GNN architecture
├── train.py                # Training loop, checkpointing, loss curves
├── pysr_analysis.py        # Message extraction → PySR → SymPy verification
│
└── outputs/
    ├── checkpoints/        # Trained model weights + scaler (.pth)
    │   └── {mode}_{datetime}.pth
    ├── plots/              # The loss curve of the GNN on train and validation
    |   └── loss_curve_{mode}{datetime}.png
    └── runs/               # PySR + SymPy results, one folder per run
        └── {datetime}/
            ├── Ex/         # PySR Hall of Fame for Ex component
            │   ├── hall_of_fame.csv
            │   └── checkpoint.pkl
            ├── Ey/
            ├── Ez/
            └── sympy_summary.txt
```

---

## Architecture

### Graph construction

Each physical configuration is encoded as a directed graph:

**Monopole** (2 nodes, 1 edge)
```
Node 0 [x, y, z, q,  r]  ──→  Node 1 [x, y, z, 0, r]
       source charge               observer
```

**Dipole** (3 nodes, 2 edges)
```
Node 0 [x, y, z, +q, r₊]  ──┐
                               ├──→  Node 2 [x, y, z, 0, 0]
Node 1 [x, y, z, -q, r₋]  ──┘         observer
```

The distance `r` is included as an explicit node feature so that PySR can directly discover `1/r` and `1/r²` dependencies without having to reconstruct distance from coordinates.

### GNN

```
Edge MLP  φᵉ : [x_source | x_observer] → message (per-edge contribution)
                  ↓ aggr='add' (superposition principle)
Node MLP  φᵛ : [x_observer | aggregated_message] → prediction
```

`aggr='add'` is not an arbitrary choice — it is a direct implementation of the superposition principle. For a dipole, the two messages from `+q` and `−q` are summed automatically, so the network generalizes without retraining.

`expose_messages()` bypasses the node MLP and returns the raw edge MLP output for each edge, which is what PySR receives as its regression target.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/GNN-PYSR-APPROACH.git
cd GNN-PYSR-APPROACH

# Create a virtual environment (recommended)
conda create -n symbolic-physics python=3.11 -y
conda activate symbolic-physics

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PySR requires Julia. On first run it will install Julia automatically. See [PySR installation docs](https://astroautomata.com/PySR/installation/) if you encounter issues.

---

## Usage

### 1 — Train the GNN

```bash
#   --- MONOPOLE ---
# Electric potential (scalar)
python train.py --mode potential

# Electric field (vector)
python train.py --mode efield_vector

#   --- DIPOLE  ---
# Electric potential (scalar)
python train.py --mode dipole_potential

# Electric field (vector)
python train.py --mode dipole_efield
```

Available modes:

| Flag | Physics | Output dim | Nodes/graph |
|---|---|---|---|
| `potential` | Monopole scalar V | 1 | 2 |
| `efield_vector` | Monopole vector **E** | 3 | 2 |
| `dipole_potential` | Dipole scalar V | 1 | 3 |
| `dipole_efield` | Dipole vector **E** | 3 | 3 |

The best checkpoint is saved automatically to `outputs/checkpoints/`. A loss curve (`loss_curve.png`) is saved alongside it.

### 2 — Run symbolic regression

```bash
python pysr_analysis.py --checkpoint outputs/checkpoints/<checkpoint.pth> 
```

Full CLI options:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pth` file from training |
| `--num_samples` | 10 000 | Samples to regenerate for message extraction |
| `--max_messages` | 5 000 | Max edges fed to PySR |
| `--niterations` | 50 | PySR evolutionary iterations (increase for harder problems) |
| `--batch_size` | 256 | DataLoader batch size during extraction |
| `--base_dir` | `outputs/runs` | Root folder for PySR + SymPy outputs |

### 3 — Read the results

After `pysr_analysis.py` completes, check:

```
outputs/runs/{datetime}/sympy_summary.txt
```

A successful run, for example, `efield_vector` should show:

```
[Ex]
  Simplified:  src_q * delta_x / src_r**3
  Known form:  src_q * delta_x / src_r**3
  Ratio:       1.0
  R²:          0.999997

[Symmetry check]
  ✓ SYMMETRIC — GNN learned one universal law:  E_i = f(q, r, Δi)
  Canonical form: src_q * delta_i / src_r**3
```

---

## Pipeline diagram

```
Synthetic data                  GNN training              Symbolic regression
─────────────────               ────────────────          ──────────────────────
generate_monopole()   ──→  MultipoleGNN.forward()  ──→  expose_messages()
generate_dipole()          aggr='add'                    ↓
                           (superposition)           PySR × 3 components
                                ↓                        ↓
                           checkpoint.pth           hall_of_fame.csv
                                                         ↓
                                                    SymPy.simplify()
                                                         ↓
                                                    sympy_summary.txt
```

---

## Key design decisions

**Why `aggr='add'`?**
The superposition principle states that the total electric field is the vector sum of individual contributions. By setting aggregation to `add`, this physical law is enforced as an architectural prior rather than learned from data. This makes the model more data-efficient and lets PySR analyze single-charge interactions instead of composite expressions.

**Why is `r` an explicit node feature?**
Without `r` as a feature, the network must infer distance from the difference between coordinate pairs. Adding it explicitly gives PySR a direct handle on `1/r` and `1/r²` terms, dramatically reducing the search space.

**Why extract messages, not predictions?**
The node MLP applies after aggregation and mixes the contributions of all source charges. Extracting the edge MLP output — before aggregation — isolates the per-charge interaction, which is the quantity with a simple closed form. PySR then only needs to find `k·q·Δi/r³`, not the full multi-charge expression.

**Why run PySR three times for vector output?**
PySR natively handles scalar regression. For vector fields, three independent runs are launched (one per component Ex, Ey, Ez). SymPy then checks whether the three results are the same functional form under index permutation — a necessary condition for a physically consistent vector law.


## References

- Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., et al. (2020). *Discovering Symbolic Models from Deep Learning with Inductive Biases.* NeurIPS 2020. [arXiv:2006.11287](https://arxiv.org/abs/2006.11287)
- Cranmer, M. (2023). *PySR: Fast & Parallelized Symbolic Regression in Python/Julia.* [GitHub](https://github.com/MilesCranmer/PySR)
- Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)

---

## License

TO-DO