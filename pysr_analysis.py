"""
pysr_analysis.py
================
Symbolic regression stage of the GNN → PySR pipeline.

For efield_vector mode the pipeline runs PySR three times in parallel
(one per component Ex, Ey, Ez) and uses SymPy to verify that all three
expressions are the same functional form under permutation of the
coordinate index — which is expected for a Coulomb field.

Workflow
--------
1. Load a trained MultipoleGNN checkpoint (.pth).
2. Rebuild the dataset using the exact scaler from training.
3. Extract raw edge-MLP messages  →  design matrix (X, [y_Ex, y_Ey, y_Ez]).
4. Run PySR independently for each component.
5. Verify with SymPy: simplify, compare against k·q·Δi/r³, cross-check symmetry.
6. Save all outputs with a datetime stamp.

Usage
-----
    python pysr_analysis.py --checkpoint <outputs/checkpoints/saved .pth file>
"""

import argparse
import os
import numpy as np
import torch
from datetime import datetime
from torch_geometric.loader import DataLoader

from generate_data import MultipoleDataGenerator
from model import MultipoleGNN


# ------------------------------------------------------------------ #
# Feature column names (matches node layout [x, y, z, q, r])          #
# ------------------------------------------------------------------ #
FEATURE_NAMES = [
    "src_x", "src_y", "src_z", "src_q", "src_r",
    "obs_x", "obs_y", "obs_z", "obs_q", "obs_r",
]

# For efield_vector the GNN outputs 3 channels: Ex, Ey, Ez
COMPONENT_NAMES = ["Ex", "Ey", "Ez"]

# Known analytic forms in Cartesian coordinates (k=1)
# E_i = k·q·Δi / r³   where Δi = obs_i - src_i
_KNOWN_VECTOR_EXPR = {
    "Ex": "src_q * delta_x / src_r**3",
    "Ey": "src_q * delta_y / src_r**3",
    "Ez": "src_q * delta_z / src_r**3",
}

# ------------------------------------------------------------------ #
# 1.  Checkpoint loading                                               #
# ------------------------------------------------------------------ #

def load_checkpoint(path: str, device: torch.device) -> tuple:
    ckpt = torch.load(path, map_location=device)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            "Old checkpoint format. Re-train with the updated train.py."
        )

    mode       = ckpt.get("mode",          "efield_vector")
    scaler     = ckpt.get("scaler",        {})
    nf         = ckpt.get("node_features", 5)
    hd         = ckpt.get("hidden_dim",    32)
    output_dim = ckpt.get("output_dim",    3)
    nodes_per_graph = ckpt.get("nodes_per_graph", 2) # 2 for monopole, 3 for dipole.

    model = MultipoleGNN(node_features=nf, hidden_dim=hd, output_dim=output_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint : {path}")
    print(f"Mode       : {mode}  |  output_dim={output_dim}  |  nodes_per_graph={nodes_per_graph}")
    return model, scaler, mode, output_dim, nodes_per_graph


# ------------------------------------------------------------------ #
# 2.  Message extraction                                               #
# ------------------------------------------------------------------ #

def extract_messages_for_pysr(
    model:       MultipoleGNN,
    loader:      DataLoader,
    device:      torch.device,
    output_dim:  int,
    max_samples: int = 5_000,
) -> tuple:
    """
    Collect raw edge-MLP outputs and their input features.

    We append the three displacement components (Δx, Δy, Δz) to X
    because E_i depends on Δi = obs_i - src_i, not on src_i and obs_i
    separately. Giving PySR Δ directly makes the search faster.

    Returns
    -------
    X : [N, 13]         FEATURE_NAMES + ["delta_x", "delta_y", "delta_z"]
    y : [N, output_dim] edge-MLP output per component
    """
    model.eval()
    X_list, y_list = [], []
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= max_samples:
                break
            batch = batch.to(device)

            msgs    = model.expose_messages(batch.x, batch.edge_index)  # [E, output_dim]
            src_idx = batch.edge_index[0]
            obs_idx = batch.edge_index[1]

            src_feats = batch.x[src_idx]   # [E, 5]
            obs_feats = batch.x[obs_idx]   # [E, 5]

            # Δ = obs_pos - src_pos  for x, y, z (indices 0,1,2)
            delta = obs_feats[:, :3] - src_feats[:, :3]  # [E, 3]

            # Full design matrix: [src | obs | Δx | Δy | Δz]
            edge_X = torch.cat([src_feats, obs_feats, delta], dim=1)  # [E, 13]

            X_list.append(edge_X.cpu().numpy())
            y_list.append(msgs.cpu().numpy())
            collected += edge_X.size(0)

    X = np.concatenate(X_list, axis=0)[:max_samples]
    y = np.concatenate(y_list, axis=0)[:max_samples]

    print(f"Messages extracted : X={X.shape}, y={y.shape}")
    return X, y


# ------------------------------------------------------------------ #
# 3.  PySR — one run per component                                     #
# ------------------------------------------------------------------ #

_FULL_FEATURE_NAMES = FEATURE_NAMES + ["delta_x", "delta_y", "delta_z"]


def run_pysr_component(
    X:           np.ndarray,
    y_component: np.ndarray,
    component:   str,
    niterations: int,
    run_dir:  str,
):
    """Run PySR for a single scalar component (Ex, Ey, or Ez)."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError("PySR is not installed. Run:  pip install pysr")

    comp_dir = os.path.join(run_dir, component)
    os.makedirs(comp_dir, exist_ok=True)

    sr_model = PySRRegressor(
        niterations      = niterations,
        binary_operators = ["+", "-", "*", "/"],
        unary_operators  = ["neg",  "cube", "square", "sqrt"],
        maxsize          = 15,
        populations      = 20,
        population_size  = 50,
        output_jax_format   = False,
        output_torch_format = True,
        tempdir          = comp_dir,    # Julia temp files
        temp_equation_file    = os.path.join(comp_dir, "hall_of_fame.csv"),  # Save PySR hall of fame here
        verbosity        = 1,
        random_state     = 42,
        deterministic    = False,
    )

    print(f"\n{'='*60}")
    print(f"PySR — component {component}  ({niterations} iterations)")
    print(f"Output dir : {comp_dir}")
    print(f"{'='*60}\n")

    sr_model.fit(X, y_component, variable_names=_FULL_FEATURE_NAMES)
    return sr_model


def run_pysr_all_components(
    X:           np.ndarray,
    y:           np.ndarray,
    output_dim:  int,
    niterations: int,
    run_dir:      str,
) -> tuple:
    """
    Run PySR once per output component.

    Scalar mode (output_dim=1) → 1 run.
    Vector mode (output_dim=3) → 3 runs: Ex, Ey, Ez.
    """
    components = COMPONENT_NAMES[:output_dim] if output_dim == 3 else ["scalar"]
    models = []

    for i, comp in enumerate(components):
        y_comp = y[:, i] if output_dim > 1 else y.squeeze()
        sr = run_pysr_component(X, y_comp, comp, niterations, run_dir)
        models.append(sr)

    return models, components


# ------------------------------------------------------------------ #
# 4.  SymPy verification                                               #
# ------------------------------------------------------------------ #

def verify_with_sympy(
    sr_models:  list,
    components: list,
    mode:       str,
    scaler:     dict,
    X:          np.ndarray,
    y:          np.ndarray,
    run_dir:     str,
) -> None:
    """
    For each component:
      1. Simplify the PySR expression with SymPy.
      2. Compare against the known analytic form k·q·Δi/r³.
      3. Compute ratio found/known → should be the learned constant k.

    Then cross-check symmetry: verify that Ex, Ey, Ez are the same
    functional form under permutation delta_x→delta_i, revealing whether
    the GNN learned one universal law or three separate ones.
    """
    try:
        import sympy as sp
    except ImportError:
        print("SymPy not installed. Run:  pip install sympy")
        return

    print("\n" + "=" * 60)
    print("SymPy Verification")
    print("=" * 60)

    summary_lines = [f"Run ID : {run_dir}", f"Mode   : {mode}", ""]

    simplified_exprs = []
    delta_syms       = []
    r2_scores        = []

    dx, dy, dz = sp.symbols("delta_x delta_y delta_z", real=True)
    delta_map  = {"Ex": dx, "Ey": dy, "Ez": dz}

    for i, (sr_model, comp) in enumerate(zip(sr_models, components)):
        print(f"\n--- Component {comp} ---")

        # Get scaler for this component
        # Both efield_vector and dipole_vector use per-component scalers
        # keyed as 'target_Ex', 'target_Ey', 'target_Ez'
        if mode in ("efield_vector", "dipole_vector"):
            col      = f"target_{comp}"
            mean_val = scaler.get(col, {}).get("mean", 0.0)
            std_val  = scaler.get(col, {}).get("std",  1.0)
        else:
            col      = list(scaler.keys())[0]
            mean_val = scaler[col]["mean"]
            std_val  = scaler[col]["std"]

        # SymPy simplification
        try:
            best_pysr  = sr_model.sympy()
            simplified = sp.simplify(best_pysr)
        except Exception as e:
            print(f"  Could not parse PySR expression: {e}")
            simplified_exprs.append(None)
            delta_syms.append(None)
            continue

        latex_expr = sp.latex(simplified)
        print(f"  PySR (raw)   : {best_pysr}")
        print(f"  Simplified   : {simplified}")
        print(f"  LaTeX        : {latex_expr}")

        # Compare with known form
        # For both monopole and dipole, the per-edge message learned by the
        # GNN should be k·q·Δi/r³ — the dipole field emerges from aggr='add'
        if mode in ("efield_vector", "dipole_vector") and comp in _KNOWN_VECTOR_EXPR:
            known_expr = sp.sympify(_KNOWN_VECTOR_EXPR[comp])
            print(f"  Known form   : {known_expr}")
 
            try:
                ratio = sp.simplify(simplified / known_expr)
                print(f"  Ratio (found/known) = {ratio}")
                print(f"  → Constant ratio = correct form (learned k ≈ {ratio})")
            except Exception as e:
                ratio = f"N/A ({e})"

            summary_lines += [
                f"[{comp}]",
                f"  PySR      : {best_pysr}",
                f"  Simplified: {simplified}",
                f"  LaTeX     : {latex_expr}",
                f"  Known     : {known_expr}",
                f"  Ratio     : {ratio}",
            ]
        else:
            summary_lines += [
                f"[{comp}]",
                f"  PySR      : {best_pysr}",
                f"  Simplified: {simplified}",
                f"  LaTeX     : {latex_expr}",
            ]

        simplified_exprs.append(simplified)
        delta_syms.append(delta_map.get(comp))

        # R² in physical units
        y_comp      = y[:, i] if y.ndim > 1 else y
        y_pred_norm = sr_model.predict(X)
        y_pred_phys = y_pred_norm * std_val + mean_val
        y_true_phys = y_comp     * std_val + mean_val

        ss_res = float(np.sum((y_true_phys - y_pred_phys) ** 2))
        ss_tot = float(np.sum((y_true_phys - y_true_phys.mean()) ** 2))
        r2   = 1.0 - ss_res / ss_tot
        rmse = float(np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2)))
        r2_scores.append(r2)

        print(f"  R²   : {r2:.6f}")
        print(f"  RMSE : {rmse:.6f}")
        summary_lines += [f"  R²   : {r2:.6f}", f"  RMSE : {rmse:.6f}", ""]

    # --- Cross-component symmetry check (vector mode only) ---
    if (mode == "efield_vector"
            and len(simplified_exprs) == 3
            and all(e is not None for e in simplified_exprs)):

        print("\n--- Symmetry cross-check (Ex, Ey, Ez same law?) ---")
        di = sp.Symbol("delta_i", real=True)

        canonical = []
        for expr, d_sym in zip(simplified_exprs, [dx, dy, dz]):
            # Replace the component-specific delta with the generic delta_i
            canonical.append(sp.simplify(expr.subs(d_sym, di)))

        print(f"  Ex(delta_x→δi) : {canonical[0]}")
        print(f"  Ey(delta_y→δi) : {canonical[1]}")
        print(f"  Ez(delta_z→δi) : {canonical[2]}")

        eq_xy = sp.simplify(canonical[0] - canonical[1]) == 0
        eq_xz = sp.simplify(canonical[0] - canonical[2]) == 0

        if eq_xy and eq_xz:
            status = "SYMMETRIC — GNN learned one universal law:  E_i = f(q, r, Δi)"
        else:
            status = "NOT symmetric — components differ (try more epochs or iterations)"

        print(f"\n  {status}")
        summary_lines += ["", "[Symmetry check]", f"  {status}"]
        if eq_xy and eq_xz:
            summary_lines.append(f"  Canonical form: {canonical[0]}")

    if r2_scores:
        print(f"\n  Mean R² across components: {np.mean(r2_scores):.6f}")

    # --- Save summary ---
    summary_path = os.path.join(run_dir, "sympy_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSymPy summary saved → {summary_path}")


# ------------------------------------------------------------------ #
# 5.  CLI                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="GNN → PySR (vector) → SymPy pipeline")
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--num_samples",  type=int, default=10_000)
    parser.add_argument("--max_messages", type=int, default=5_000)
    parser.add_argument("--niterations",  type=int, default=50)
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--base_dir",   type=str, default="outputs/runs")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Run ID  : {run_id}")

    # --- Load model ---
    model, scaler, mode, output_dim, nodes_per_graph = load_checkpoint(args.checkpoint, device)

    # --- Rebuild dataset ---
    print(f"\nGenerating {args.num_samples} samples (mode={mode}) …")
    generator = MultipoleDataGenerator(num_samples=args.num_samples, space_size=5.0)

    if mode == "potential":
        df         = generator.generate_monopole()
        target_col = "target_V"
        dataset, _ = generator.df_to_pytorch_geometric(df, scaler=scaler, target_col=target_col)

    elif mode == "efield_vector":
        df         = generator.generate_monopole_efield()
        target_col = ["target_Ex", "target_Ey", "target_Ez"]
        dataset, _ = generator.df_to_pytorch_geometric(df, scaler=scaler, target_col=target_col)

    elif mode == "dipole_potential":
        df         = generator.generate_dipole_potential()
        target_col = "target_V"
        dataset, _ = generator.dipole_df_to_pytorch_geometric(df, scaler=scaler, target_col=target_col)

    elif mode == "dipole_vector":
        df         = generator.generate_dipole()
        target_col = ["target_Ex", "target_Ey", "target_Ez"]
        dataset, _ = generator.df_to_pytorch_geometric(df, scaler=scaler, target_col=target_col)

    else:
        raise ValueError(f"Unknown mode: {mode} in checkpoint. Expected 'potential', 'efield_vector','dipole_potential' or 'dipole_vector'.")

    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Extract messages ---
    X, y = extract_messages_for_pysr(model, loader, device, output_dim, max_samples=args.max_messages)

    # --- PySR (one run per component) ---
    sr_models, components = run_pysr_all_components(
        X, y, output_dim,
        niterations=args.niterations,
        run_dir=run_dir,
    )

    for sr, comp in zip(sr_models, components):
        print(f"\n{'='*60}")
        print(f"Hall of Fame — {comp}")
        print(sr)

    # --- SymPy ---
    verify_with_sympy(sr_models, components, mode, scaler, X, y, run_dir)

    print(f"\nAll outputs saved under '{run_dir}/'")


if __name__ == "__main__":
    main()