"""
pysr_analysis.py
================
Symbolic regression stage of the GNN → PySR pipeline.

Workflow
--------
1. Load a trained MultipoleGNN checkpoint (.pth).
2. Rebuild the dataset with the same scaler used during training.
3. Extract raw edge-MLP outputs (messages) → design matrix (X, y).
4. Run PySR to discover the analytic expression for the electric potential.
5. Evaluate, denormalise, and save results.

Usage
-----
    python pysr_analysis.py --checkpoint monopole_YYYY-MM-DD_HH-MM-SS_multipole_gnn.pth
"""

import argparse
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from generate_data import MultipoleDataGenerator
from model import MultipoleGNN


# ------------------------------------------------------------------ #
# 1.  Message extraction                                               #
# ------------------------------------------------------------------ #

def extract_messages_for_pysr(
    model: MultipoleGNN,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the trained GNN in inference mode and collect the raw edge-MLP
    outputs together with their input features.

    Returns
    -------
    X : np.ndarray  shape [N, 10]
        Columns (in order):
        [src_x, src_y, src_z, src_q, src_r,
         obs_x, obs_y, obs_z, obs_q, obs_r]

    y : np.ndarray  shape [N]
        Raw output of the edge_mlp (the "message"), i.e. the network's
        learned representation of the per-edge potential contribution.
        This is what PySR will try to express symbolically.

    Notes
    -----
    For a monopole, we expect PySR to recover something proportional to
    src_q / src_r  (Coulomb's law), possibly up to the normalisation
    constant baked in during training.
    """
    model.eval()
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= max_samples:
                break

            batch = batch.to(device)
            msgs  = model.expose_messages(batch.x, batch.edge_index)  # [E, 1]

            src_idx = batch.edge_index[0]
            obs_idx = batch.edge_index[1]

            # Build the design matrix row by row: [source_features | obs_features]
            edge_X = torch.cat(
                [batch.x[src_idx], batch.x[obs_idx]], dim=1
            )  # [E, 10]

            X_list.append(edge_X.cpu().numpy())
            y_list.append(msgs.squeeze(-1).cpu().numpy())
            collected += edge_X.size(0)

    X = np.concatenate(X_list, axis=0)[:max_samples]
    y = np.concatenate(y_list, axis=0)[:max_samples]
    return X, y


# ------------------------------------------------------------------ #
# 2.  PySR configuration and run                                       #
# ------------------------------------------------------------------ #

FEATURE_NAMES = [
    "src_x", "src_y", "src_z", "src_q", "src_r",
    "obs_x", "obs_y", "obs_z", "obs_q", "obs_r",
]


def run_pysr(
    X: np.ndarray,
    y: np.ndarray,
    niterations: int = 50,
    output_dir: str = "pysr_results",
) -> "PySRRegressor":  # noqa: F821  (type hint only, avoids hard import at module level)
    """
    Run PySR symbolic regression on the extracted message data.

    PySR configuration rationale
    ----------------------------
    • binary_operators: +, *, / are the core Coulomb primitives.
      '-' is included for completeness (dipole/quadrupole extensions).
    • unary_operators: 'neg' and 'square' cover sign flips and 1/r²
      scenarios; 'sqrt' helps if the network encodes r from components.
    • maxsize=15: allows expressions like k*src_q/src_r without being
      too permissive (avoids overfitting to noise in the messages).
    • populations=20, niterations=50: good balance for a first run on
      a physics problem with a known closed form.

    Parameters
    ----------
    X            : design matrix [N, 10]
    y            : target messages [N]
    niterations  : PySR evolution steps (increase for harder problems)
    output_dir   : folder where PySR saves Hall-of-Fame CSV and PKL

    Returns
    -------
    sr_model : fitted PySRRegressor — inspect with print(sr_model) or
               sr_model.latex()
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError(
            "PySR is not installed. Run:  pip install pysr"
        )

    sr_model = PySRRegressor(
        # --- Search space ---
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["neg", "square", "sqrt"],
        maxsize=15,           # max symbolic complexity

        # --- Population / parallelism ---
        populations=20,       # independent evolutionary populations
        population_size=50,

        # --- Output ---
        output_jax_format=False,
        output_torch_format=True,   # get a torch-callable expression
        tempdir=output_dir,
        verbosity=1,

        # --- Reproducibility ---
        random_state=42,
        deterministic=True,
    )

    print("\n" + "=" * 60)
    print("Starting PySR symbolic regression …")
    print(f"  Samples  : {X.shape[0]}")
    print(f"  Features : {FEATURE_NAMES}")
    print(f"  Iters    : {niterations}")
    print("=" * 60 + "\n")

    sr_model.fit(X, y, variable_names=FEATURE_NAMES)
    return sr_model


# ------------------------------------------------------------------ #
# 3.  Evaluation and denormalisation                                   #
# ------------------------------------------------------------------ #

def evaluate_symbolic_model(
    sr_model,
    X: np.ndarray,
    y_norm: np.ndarray,
    scaler: dict,
) -> None:
    """
    Compare the PySR expressions against the true (denormalised) potential.

    The GNN was trained on normalised targets:
        V_norm = (V - V_mean) / V_std

    After PySR fits the messages (which live in normalised space) we
    denormalise to recover V in physical units and compute R².

    Parameters
    ----------
    sr_model : fitted PySRRegressor
    X        : design matrix used for fitting
    y_norm   : normalised messages (raw GNN output)
    scaler   : {'V_mean': float, 'V_std': float}
    """
    V_mean = scaler["V_mean"]
    V_std  = scaler["V_std"]

    print("\n" + "=" * 60)
    print("PySR Hall of Fame (normalised message space)")
    print("=" * 60)
    print(sr_model)

    # Best expression
    best_expr = sr_model.sympy()
    print(f"\nBest symbolic expression (normalised): {best_expr}")

    # Predict with the best expression
    y_pred_norm = sr_model.predict(X)

    # Denormalise both prediction and target to physical V
    y_pred_phys = y_pred_norm * V_std + V_mean
    y_true_phys = y_norm     * V_std + V_mean

    # R² in physical space
    ss_res = np.sum((y_true_phys - y_pred_phys) ** 2)
    ss_tot = np.sum((y_true_phys - y_true_phys.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    rmse = np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2))

    print(f"\nEvaluation in physical units (V = k·q/r)")
    print(f"  R²   : {r2:.6f}")
    print(f"  RMSE : {rmse:.6f}")

    # Latex form for the paper/notebook
    try:
        latex_expr = sr_model.latex()
        print(f"\nLaTeX expression:\n  {latex_expr}")
    except Exception:
        pass


# ------------------------------------------------------------------ #
# 4.  CLI entry point                                                  #
# ------------------------------------------------------------------ #

def load_checkpoint(path: str, device: torch.device):
    """Load model weights and scaler from a checkpoint saved by train.py."""
    checkpoint = torch.load(path, map_location=device)

    # Support both old format (state_dict only) and new format (dict with scaler)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        scaler     = checkpoint["scaler"]
    else:
        state_dict = checkpoint
        scaler     = None
        print(
            "Warning: checkpoint does not contain a scaler. "
            "Denormalisation will be skipped (V_mean=0, V_std=1)."
        )
        scaler = {"V_mean": 0.0, "V_std": 1.0}

    model = MultipoleGNN(node_features=5, hidden_dim=32).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="GNN → PySR symbolic regression")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pth file saved by train.py",
    )
    parser.add_argument(
        "--num_samples", type=int, default=10_000,
        help="Samples to regenerate for message extraction (default: 10 000)",
    )
    parser.add_argument(
        "--max_messages", type=int, default=5_000,
        help="Max edges fed to PySR (default: 5 000)",
    )
    parser.add_argument(
        "--niterations", type=int, default=50,
        help="PySR evolutionary iterations (default: 50)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="DataLoader batch size for message extraction (default: 256)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="pysr_results",
        help="Directory for PySR Hall-of-Fame outputs (default: pysr_results)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model, scaler = load_checkpoint(args.checkpoint, device)

    # --- Rebuild dataset (same distribution as training) ---
    # We regenerate data so this script is self-contained;
    # alternatively you could serialise the dataset to disk in train.py.
    print(f"Regenerating {args.num_samples} monopole samples …")
    generator = MultipoleDataGenerator(num_samples=args.num_samples, space_size=5.0)
    df        = generator.generate_monopole()
    dataset, _  = generator.df_to_pytorch_geometric(df, scaler=scaler)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Extract messages ---
    print("Extracting edge-MLP messages …")
    X, y = extract_messages_for_pysr(
        model, loader, device, max_samples=args.max_messages
    )
    print(f"Design matrix: X={X.shape}, y={y.shape}")

    # --- Run PySR ---
    sr_model = run_pysr(X, y, niterations=args.niterations, output_dir=args.output_dir)

    # --- Evaluate ---
    evaluate_symbolic_model(sr_model, X, y, scaler)

    print(f"\nDone. Full Hall-of-Fame saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
