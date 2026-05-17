import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MultipoleDataGenerator:
    """
    Generates synthetic datasets for training a GNN to learn electrostatic
    quantities (potential V or electric field E) of monopoles, dipoles, and
    quadrupoles.

    Graph layout (both modes)
    -------------------------
    Node 0 — source  : [x, y, z, q, r]
    Node 1 — observer: [x, y, z, 0, r]
    Directed edge    : 0 → 1  (information flows source → observer)
    """

    def __init__(self, num_samples: int = 10_000, space_size: float = 10.0, k_constant: float = 1.0):
        self.num_samples = num_samples
        self.space_size  = space_size
        self.k           = k_constant

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _generate_random_position(self) -> np.ndarray:
        return np.random.uniform(-self.space_size, self.space_size, 3)

    def _generate_random_charge(self, min_val: float = -5.0, max_val: float = 5.0) -> float:
        """Resample until |q| >= 0.1 to avoid near-zero charges."""
        while True:
            q = np.random.uniform(min_val, max_val)
            if abs(q) >= 0.1:
                return q

    def _base_sample(self):
        """
        Generate a single valid (source, observer) pair.
        Returns None if the distance is too small (singularity guard).
        """
        pos_src = self._generate_random_position()
        q_src   = self._generate_random_charge()
        pos_obs = self._generate_random_position()

        delta = pos_obs - pos_src
        r     = np.linalg.norm(delta)

        if r < 0.1:
            return None

        return pos_src, q_src, pos_obs, delta, r

    # ------------------------------------------------------------------ #
    # Data generation                                                       #
    # ------------------------------------------------------------------ #

    def generate_monopole(self) -> pd.DataFrame:
        """
        Electric potential of a monopole.

            V = k * q / r

        Target column: 'target_V'
        """
        rows = []
        while len(rows) < self.num_samples:
            sample = self._base_sample()
            if sample is None:
                continue
            pos_src, q_src, pos_obs, delta, r = sample

            rows.append({
                'source_x': pos_src[0], 'source_y': pos_src[1], 'source_z': pos_src[2],
                'source_q': q_src,
                'obs_x':    pos_obs[0], 'obs_y':    pos_obs[1], 'obs_z':    pos_obs[2],
                'obs_q':    0.0,
                'distance_r': r,
                'target_V': self.k * q_src / r,
            })
        return pd.DataFrame(rows)

    def generate_monopole_efield(self) -> pd.DataFrame:
        """
        Electric field magnitude of a monopole.

            |E| = k * q / r²

        The sign of q is preserved so the GNN learns the signed magnitude
        (field points outward for +q, inward for -q).

        Target column : 'target_E_mag'
        Extra columns : 'target_Ex', 'target_Ey', 'target_Ez'
                        (kept for analysis / future vector training)
        """
        rows = []
        while len(rows) < self.num_samples:
            sample = self._base_sample()
            if sample is None:
                continue
            pos_src, q_src, pos_obs, delta, r = sample

            E_mag = self.k * q_src / r**2          # signed scalar  k·q/r²
            r_hat = delta / r                       # unit vector source→observer
            E_vec = E_mag * r_hat                   # [Ex, Ey, Ez]

            rows.append({
                'source_x': pos_src[0], 'source_y': pos_src[1], 'source_z': pos_src[2],
                'source_q': q_src,
                'obs_x':    pos_obs[0], 'obs_y':    pos_obs[1], 'obs_z':    pos_obs[2],
                'obs_q':    0.0,
                'distance_r': r,
                'target_E_mag': E_mag,
                'target_Ex':    E_vec[0],
                'target_Ey':    E_vec[1],
                'target_Ez':    E_vec[2],
                'target_V':     self.k * q_src / r,  # potential kept for reference
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # DataFrame → PyTorch Geometric                                         #
    # ------------------------------------------------------------------ #

    def df_to_pytorch_geometric(
        self,
        df:         pd.DataFrame,
        scaler:     dict | None       = None,
        target_col: str | list[str]   = 'target_V',
    ) -> tuple[list[Data], dict]:
        """
        Convert a multipole DataFrame to a list of PyG graph objects.

        Parameters
        ----------
        df         : DataFrame produced by generate_monopole[_efield]()
        scaler     : dict with per-column 'mean' and 'std' entries.
                     Pass None on the training set to compute from data;
                     pass the returned scaler to all subsequent sets.
        target_col : str  → scalar target, y shape [1, 1]
                     list → vector target, y shape [1, len(list)]
                     Examples:
                       'target_V'                          — electric potential
                       'target_E_mag'                      — |E| scalar
                       ['target_Ex','target_Ey','target_Ez'] — E vector (3-component)

        Returns
        -------
        dataset : list[Data]
        scaler  : dict  {col: {'mean': float, 'std': float}, ...}
        """
        # Normalise target_col to always be a list internally
        cols = [target_col] if isinstance(target_col, str) else list(target_col)

        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"target_col='{col}' not found in DataFrame. "
                    f"Available: {[c for c in df.columns if c.startswith('target_')]}"
                )

        # --- Per-column scaler ---
        if scaler is None:
            scaler = {}
            for col in cols:
                mean = float(df[col].mean())
                std  = float(df[col].std())
                std  = std if std > 1e-8 else 1.0
                scaler[col] = {'mean': mean, 'std': std}

        dataset = []
        for _, row in df.iterrows():
            r = row['distance_r']

            # Node features: [x, y, z, q, r]
            # r explicit so PySR can discover 1/r, 1/r², Δi/r³ directly
            node_src = [row['source_x'], row['source_y'], row['source_z'], row['source_q'], r]
            node_obs = [row['obs_x'],    row['obs_y'],    row['obs_z'],    row['obs_q'],    r]

            x          = torch.tensor([node_src, node_obs], dtype=torch.float)
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)

            # y shape: [1, output_dim]  — works for both scalar and vector
            target_norm = [
                (row[col] - scaler[col]['mean']) / scaler[col]['std']
                for col in cols
            ]
            y = torch.tensor([target_norm], dtype=torch.float)  # [1, output_dim]

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

        return dataset, scaler