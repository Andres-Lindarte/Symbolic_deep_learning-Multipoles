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

    Graph layouts 
    - Monopole potential or electric field:
        2 nodes, 1 edge
    -------------------------
    Node 0 — source  : [x, y, z, q, r]
    Node 1 — observer: [x, y, z, 0, r]
    Directed edge    : 0 → 1  (information flows source → observer)

    - Dipole potential or electric field:
        3 nodes, 2 edges
    -------------------------
    Node 0 — positive source  : [x, y, z, +q, r]
    Node 1 — negative source  : [x, y, z, -q, r]
    Node 2 — observer: [x, y, z, 0, r]
    Directed edge    : 0 → 2  and 1 → 2  (information flows sources → observer)

    In both cases the observer is always the last node.
    aggr='add' in the GNN handles superposition automatically.
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

    def _random_unit_vector(self) -> np.ndarray:
        """Return a random unit vector uniformly distributed on the surface of the unit sphere."""
        v = np.random.randn(3)
        return v / np.linalg.norm(v)

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
    
    def generate_dipole(self, min_separation: float = 0.2, max_separation: float = 2.0) -> pd.DataFrame:
        """
        Electric potential of a physical dipole (two opposite point charges).

        The dipole is built from two monopoles +q and -q separated by a
        random distance d along a random orientation axis.  The total potential
        at the observer is the scalar superposition of the two monopole potentials:

            V_total = V(+q, r_plus) + V(-q, r_minus)

        where each monopole contribution is V_i = k·q / r.

        The GNN learns this automatically via aggr='add', so no change to
        model.py is needed.

        Graph layout — 3 nodes, 2 directed edges
        -----------------------------------------
            Node 0 — charge +q : [x, y, z, +q, r_plus ]
            Node 1 — charge -q : [x, y, z, -q, r_minus]
            Node 2 — observer  : [x, y, z,  0,  0      ]
            Edges  : 0 → 2  and  1 → 2

        Parameters
        ----------
        min_separation : float
            Minimum distance between +q and -q (avoids degenerate dipoles).
        max_separation : float
            Maximum distance between +q and -q.

        Target column
        -------------
        target_V — total potential at the observer

        Extra columns (per-charge geometry, useful for analysis)
        ---------------------------------------------------------
        plus_x/y/z, minus_x/y/z
        r_plus, r_minus
        dipole_px/py/pz   — dipole moment vector p = q·d·orientation
        """
        rows = []
        while len(rows) < self.num_samples:

            # --- Dipole geometry ---
            pos_center  = self._generate_random_position()
            orientation = self._random_unit_vector()           # axis of the dipole
            d = np.random.uniform(min_separation, max_separation)  # separation

            pos_plus  = pos_center + (d / 2.0) * orientation  # +q position
            pos_minus = pos_center - (d / 2.0) * orientation  # −q position
            q = self._generate_random_charge(min_val=0.1, max_val=5.0)  # always positive

            # --- Observer ---
            pos_obs = self._generate_random_position()

            rows.append({
                'plus_x':  pos_plus[0],  'plus_y':  pos_plus[1],  'plus_z':  pos_plus[2],
                'minus_x': pos_minus[0], 'minus_y': pos_minus[1], 'minus_z': pos_minus[2],
                'obs_x': pos_obs[0], 'obs_y': pos_obs[1], 'obs_z': pos_obs[2],
                'r_plus':  np.linalg.norm(pos_obs - pos_plus),
                'r_minus': np.linalg.norm(pos_obs - pos_minus),
                'dipole_px': q * d * orientation[0],
                'dipole_py': q * d * orientation[1],
                'dipole_pz': q * d * orientation[2],
                'target_V': self.k * q / np.linalg.norm(pos_obs - pos_plus) + self.k * (-q) / np.linalg.norm(pos_obs - pos_minus),
            })

            return pd.DataFrame(rows)

    def generate_dipole_efield(self, min_separation: float = 0.2, max_separation: float = 2.0) -> pd.DataFrame:
            """
            Electric field of a physical dipole (two opposite point charges).
    
            The dipole is built from two monopoles +q and -q separated by a
            random distance d along a random orientation axis.  The total field
            at the observer is the vector superposition of the two monopole fields:
    
                E_total = E(+q, r_plus) + E(-q, r_minus)
    
            where each monopole contribution is  E_i = k·q·Δi / r³.
    
            The GNN learns this automatically via aggr='add', so no change to
            model.py is needed.
    
            Graph layout — 3 nodes, 2 directed edges
            -----------------------------------------
                Node 0 — charge +q : [x, y, z, +q, r_plus ]
                Node 1 — charge -q : [x, y, z, -q, r_minus]
                Node 2 — observer  : [x, y, z,  0,  0      ]
                Edges  : 0 → 2  and  1 → 2
    
            Parameters
            ----------
            min_separation : float
                Minimum distance between +q and -q (avoids degenerate dipoles).
            max_separation : float
                Maximum distance between +q and -q.
    
            Target columns
            --------------
            target_Ex, target_Ey, target_Ez — total vector field at the observer
            target_E_mag                    — magnitude |E_total| (for reference)
    
            Extra columns (per-charge geometry, useful for analysis)
            ---------------------------------------------------------
            plus_x/y/z, minus_x/y/z
            r_plus, r_minus
            dipole_px/py/pz   — dipole moment vector p = q·d·orientation
            """
            rows = []
            while len(rows) < self.num_samples:
    
                # --- Dipole geometry ---
                pos_center  = self._generate_random_position()
                orientation = self._random_unit_vector()           # axis of the dipole
                d = np.random.uniform(min_separation, max_separation)  # separation
    
                pos_plus  = pos_center + (d / 2.0) * orientation  # +q position
                pos_minus = pos_center - (d / 2.0) * orientation  # −q position
                q = self._generate_random_charge(min_val=0.1, max_val=5.0)  # always positive
    
                # --- Observer ---
                pos_obs = self._generate_random_position()
    
                # --- Displacements and distances ---
                delta_plus  = pos_obs - pos_plus   # vector from +q to observer
                delta_minus = pos_obs - pos_minus  # vector from −q to observer
                r_plus      = np.linalg.norm(delta_plus)
                r_minus     = np.linalg.norm(delta_minus)
    
                # Singularity guard: observer must be far enough from both charges
                if r_plus < 0.1 or r_minus < 0.1:
                    continue
    
                # --- Electric field contributions (k·q·Δ/r³ per charge) ---
                E_plus  = self.k *  q * delta_plus  / r_plus**3   # [Ex, Ey, Ez] from +q
                E_minus = self.k * -q * delta_minus / r_minus**3  # [Ex, Ey, Ez] from −q
                E_total = E_plus + E_minus                         # superposition
    
                # --- Potential (for reference) ---
                V_total = self.k * q / r_plus + self.k * (-q) / r_minus
    
                # --- Dipole moment vector p = q·d·orientation ---
                p_vec = q * d * orientation
    
                rows.append({
                    'plus_x':  pos_plus[0],  'plus_y':  pos_plus[1],  'plus_z':  pos_plus[2],
                    'minus_x': pos_minus[0], 'minus_y': pos_minus[1], 'minus_z': pos_minus[2],
                    'obs_x': pos_obs[0], 'obs_y': pos_obs[1], 'obs_z': pos_obs[2],
                    'r_plus':  r_plus,
                    'r_minus': r_minus,
                    'dipole_px': p_vec[0], 'dipole_py': p_vec[1], 'dipole_pz': p_vec[2],
                    'target_Ex':    E_total[0],
                    'target_Ey':    E_total[1],
                    'target_Ez':    E_total[2],
                    'target_E_mag': float(np.linalg.norm(E_total)),
                    'target_V':     V_total,
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

    def dipole_df_to_pytorch_geometric(
        self,
        df:         pd.DataFrame,
        scaler:     dict | None     = None,
        target_col: str | list[str] = ['target_Ex', 'target_Ey', 'target_Ez'],
    ) -> tuple[list[Data], dict]:
        """
        Convert a dipole DataFrame to a list of PyG graph objects.
 
        Graph : 3 nodes, 2 directed edges
        -----------------------------------
            Node 0 : [x, y, z, +q, r_plus ]   charge +q
            Node 1 : [x, y, z, -q, r_minus]   charge -q
            Node 2 : [x, y, z,  0,  0     ]   observer (r=0 placeholder)
            Edges  : [[0, 1], [2, 2]]          both sources → observer
 
        The node feature layout [x, y, z, q, r] is identical to the
        monopole case so the same trained model.py weights are reusable.
 
        Parameters
        ----------
        df         : DataFrame from generate_dipole_efield()
        scaler     : per-column {'mean', 'std'} dict.
                     Pass None on train set; reuse on val/test.
        target_col : columns to use as prediction targets (default: Ex,Ey,Ez).
 
        Returns
        -------
        dataset : list[Data]
        scaler  : dict {col: {'mean': float, 'std': float}}
        """
        cols = [target_col] if isinstance(target_col, str) else list(target_col)
 
        for col in cols:
            if col not in df.columns:
                raise ValueError(
                    f"target_col='{col}' not found. "
                    f"Available: {[c for c in df.columns if c.startswith('target_')]}"
                )
 
        # --- Per-column scaler (train set only) ---
        if scaler is None:
            scaler = {}
            for col in cols:
                mean = float(df[col].mean())
                std  = float(df[col].std())
                scaler[col] = {'mean': mean, 'std': std if std > 1e-8 else 1.0}
 
        dataset = []
        for _, row in df.iterrows():
            r_plus  = row['r_plus']
            r_minus = row['r_minus']
 
            # Recover q from the stored dipole moment:  |p| = q * d
            d_vec = np.array([row['plus_x']  - row['minus_x'],
                               row['plus_y']  - row['minus_y'],
                               row['plus_z']  - row['minus_z']])
            d     = float(np.linalg.norm(d_vec)) + 1e-9
            p_mag = float(np.linalg.norm([row['dipole_px'],
                                          row['dipole_py'],
                                          row['dipole_pz']]))
            q_val = p_mag / d   # q = |p| / d
 
            # Node features: [x, y, z, q, r]  — identical layout to monopole
            node_plus  = [row['plus_x'],  row['plus_y'],  row['plus_z'],  +q_val, r_plus ]
            node_minus = [row['minus_x'], row['minus_y'], row['minus_z'], -q_val, r_minus]
            node_obs   = [row['obs_x'],   row['obs_y'],   row['obs_z'],    0.0,   0.0    ]
 
            x = torch.tensor([node_plus, node_minus, node_obs], dtype=torch.float)
 
            # Both source nodes point to the observer (node 2)
            edge_index = torch.tensor([[0, 1],
                                       [2, 2]], dtype=torch.long)
 
            # Normalised target — shape [1, output_dim]
            target_norm = [
                (row[col] - scaler[col]['mean']) / scaler[col]['std']
                for col in cols
            ]
            y = torch.tensor([target_norm], dtype=torch.float)
 
            dataset.append(Data(x=x, edge_index=edge_index, y=y))
 
        return dataset, scaler