import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class MultipoleDataGenerator:
    """
    Class to generate synthetic datasets for training a GNN to learn electrostatic
    potentials of monopoles, dipoles, and quadrupoles.

    Each sample consists of:
    - Source node: (x, y, z, q)
    - Observer node: (x, y, z, q=0)
    - Distance r between source and observer (explicit feature for PySR)
    - Target potential V at the observer due to the source configuration
    """

    def __init__(self, num_samples=10000, space_size=10.0, k_constant=1.0):
        self.num_samples = num_samples
        self.space_size = space_size
        self.k = k_constant

    def _generate_random_position(self):
        """Generate a random 3D position."""
        return np.random.uniform(-self.space_size, self.space_size, 3)

    def _generate_random_charge(self, min_val=-5.0, max_val=5.0):
        """
        Generate a random charge avoiding values near zero.

        FIX: np.sign(0) == 0, so if q is exactly 0.0 the original code left
        it unchanged.  We now resample instead of nudging, which also gives a
        cleaner uniform distribution over the non-zero range.
        """
        while True:
            q = np.random.uniform(min_val, max_val)
            if abs(q) >= 0.1:
                return q

    def generate_monopole(self):
        """Generate training data for the potential of an electric monopole."""
        data = []
        samples_generated = 0

        while samples_generated < self.num_samples:
            # Source node (node 0)
            pos_source = self._generate_random_position()
            q_source = self._generate_random_charge()

            # Observer node (node 1) — test charge q=0 so it doesn't perturb the field
            pos_obs = self._generate_random_position()
            q_obs = 0.0

            # Displacement and distance
            delta_pos = pos_obs - pos_source
            r = np.linalg.norm(delta_pos)

            if r < 0.1:          # Avoid singularities
                continue

            # Ground-truth potential  V = k * q / r
            V = self.k * (q_source / r)

            data.append({
                'source_x': pos_source[0],
                'source_y': pos_source[1],
                'source_z': pos_source[2],
                'source_q': q_source,
                'obs_x':    pos_obs[0],
                'obs_y':    pos_obs[1],
                'obs_z':    pos_obs[2],
                'obs_q':    q_obs,
                'distance_r': r,   # explicit distance — helps PySR discover 1/r
                'target_V': V,
            })
            samples_generated += 1

        return pd.DataFrame(data)

    def df_to_pytorch_geometric(self, df, scaler=None):
        """
        Convert a DataFrame of multipole data into a list of PyTorch Geometric graphs.

        Graph layout
        ------------
        Node 0 — source:   [x, y, z, q,  r]   (r is the distance to the observer)
        Node 1 — observer: [x, y, z, q=0, r]

        The directed edge goes 0 → 1, i.e. information flows from source to observer.

        Parameters
        ----------
        df : pd.DataFrame
        scaler : dict or None
            If provided, must contain 'V_mean' and 'V_std' so that targets are
            normalised consistently between train and validation sets.
            Pass None on the *training* set to compute stats from scratch;
            pass the returned scaler to subsequent sets.

        Returns
        -------
        dataset : list[Data]
        scaler  : dict  {'V_mean': float, 'V_std': float}
        """
        # --- Target normalisation (stabilises training when V spans many orders) ---
        if scaler is None:
            V_mean = df['target_V'].mean()
            V_std  = df['target_V'].std()
            if V_std < 1e-8:          # constant dataset edge case
                V_std = 1.0
            scaler = {'V_mean': V_mean, 'V_std': V_std}
        else:
            V_mean = scaler['V_mean']
            V_std  = scaler['V_std']

        dataset = []

        for _, row in df.iterrows():
            r = row['distance_r']

            # Node features: [x, y, z, q, r]
            # FIX: r is now an explicit node feature so PySR can discover 1/r directly.
            node_source = [row['source_x'], row['source_y'], row['source_z'], row['source_q'], r]
            node_obs    = [row['obs_x'],    row['obs_y'],    row['obs_z'],    row['obs_q'],    r]

            x = torch.tensor([node_source, node_obs], dtype=torch.float)

            # Directed edge: source (0) → observer (1)
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)

            # Normalised target
            V_norm = (row['target_V'] - V_mean) / V_std
            y = torch.tensor([[V_norm]], dtype=torch.float)

            dataset.append(Data(x=x, edge_index=edge_index, y=y))

        return dataset, scaler
