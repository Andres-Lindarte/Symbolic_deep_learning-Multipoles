import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datetime import datetime

from generate_data import MultipoleDataGenerator
from model import MultipoleGNN


def train_model():
    # ------------------------------------------------------------------ #
    # 1. Hyperparameters                                                   #
    # ------------------------------------------------------------------ #
    EPOCHS        = 50
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-3
    NUM_SAMPLES   = 10_000
    SPACE_SIZE    = 5.0
    TRAIN_RATIO   = 0.8

    # ------------------------------------------------------------------ #
    # 2. Data generation                                                   #
    # ------------------------------------------------------------------ #
    generator = MultipoleDataGenerator(num_samples=NUM_SAMPLES, space_size=SPACE_SIZE)
    df = generator.generate_monopole()
    print(f"Dataset generated: {len(df)} samples.")

    # Train / validation split at DataFrame level so we can compute the
    # normalisation scaler only from the training portion.
    split = int(TRAIN_RATIO * len(df))
    df_train = df.iloc[:split].reset_index(drop=True)
    df_val   = df.iloc[split:].reset_index(drop=True)

    # FIX: scaler is computed on train set and reused for validation,
    # preventing data leakage and ensuring consistent normalisation.
    train_dataset, scaler = generator.df_to_pytorch_geometric(df_train, scaler=None)
    val_dataset,   _      = generator.df_to_pytorch_geometric(df_val,   scaler=scaler)

    print(f"Train: {len(train_dataset)} graphs | Val: {len(val_dataset)} graphs")
    print(f"Target normalisation — mean: {scaler['V_mean']:.4f}, std: {scaler['V_std']:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------ #
    # 3. Model, optimiser, loss                                            #
    # ------------------------------------------------------------------ #
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # node_features=5 because we added distance r as an explicit feature
    model     = MultipoleGNN(node_features=5, hidden_dim=32).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------ #
    # 4. Training loop                                                     #
    # ------------------------------------------------------------------ #
    best_val_loss = float('inf')
    best_state    = None

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)   # [num_nodes_in_batch, 1]

            # Each graph has exactly 2 nodes (source=even, observer=odd).
            # FIX: select observer indices robustly using batch.batch so this
            # works even if graphs ever have different numbers of nodes.
            # For the current 2-node layout: observer nodes are at odd positions.
            obs_mask = torch.arange(out.size(0), device=device) % 2 == 1
            pred_V   = out[obs_mask]                 # [num_graphs, 1]

            # FIX: ensure shapes match explicitly before computing loss
            target_V = batch.y.view(-1, 1)           # [num_graphs, 1]
            assert pred_V.shape == target_V.shape, (
                f"Shape mismatch: pred {pred_V.shape} vs target {target_V.shape}"
            )

            loss = criterion(pred_V, target_V)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(train_dataset)

        # --- Validate ---
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(device)
                out      = model(batch.x, batch.edge_index)
                obs_mask = torch.arange(out.size(0), device=device) % 2 == 1
                pred_V   = out[obs_mask]
                target_V = batch.y.view(-1, 1)
                loss     = criterion(pred_V, target_V)
                total_val_loss += loss.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_dataset)

        # Track best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss:   {avg_val_loss:.6f}"
                + (" ← best" if avg_val_loss == best_val_loss else "")
            )

    # ------------------------------------------------------------------ #
    # 5. Restore best weights and save                                     #
    # ------------------------------------------------------------------ #
    model.load_state_dict(best_state)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"monopole_{run_id}_multipole_gnn.pth"
    torch.save(
        {'model_state_dict': model.state_dict(), 'scaler': scaler},
        save_path,
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model + scaler saved to '{save_path}'")

    return model, train_loader, scaler


# ------------------------------------------------------------------ #
# 6. PySR analysis scaffold (run after training)                       #
# ------------------------------------------------------------------ #

def extract_messages_for_pysr(model, loader, device, max_samples=5000):
    """
    Run the trained GNN in inference mode and collect the raw edge-MLP
    outputs together with their input features.

    Returns
    -------
    X : np.ndarray  shape [N, 10]  — [src_x, src_y, src_z, src_q, src_r,
                                       obs_x, obs_y, obs_z, obs_q, obs_r]
    y : np.ndarray  shape [N]      — edge_mlp output (message)

    Usage with PySR
    ---------------
        import pysr
        model_sr = pysr.PySRRegressor(niterations=40, binary_operators=["+", "*", "/"])
        model_sr.fit(X, y)
        print(model_sr)
    """
    import numpy as np

    model.eval()
    X_list, y_list = [], []
    collected = 0

    with torch.no_grad():
        for batch in loader:
            if collected >= max_samples:
                break
            batch = batch.to(device)
            msgs  = model.expose_messages(batch.x, batch.edge_index)   # [E, 1]

            src_idx = batch.edge_index[0]
            obs_idx = batch.edge_index[1]
            edge_X  = torch.cat([batch.x[src_idx], batch.x[obs_idx]], dim=1)  # [E, 10]

            X_list.append(edge_X.cpu().numpy())
            y_list.append(msgs.squeeze(-1).cpu().numpy())
            collected += edge_X.size(0)

    X = np.concatenate(X_list, axis=0)[:max_samples]
    y = np.concatenate(y_list, axis=0)[:max_samples]
    return X, y


if __name__ == "__main__":
    trained_model, dataloader, scaler = train_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_pysr, y_pysr = extract_messages_for_pysr(trained_model, dataloader, device)

    print(f"\nPySR design matrix ready: X shape={X_pysr.shape}, y shape={y_pysr.shape}")
    print("Feature columns: [src_x, src_y, src_z, src_q, src_r, obs_x, obs_y, obs_z, obs_q, obs_r]")
    print("Run PySR on (X_pysr, y_pysr) to discover the symbolic expression for V.")
