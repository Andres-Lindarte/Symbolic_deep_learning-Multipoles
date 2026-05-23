"""
Usage 
-----
     Python train.py --mode <potential|efield_mag|efield_vector>
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from datetime import datetime

from generate_data import MultipoleDataGenerator
from model import MultipoleGNN


def train_model(MODE: str = 'efield_vector'):
    # ------------------------------------------------------------------ #
    # 1. Hyperparameters                                                   #
    # ------------------------------------------------------------------ #
    EPOCHS        = 200 
    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-3
    NUM_SAMPLES   = 10_000 
    SPACE_SIZE    = 10.0
    TRAIN_RATIO   = 0.8

    # MODE controls what the GNN learns:
    #   'potential'    → V = k·q/r          scalar, output_dim=1
    #   'efield_mag'   → |E| = k·q/r²       scalar, output_dim=1
    #   'efield_vector'→ E = (Ex, Ey, Ez)   vector, output_dim=3

    MODE_CONFIG = {
        'potential':     {'target_col': 'target_V',                              'output_dim': 1},
        'efield_mag':    {'target_col': 'target_E_mag',                          'output_dim': 1},
        'efield_vector': {'target_col': ['target_Ex', 'target_Ey', 'target_Ez'], 'output_dim': 3},
    }

    if MODE not in MODE_CONFIG:
        raise ValueError(f"Unknown MODE='{MODE}'. Choose from: {list(MODE_CONFIG)}")

    TARGET_COL = MODE_CONFIG[MODE]['target_col']
    OUTPUT_DIM = MODE_CONFIG[MODE]['output_dim']

    # ------------------------------------------------------------------ #
    # 2. Data generation                                                   #
    # ------------------------------------------------------------------ #
    generator = MultipoleDataGenerator(num_samples=NUM_SAMPLES, space_size=SPACE_SIZE)

    if MODE == 'potential':
        df = generator.generate_monopole()
    else:
        df = generator.generate_monopole_efield()   # has Ex, Ey, Ez and E_mag

    print(f"[{MODE}] Dataset generated: {len(df)} samples.")

    split    = int(TRAIN_RATIO * len(df))
    df_train = df.iloc[:split].reset_index(drop=True)
    df_val   = df.iloc[split:].reset_index(drop=True)

    train_dataset, scaler = generator.df_to_pytorch_geometric(df_train, scaler=None,   target_col=TARGET_COL)
    val_dataset,   _      = generator.df_to_pytorch_geometric(df_val,   scaler=scaler, target_col=TARGET_COL)

    print(f"Train: {len(train_dataset)} graphs | Val: {len(val_dataset)} graphs")
    if OUTPUT_DIM == 1:
        col = TARGET_COL
        print(f"Scaler [{col}] — mean: {scaler[col]['mean']:.4f}, std: {scaler[col]['std']:.4f}")
    else:
        for col in TARGET_COL:
            print(f"  Scaler [{col}] — mean: {scaler[col]['mean']:.4f}, std: {scaler[col]['std']:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------ #
    # 3. Model, optimiser, loss                                            #
    # ------------------------------------------------------------------ #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultipoleGNN(node_features=5, hidden_dim=32, output_dim=OUTPUT_DIM).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training on : {device}")
    print(f"Output dim  : {OUTPUT_DIM}  {'(vector Ex,Ey,Ez)' if OUTPUT_DIM == 3 else '(scalar)'}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

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

            out = model(batch.x, batch.edge_index)   # [num_nodes_in_batch, OUTPUT_DIM]

            # Observer nodes are at odd positions in the flat node list
            obs_mask = torch.arange(out.size(0), device=device) % 2 == 1
            pred     = out[obs_mask]                 # [num_graphs, OUTPUT_DIM]
            target   = batch.y.view(-1, OUTPUT_DIM)  # [num_graphs, OUTPUT_DIM]

            assert pred.shape == target.shape, (
                f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
            )

            loss = criterion(pred, target)
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
                pred     = out[obs_mask]
                target   = batch.y.view(-1, OUTPUT_DIM)
                loss     = criterion(pred, target)
                total_val_loss += loss.item() * batch.num_graphs

        avg_val_loss = total_val_loss / len(val_dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == 0:
            marker = " ← best" if avg_val_loss == best_val_loss else ""
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f}{marker}"
            )

    # ------------------------------------------------------------------ #
    # 5. Save best checkpoint                                              #
    # ------------------------------------------------------------------ #
    model.load_state_dict(best_state)

    run_id    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_path = f"outputs/checkpoints/{MODE}_monopole_{run_id}.pth"

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'scaler':           scaler,
            'mode':             MODE,
            'target_col':       TARGET_COL,
            'node_features':    5,
            'hidden_dim':       32,
            'output_dim':       OUTPUT_DIM,
        },
        save_path,
    )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved → {save_path}")

    return model, train_loader, scaler, MODE


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train GNN to learn monopole potential or E-field.")
    parser.add_argument(
        '--mode',
        type=str,
        default='efield_vector',
        choices=['potential', 'efield_mag', 'efield_vector'],
        help="What the GNN should learn: 'potential' for V=k·q/r, 'efield_mag' for |E|=k·q/r², or 'efield_vector' for E=(Ex,Ey,Ez)."
    )
    args = parser.parse_args()
    trained_model, dataloader, scaler, mode = train_model(args.mode)
    print(f"\nNext step: python pysr_analysis.py --checkpoint <outputs/checkpoints/saved .pth file>")