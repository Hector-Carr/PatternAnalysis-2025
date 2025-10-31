import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import get_dataloaders 
from modules import SimpleUNet, DiceCELoss 

def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
    save_path: str = "unet3d_checkpoint.pt",
    criterion=None,
    optimizer=None,
):
    """
    Generic training loop for a 3D U-Net model.
    
    Args:
        model: 3D U-Net model (nn.Module)
        train_loader: DataLoader for training set
        val_loader: optional DataLoader for validation set
        num_epochs: number of epochs to train
        lr: learning rate
        device: 'cuda' or 'cpu'
        save_path: path to save best model checkpoint
        criterion: loss function (default: BCEWithLogitsLoss)
        optimizer: optimizer (default: Adam)
    """

    # Defaults
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()  # good for segmentation
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # if model exist keep training
    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

    model = model.to(device)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        # --- Training ---
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.4f}")

        # --- Validation (optional) ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            print(f"Val Loss:   {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved new best model to {save_path}")
        else:
            # Save last model if no validation
            torch.save(model.state_dict(), save_path)

    print("\nTraining complete.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader = get_dataloaders(train_val=True)
    model = SimpleUNet(in_channels=1, out_channels=6, dropout_p=0.2)

    losses = train(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=10,
        lr=0.001,
        device=device,
        criterion=DiceCELoss(),
        save_path="model.pt"
    )
