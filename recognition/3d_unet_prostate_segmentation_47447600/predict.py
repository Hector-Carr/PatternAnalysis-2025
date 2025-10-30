import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders 
from modules import SimpleUNet, DiceLoss

def test(
    model,
    test_loader,
    checkpoint_path: str,
    device: str = "cuda",
    criterion=None,
):
    """
    Evaluate a trained 3D U-Net model on a test set.

    Args:
        model: 3D U-Net model (nn.Module)
        test_loader: DataLoader for test data
        checkpoint_path: path to saved model weights (.pt file)
        device: 'cuda' or 'cpu'
        criterion: loss function for evaluation (default BCEWithLogitsLoss)
        compute_dice: if True, computes average Dice score

    Returns:
        (avg_loss, avg_dice)
    """
    # Default loss
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    total_dice = 0.0
    all_dice = []
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_dice += loss.item() * inputs.size(0)
            all_dice.append(loss.item())

            num_batches += inputs.size(0)

    avg_dice = total_dice / num_batches

    print(f"\nTest DiceLoss: {avg_dice:.4f}")
    print(f"\nAll DiceLoss: {all_dice}")
    print(f"\nMax DiceLoss: {max(all_dice)}")

    return avg_dice, all_dice

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_dataloaders()
    model = SimpleUNet(in_channels=1, out_channels=6, dropout_p=0.2)

    losses = test(
        model,
        test_loader,
        checkpoint_path="testing.pt",
        device=device,
        criterion=DiceLoss()
    )
