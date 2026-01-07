import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloaders 
from modules import SimpleUNet
from modules import Diceloss 

def test(
    model,
    test_loader,
    checkpoint_path,
    unnormalised_loader = None,
    device="cuda",
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
    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    total_dice = 0.0
    all_dice = []
    all_class_dice = []
    num_batches = 0

    non_ave_ds = (lambda x, y: Diceloss._dice(x, y, smoothing=1e-5, mean=False))

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            dice = criterion._dice(outputs, targets)
            total_dice += dice
            all_dice.append(dice)
            all_class_dice += list(non_ave_ds(outputs, targets))

            if unnormalised_loader:
                plot_images(unnormalised_loader.dataset[num_batches][0], outputs, targets, num_batches, dice)

            num_batches += inputs.size(0)

    avg_dice = total_dice / num_batches

    print(f"\nTest Dice Coeficient: {avg_dice:.4f}")
    print(f"\nAll Dice Coeficients: {all_dice}")
    print(f"\nMin Dice Coeficient: {min(all_dice)}")
    print(f"\nMin Dice Coeficient in any class: {min(all_class_dice)}")

    return avg_dice, all_dice, all_class_dice

def plot_images(inputs, preds, targets, batch, dice):
    # only plot if batch size is one
    if inputs.size(0) != 1:
        return

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # find index to plot over
    i = inputs[0].size(-1)//2

    # plot input
    axes[0].imshow(inputs[0][:,:,i].cpu().numpy(), cmap="inferno")
    axes[0].axis('off')

    # plot target
    axes[1].imshow(torch.argmax(targets[0], 0)[:,:,i].cpu().numpy(), cmap="inferno")
    axes[1].axis('off')

    # plot prediction
    axes[2].imshow(torch.argmax(preds[0], 0)[:,:,i].cpu().numpy(), cmap="inferno")
    axes[2].axis('off')

    plt.tight_layout()
    plt.title(f"Test {batch}, dice coeficient={dice}")

    # Save the figure to file
    plt.savefig(f"images/comparison_{batch}_d{dice}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = get_dataloaders(test=True)
    unnormalised_loader = get_dataloaders(test=True, normalised=False)
    model = SimpleUNet(in_channels=1, out_channels=6, dropout_p=0.2)

    print(test(
        model,
        test_loader,
        checkpoint_path="model.pt",
        unnormalised_loader = unnormalised_loader, 
        device=device,
        criterion=DiceLoss()
    ))
