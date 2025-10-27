import torch
import torch.optim as optim

from dataset import get_dataloaders 
from modules import SimpleUNet, DiceLoss 

def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            pred_pet = outputs[:, 0]  # Pet class probability from sigmoid
            #print the shape of pred_pet and masks for debugging
            # print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")
            loss = criterion(pred_pet, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Visualize predictions after each epoch (or every few epochs)
        if (epoch) % visualize_every == 0:
            show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

    print(" Training complete with enhanced U-Net!")
    return losses

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders()

    #for img, lab in test_dataloader:
    #    print(len(img))
    
    #exit()
    model = SimpleUNet(in_channels=1, out_channels=3, dropout_p=0.2)
    losses = train(model, train_loader, test_loader, epochs=3, lr=0.001, visualize_every=50)
