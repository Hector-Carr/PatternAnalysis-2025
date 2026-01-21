import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    """
    Simplified U-Net for demo purposes with batch normalization, LeakyReLU, dropout and sigmoid activation.
    Modified for 3D data
    """

    def __init__(self, in_channels=1, out_channels=6, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 8, 16, dropout_p)
        self.enc2 = self._conv_block(16, 32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, 256, dropout_p)
        self.enc4 = self._conv_block(256, 256, 512, dropout_p)

        # Decoder (upsampling)
        self.dec4 = self._conv_block(512 + 256, 256, 256, dropout_p)
        self.dec3 = self._conv_block(256 + 64, 128, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 16, 32, 32, dropout_p)
        self.dec1 = nn.Conv3d(32, out_channels, 1)

        self.pool = nn.MaxPool3d(2, stride=2)
        self.up1 = nn.ConvTranspose3d(512, 512, 2, 2)
        self.up2 = nn.ConvTranspose3d(256, 256, 2, 2)
        self.up3 = nn.ConvTranspose3d(64, 64, 2, 2)
        self.out_act = nn.Softmax(dim=1)  # Softmax activation for final output

    def _conv_block(self, in_ch, mid_ch, out_ch, dropout_p=0.2):
        """Conv block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout"""
        return nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # 256x256x128
        e2 = self.enc2(self.pool(e1))  # 128x128x64
        e3 = self.enc3(self.pool(e2))  # 64x64x32
        e4 = self.enc4(self.pool(e3))  # 32x32x16

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up1(e4), e3], 1))  # 64x64x32
        d3 = self.dec3(torch.cat([self.up2(d4), e2], 1))  # 128x128x64
        d2 = self.dec2(torch.cat([self.up3(d3), e1], 1))  # 256x256x128
        out = self.dec1(d2)

        # Apply softmax activation to final output
        out = self.out_act(out)

        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Args:
            weight (Tensor, optional): Class weights for CrossEntropyLoss.
            dice_weight (float): Weight of Dice loss component.
            ce_weight (float): Weight of Cross-Entropy loss component.
            smooth (float): Smoothing term to avoid division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Computes per-class and mean Dice coefficients for multi-class segmentation.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (N, C, ...) with probabilities or logits.
            target (torch.Tensor): Ground truth tensor of shape (N, ...) with class indices.
            epsilon (float): Small constant for numerical stability.

        Returns:
            dice_per_class (torch.Tensor): Dice score for each class (C,).
            mean_dice (torch.Tensor): Mean Dice score across classes.
        """
        # dice score
        dice = self._dice(pred, target, smoothing=self.smooth)
        
        return 1 - dice

    def _dice(self, pred, target, num_classes=6, smoothing=1e-6, mean=True):
        """
        calculate the dice coefficient for any two images of the same dimensions
        """
        # Compute Dice per class
        intersection = torch.sum(pred * target, (0, 2, 3, 4))
        union = torch.sum(pred + target, (0, 2, 3, 4))

        dice = (2.0 * intersection + smoothing) / (union + smoothing)
        
        if mean:
            return dice.mean()
        else:
            return dice

