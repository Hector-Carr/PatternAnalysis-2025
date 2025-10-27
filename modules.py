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
        self.enc1 = self._conv_block(in_channels, 16, dropout_p)
        self.enc2 = self._conv_block(16, 32, dropout_p)
        self.enc3 = self._conv_block(32, 64, dropout_p)
        self.enc4 = self._conv_block(64, 128, dropout_p)

        # Decoder (upsampling)
        self.dec4 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec3 = self._conv_block(64 + 32, 32, dropout_p)
        self.dec2 = self._conv_block(32 + 16, 16, dropout_p)
        self.dec1 = nn.Conv3d(16, out_channels, 1)

        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        """Conv block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout"""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
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
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], 1))  # 64x64x32
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], 1))  # 128x128x64
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1))  # 256x256x128
        out = self.dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        """
        Multi-class Dice Loss for 3D segmentation (one-hot targets).

        Args:
            smooth (float): Smoothing term to prevent division by zero.
            ignore_index (int, optional): Channel index to ignore (e.g. background).
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (N, C, D, H, W) — raw network outputs.
            targets: Tensor of shape (N, C, D, H, W) — one-hot encoded ground truth.
        Returns:
            dice_loss (float): Mean dice loss over classes.
        """
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities

        if self.ignore_index is not None:
            mask = torch.ones_like(targets, dtype=torch.bool)
            mask[:, self.ignore_index] = False
            probs = probs * mask
            targets = targets * mask

        # Compute per-class Dice
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        cardinality = torch.sum(probs + targets, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        if self.ignore_index is not None:
            valid_classes = torch.ones(dice_score.shape[0], device=logits.device, dtype=torch.bool)
            valid_classes[self.ignore_index] = False
            dice_score = dice_score[valid_classes]

        # Dice loss = 1 - mean dice coefficient
        return 1 - dice_score.mean()
