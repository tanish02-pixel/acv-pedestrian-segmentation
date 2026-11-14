import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = conv_block(3, 16)
        self.encoder2 = conv_block(16, 32)
        self.encoder3 = conv_block(32, 64)
        self.bottleneck = conv_block(64, 128)
        self.decoder3 = conv_block(128 + 64, 64)
        self.decoder2 = conv_block(64 + 32, 32)
        self.decoder1 = conv_block(32 + 16, 16)
        self.classifier = nn.Conv2d(16, 1, 1)

    @staticmethod
    def upsample_and_join(features, skip):
        features = F.interpolate(features, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat((features, skip), dim=1)

    def forward(self, image):
        first = self.encoder1(image)
        second = self.encoder2(F.max_pool2d(first, 2))
        third = self.encoder3(F.max_pool2d(second, 2))
        features = self.bottleneck(F.max_pool2d(third, 2))
        features = self.decoder3(self.upsample_and_join(features, third))
        features = self.decoder2(self.upsample_and_join(features, second))
        features = self.decoder1(self.upsample_and_join(features, first))
        return self.classifier(features)


def segmentation_loss(logits, target, valid):
    pixel_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (pixel_loss * valid).sum() / valid.sum()
    probability = logits.sigmoid() * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    area = (probability + target).sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1) / (area + 1)).mean()
    return (bce + 1 - dice) / 2
