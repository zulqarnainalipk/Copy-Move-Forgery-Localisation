"""DINOv2 encoder with a convolutional decoder for binary forgery masks."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Decoder(nn.Module):
    """Three 3x3 conv blocks with progressive upsampling and a 1x1 output convolution."""

    def __init__(self, in_channels=768):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(in_channels, 384, 3, padding=1), nn.ReLU(True), nn.Dropout2d(0.1))
        self.block2 = nn.Sequential(nn.Conv2d(384, 192, 3, padding=1), nn.ReLU(True), nn.Dropout2d(0.1))
        self.block3 = nn.Sequential(nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(True))
        self.out = nn.Conv2d(96, 1, 1)

    def forward(self, features, size):
        x = F.interpolate(self.block1(features), size=(74, 74), mode="bilinear", align_corners=False)
        x = F.interpolate(self.block2(x), size=(148, 148), mode="bilinear", align_corners=False)
        x = F.interpolate(self.block3(x), size=(296, 296), mode="bilinear", align_corners=False)
        return F.interpolate(self.out(x), size=size, mode="bilinear", align_corners=False)


class LinearHead(nn.Module):
    """Single 1x1 convolution on the token grid followed by bilinear upsampling."""

    def __init__(self, in_channels=768):
        super().__init__()
        self.out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, features, size):
        return F.interpolate(self.out(features), size=size, mode="bilinear", align_corners=False)


class ForgerySegmenter(nn.Module):
    """encoder_input='crop224' uses the standard resize-256 and centre-crop-224 preprocessing; 'full' feeds the whole image."""

    def __init__(self, encoder_dir, head="conv3", encoder_input="crop224", img_size=518):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_dir)
        self.encoder_input, self.img_size = encoder_input, img_size
        dim = self.encoder.config.hidden_size
        self.head = Decoder(dim) if head == "conv3" else LinearHead(dim)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def preprocess(self, x):
        if self.encoder_input == "crop224":
            x = F.interpolate(x, size=(256, 256), mode="bicubic", align_corners=False, antialias=True)
            x = x[:, :, 16:240, 16:240]
        return (x - self.mean) / self.std

    def encode(self, x):
        tokens = self.encoder(pixel_values=self.preprocess(x)).last_hidden_state[:, 1:]
        side = int(tokens.shape[1] ** 0.5)
        return tokens.permute(0, 2, 1).reshape(tokens.shape[0], -1, side, side)

    def forward(self, x):
        return self.head(self.encode(x), (self.img_size, self.img_size))

    def unfreeze_encoder(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True


def count_parameters(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
