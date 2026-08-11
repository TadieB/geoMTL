"""
Classical deep-learning captioning baselines for Geo-MTL comparison.
Both take multi-temporal, multispectral image input and output text only
(no segmentation branch).
"""

import torch
import torch.nn as nn
from torchvision import models


# ============================================================================
# Encoders
# ============================================================================

class ResNetEncoder(nn.Module):
    """2D ResNet-50 encoder with early-fusion channel stacking.

    Multi-temporal input (T frames x C bands) is stacked into T*C channels
    and fed through a modified first conv layer.
    """

    def __init__(self, embed_size, num_frames=3, bands=6, pretrained=True):
        super().__init__()
        input_channels = num_frames * bands

        resnet = models.resnet50(pretrained=pretrained)

        if pretrained:
            original_weights = resnet.conv1.weight.data
            new_weights = torch.zeros(64, input_channels, 7, 7)
            for i in range(input_channels // 3):
                new_weights[:, i * 3:(i + 1) * 3, :, :] = original_weights
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1.weight.data = new_weights
        else:
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        B, T, C, H, W = images.shape
        images = images.view(B, T * C, H, W)
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        return self.bn(self.embed(features))


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3DEncoder(nn.Module):
    """3D U-Net encoder path, pooled and projected to a fixed embedding."""

    def __init__(self, embed_size, in_channels=6, base_features=32):
        super().__init__()
        f = base_features
        self.encoder1 = Conv3DBlock(in_channels, f)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.encoder2 = Conv3DBlock(f, f * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.encoder3 = Conv3DBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bottleneck = Conv3DBlock(f * 4, f * 8)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(f * 8, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        features = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        return self.bn(self.fc(features))


# ============================================================================
# Shared decoder
# ============================================================================

class DecoderRNN(nn.Module):
    """LSTM captioning decoder, conditioned on a single pooled image embedding."""

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1)
        inputs = torch.cat((features, embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        return self.linear(hiddens)
    
    def sample_with_penalty(self, features, max_len=40, repetition_penalty=1.5):
        """Greedy sampling with a repetition penalty, for fair comparison
        against Geo-MTL's penalized generation."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        generated = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            logits = self.linear(hiddens.squeeze(1))
            for token_id in generated:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
            predicted = logits.argmax(1)
            generated.append(predicted.item())
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return torch.stack(sampled_ids, 1)

    def sample(self, features, max_len=40):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        return torch.stack(sampled_ids, 1)


# ============================================================================
# Combined models
# ============================================================================

class ResNetLSTM(nn.Module):
    def __init__(self, embed_size=768, hidden_size=768, vocab_size=50257, num_layers=1, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(embed_size, num_frames=3, bands=6, pretrained=pretrained)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    def sample(self, images, max_len=40):
        return self.decoder.sample(self.encoder(images), max_len)


class UNet3DLSTM(nn.Module):
    def __init__(self, embed_size=768, hidden_size=768, vocab_size=50257, num_layers=1):
        super().__init__()
        self.encoder = UNet3DEncoder(embed_size, in_channels=6)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        return self.decoder(self.encoder(images), captions)

    def sample(self, images, max_len=40):
        return self.decoder.sample(self.encoder(images), max_len)
