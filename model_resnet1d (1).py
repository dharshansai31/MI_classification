import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, downsample=False):
        super().__init__()
        padding = kernel_size // 2
        s = 2 if downsample else stride

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=s, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if downsample or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class HighAccuracyECGNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            ResidualBlock1D(32, 32, kernel_size=7, downsample=False),
            ResidualBlock1D(32, 32, kernel_size=7, downsample=False),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(32, 64, kernel_size=7, downsample=True),
            ResidualBlock1D(64, 64, kernel_size=7, downsample=False),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=5, downsample=True),
            ResidualBlock1D(128, 128, kernel_size=5, downsample=False),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock1D(128, 256, kernel_size=5, downsample=True),
            ResidualBlock1D(256, 256, kernel_size=5, downsample=False),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
