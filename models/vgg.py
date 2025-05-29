import torch.nn as nn


class VGGConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if pool:
            self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x if not hasattr(self, "maxpool") else self.maxpool(x)
        return x


class CVGG13(nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            VGGConvBlock(3, 64, pool=False),
            VGGConvBlock(64, 64, pool=True),
            VGGConvBlock(64, 128, pool=False),
            VGGConvBlock(128, 128, pool=True),
            VGGConvBlock(128, 256, pool=False),
            VGGConvBlock(256, 256, pool=True),
            VGGConvBlock(256, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
            VGGConvBlock(512, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG13(nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            VGGConvBlock(3, 64, pool=False),
            VGGConvBlock(64, 64, pool=True),
            VGGConvBlock(64, 128, pool=False),
            VGGConvBlock(128, 128, pool=True),
            VGGConvBlock(128, 256, pool=False),
            VGGConvBlock(256, 256, pool=True),
            VGGConvBlock(256, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
            VGGConvBlock(512, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            VGGConvBlock(3, 64, pool=False),
            VGGConvBlock(64, 64, pool=True),
            VGGConvBlock(64, 128, pool=False),
            VGGConvBlock(128, 128, pool=True),
            VGGConvBlock(128, 256, pool=False),
            VGGConvBlock(256, 256, pool=False),
            VGGConvBlock(256, 256, pool=True),
            VGGConvBlock(256, 512, pool=False),
            VGGConvBlock(512, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
            VGGConvBlock(512, 512, pool=False),
            VGGConvBlock(512, 512, pool=False),
            VGGConvBlock(512, 512, pool=True),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
