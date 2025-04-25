import torch 
import torch.nn as nn


class YAMNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super().__init__()
        self.initial_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block1 = self._get_block_build(32, 64, 1)
        self.block2 = self._get_block_build(64, 128, 2)
        self.block3 = self._get_block_build(128, 128, 1)
        self.block4 = self._get_block_build(128, 256, 2)
        self.block5 = self._get_block_build(256, 256, 1)
        self.block6 = self._get_block_build(256, 512, 2)
        self.blocks_7_to_11 = nn.ModuleList([
        self._get_block_build(512, 512, 1) for _ in range(5)
        ])
        self.block12 = self._get_block_build(512, 1024, 2)
        self.block13 = self._get_block_build(1024, 1024, 1)
        self.final_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2)
        )
        self.fully_connected_layer = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _get_block_build(self, in_channel, out_channel, stride):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, 
                      kernel_size=3, stride=stride, groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        x = self.initial_conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        for block in self.blocks_7_to_11:
            x = block(x)
        x = self.block12(x)
        x = self.block13(x) 
        x = self.final_layer(x)
        x = torch.flatten(x, 1)
        x = self.fully_connected_layer(x)
        x = self.softmax(x)

        return x 
