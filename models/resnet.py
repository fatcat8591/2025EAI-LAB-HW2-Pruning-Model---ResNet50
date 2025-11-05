import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Bottleneck Block
# ----------------------


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        assert len(
            out_channels) == 3, "Bottleneck requires out_channels list of length 3"

        ################################################
        # Please replace ??? with the correct variable #
        # example: in_channels, out_channels[0], ...   #
        ################################################

        # conv1: 輸入是 Bottleneck 的input channel -> in_channels = 3 ， output是 cfg 中的第一個數 -> out_channels[0]
        self.conv1 = nn.Conv2d(
            in_channels, out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        # 根據conv1的輸出來設計conv2
        self.conv2 = nn.Conv2d(
            out_channels[0], out_channels[1], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        # 根據conv2的輸出來設計conv3
        self.conv3 = nn.Conv2d(
            out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out

# ----------------------
# ResNet
# ----------------------


class ResNet(nn.Module):
    def __init__(self, block, layers, cfg, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.current_cfg_idx = 0

        # Conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)  # 3x3,64,stride=1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)  # 3x3,maxpool,stride=2

        # Update cfg index
        # cfg[0]=64 is for conv1 ,  so we move to the next index
        self.current_cfg_idx += 1
        self.inplanes = 64

        # Layer1~Layer4
        # ResNet Acrhitecture圖片中的conv2_x
        self.layer1 = self._make_layer(block, 64, layers[0], cfg)
        # ResNet Acrhitecture圖片中的conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], cfg, stride=2)
        # ResNet Acrhitecture圖片中的conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], cfg, stride=2)
        # ResNet Acrhitecture圖片中的conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], cfg, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, num_classes)

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        #############################################################################
        # Figure out how to generate the correct layers and downsample based on cfg #
        #############################################################################
        downsample = None

        # 1. 取得第一個 Bottleneck 的通道數
        out_channels_1 = cfg[self.current_cfg_idx:self.current_cfg_idx+3]

        # 2. 檢查是否需要 Projection Shortcut (使用 "原始" 通道數)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # Downsample 層使用 "原始" (未剪枝) 的輸出通道數
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # 助教提示：downsample 沒有 BN
            )

        layers = []

        # 3. 建立第一個 Bottleneck
        layers.append(block(self.inplanes, planes,
                      out_channels_1, downsample, stride))

        # 4. 更新 "剪枝後" 的輸入通道數
        self.inplanes = out_channels_1[-1]
        self.current_cfg_idx += 3

        # 5. 建立後續的 "Identity" Bottlenecks
        for _ in range(1, blocks):
            out_channels_i = cfg[self.current_cfg_idx:self.current_cfg_idx+3]

            # 這裡 "強制" downsample=None，保持模組列表一致 (解決 AttributeError)
            layers.append(block(self.inplanes, planes,
                          out_channels_i, downsample=None, stride=1))
            self.inplanes = out_channels_i[-1]  # 更新 inplanes
            self.current_cfg_idx += 3

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet50(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*6 + \
              [512, 512, 2048]*3
    layers = [3, 4, 6, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)


def ResNet101(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*4 + \
              [256, 256, 1024]*23 + \
              [512, 512, 2048]*3
    layers = [3, 4, 23, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)


def ResNet152(num_classes=10, in_channels=3, cfg=None):
    # cfg: number of channels for conv1 and the three conv layers in each Bottleneck
    if cfg is None:
        cfg = [64] + \
              [64, 64, 256]*3 + \
              [128, 128, 512]*8 + \
              [256, 256, 1024]*36 + \
              [512, 512, 2048]*3
    layers = [3, 8, 36, 3]
    return ResNet(Bottleneck, layers, cfg, num_classes=num_classes, in_channels=in_channels)
