# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Original Author: Wei Yang
"""


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if isinstance(x, tuple):
            x, features = x
            features.append(self.bn1(x))
        else:
            features = [self.bn1(x)]

        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out, features


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,dropout_p=0.5):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # # 1st conv before any network block
        # self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=7, stride=2,
        #                        padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.bn0 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(num_classes, 2)
        self.last_linear.weight.data.normal_(0, 0.01)
        self.last_linear.bias.data.fill_(0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.block1)
        feat_m.append(self.block2)
        feat_m.append(self.block3)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=True):
        # out = self.conv1(x)
        x = self.conv1(x)
        x = self.bn0(x)
        # f0=copy.deepcopy(x)
        x = self.relu(x)
        out = self.maxpool(x)

        out, f1 = self.block1(out)
        f1_act = [self.relu(f) for f in f1]
        out, f2 = self.block2(out)
        f2_act = [self.relu(f) for f in f2]
        out, f3 = self.block3(out)
        f3_act = [self.relu(f) for f in f3]

        f4 = self.bn1(out)
        out = self.relu(f4)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.nChannels)
        f5 = out
        feat = self.fc(out)
        feat_norm = F.normalize(feat, dim=1)
        x = self.dropout(feat)
        x = self.last_linear(x)

        if is_feat:
            return f1_act + f2_act + f3_act + [self.relu(f4), f5], x, feat, feat_norm
        else:
            return out


def wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model


def wrn40x2(**kwargs):
    model = WideResNet(depth=40, widen_factor=2, **kwargs)
    return model


def wrn40x1(**kwargs):
    model = WideResNet(depth=40, widen_factor=1, **kwargs)
    return model


def wrn28x2(**kwargs):
    model = WideResNet(depth=28, widen_factor=2, **kwargs)
    return model


def wrn28x1(**kwargs):
    model = WideResNet(depth=28, widen_factor=1, **kwargs)
    return model


def wrn16x2(**kwargs):
    model = WideResNet(depth=16, widen_factor=2, **kwargs)
    return model


def wrn16x1(**kwargs):
    model = WideResNet(depth=16, widen_factor=1, **kwargs)
    return model
