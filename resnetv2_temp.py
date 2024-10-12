# Attention-based Feature-level Distillation
# Original Source : https://github.com/HobbitLong/RepDistiller

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


import torch.utils.model_zoo as model_zoo
import collections

__all__ = ['ResNet', 'resnet18_', 'resnet34_', 'resnet50_', 'resnet101_',
           'resnet152_']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []
        x = self.relu(x)
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        out += self.shortcut(identity)
        # out += self.shortcut(x)

        return out, features + [out]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if isinstance(x, tuple):
            x, features = x
        else:
            features = []
        identity = x
        x = self.relu(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # out += identity
        out = self.relu(out)
        out += self.shortcut(x)
        return out, features + [out]


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=50, zero_init_residual=False,dropout_p=0.5):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.softmax = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(num_classes, 2)
        self.last_linear.weight.data.normal_(0, 0.01)
        self.last_linear.bias.data.fill_(0.0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    #     norm_layer = self._norm_layer
    #     downsample = None
    #     previous_dilation = self.dilation
    #     if dilate:
    #         self.dilation *= stride
    #         stride = 1
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             conv1x1(self.inplanes, planes * block.expansion, stride),
    #             norm_layer(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
    #                         self.base_width, previous_dilation, norm_layer))
    #     self.inplanes = planes * block.expansion
    #     for _ in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, groups=self.groups,
    #                             base_width=self.base_width, dilation=self.dilation,
    #                             norm_layer=norm_layer))

    #     return nn.Sequential(*layers)

    def forward(self, x, is_feat=True):
        # x = self.bn1(self.conv1(x))
        # f0 = x

        x = self.conv1(x)
        x_ = self.bn1(x)
        f0=x_
        x = self.relu(x_)
        x = self.maxpool(x)

        x, f1 = self.layer1(x)
        f1_act = [self.relu(f) for f in f1]
        x, f2 = self.layer2(x)
        f2_act = [self.relu(f) for f in f2]
        x, f3 = self.layer3(x)
        f3_act = [self.relu(f) for f in f3]
        x, f4 = self.layer4(x)
        f4_act = [self.relu(f) for f in f4]


        out = self.avgpool(self.relu(x))
        out = out.view(out.size(0), -1)
        f5 = out
        feat = self.linear(out)

        feat_norm = F.normalize(feat, dim=1)
        x = self.dropout(feat)
        x = self.last_linear(x)
        # print('[self.relu(f0)] + f1_act + f2_act + f3_act + [f4]',[self.relu(f0)] + f1_act + f2_act + f3_act + [f4])
        if is_feat:
            return [self.relu(f0)] + f1_act + f2_act + f3_act + f4_act + [f5], x, feat,feat_norm
        else:
            return out

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls[arch], model_dir='.')
        fsd = collections.OrderedDict()
        res_iter = state_dict.items()
        for i in range(len(res_iter)):
            temp_key = list(res_iter)[i][0]
            if 'fc' in temp_key:
                continue
            fsd[temp_key] = list(res_iter)[i][1]
        model.load_state_dict(fsd, strict=False)
    return model


def resnet18_(pretrained=True, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_(pretrained=True, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_(pretrained=True, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_(pretrained=True, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152_(pretrained=True, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

# # def resnet18(**kwargs):
# def resnet18(pretrained=False, progress=True, **kwargs):
    
#     return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
#                    **kwargs)
#     # return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


# def resnet34(**kwargs):
#     return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


# def resnet50(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


# def resnet101(**kwargs):
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


# def resnet152(**kwargs):
#     return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

# if __name__ == '__main__':
#     model  = resnet50_(pretrained=True)
#     print(model)