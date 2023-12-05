"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torchvision.models as models_ts

# from torchvision.models.resnet import resnet50
# from torchvision.models.mobilenet import mobilenet_v2, InvertedResidual
from vitg.mobilenetSSD.src.model_mobilenetv2_local import InvertedResidual, mobilenet_v2


class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            print("item box shape")
            print(len(src))

            ret.append(
                (
                    l(s).view(s.size(0), 4, -1),
                    c(s).view(s.size(0), self.num_classes, -1),
                )
            )

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs


feature_maps = {}


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = mobilenet_v2(pretrained=True).features
        self.feature_extractor[14].conv[0][2].register_forward_hook(
            self.get_activation()
        )

    def get_activation(self):
        def hook(self, input, output):
            feature_maps[0] = output.detach()

        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


class MobileNetV3s(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = models_ts.mobilenet_v3_small(pretrained=True).features
        self.feature_extractor[11].block[0][2].register_forward_hook(
            self.get_activation()
        )

    def get_activation(self):
        def hook(self, input, output):
            feature_maps[0] = output.detach()

        return hook

    def forward(self, x):
        x = self.feature_extractor(x)
        return feature_maps[0], x


def SeperableConv2d(in_channels, out_channels, kernel_size=3):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
        ),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def StackedSeperableConv2d(ls_channels, multiplier):
    out_channels = 6 * multiplier
    layers = [
        SeperableConv2d(in_channels=in_channels, out_channels=out_channels)
        for in_channels in ls_channels
    ]
    layers.append(
        nn.Conv2d(in_channels=ls_channels[-1], out_channels=out_channels, kernel_size=1)
    )
    return nn.ModuleList(layers)


class SSDLite(Base):
    def __init__(self, backbone=MobileNetV2(), num_classes=81, width_mul=1.0):
        super(SSDLite, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes

        self.additional_blocks = nn.ModuleList(
            [
                InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
                InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
                InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
                InvertedResidual(256, 64, stride=2, expand_ratio=0.25),
            ]
        )
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)
        self.init_weights()

    def forward(self, x):
        y, x = self.feature_extractor(x)
        detection_feed = [y, x]
        for l in self.additional_blocks:
            print("x shape")
            print(x.shape)
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs


class SSDLitev3(Base):
    def __init__(self, backbone=MobileNetV3s(), num_classes=81, width_mul=1.0):
        super(SSDLitev3, self).__init__()
        self.feature_extractor = backbone
        self.num_classes = num_classes

        self.additional_blocks = nn.ModuleList(
            [
                InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
                InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
                InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
                InvertedResidual(256, 64, stride=2, expand_ratio=0.25),
            ]
        )
        header_channels = [round(576 * width_mul), 1280, 512, 256, 256, 64]
        self.loc = StackedSeperableConv2d(header_channels, 4)
        self.conf = StackedSeperableConv2d(header_channels, self.num_classes)
        self.init_weights()

    def forward(self, x):
        y, x = self.feature_extractor(x)
        detection_feed = [y, x]
        for l in self.additional_blocks:
            print("x shape")
            print(x.shape)
            x = l(x)
            detection_feed.append(x)
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)
        return locs, confs
