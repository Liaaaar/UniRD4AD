from typing import Any, Callable, List, Optional, Type, Union

import torch
from torch import Tensor
import torch.nn as nn

__all__ = [
    "DecoderResNet",
    "de_resnet18",
    "de_resnet34",
    "de_resnet50",
    "de_wide_resnet50_2",
    "de_wide_resnet101_2",
]


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=2,
        stride=stride,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 is not supported in BasicBlock")

        self.conv1 = (
            deconv2x2(inplanes, planes, stride)
            if stride == 2
            else conv3x3(inplanes, planes, stride)
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.upsample is not None:
            identity = self.upsample(x)

        out = self.relu(out + identity)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = (
            deconv2x2(width, width, stride, groups, dilation)
            if stride == 2
            else conv3x3(width, width, stride, groups, dilation)
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.upsample is not None:
            identity = self.upsample(x)

        out = self.relu(out + identity)
        return out


class DecoderResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element list"
            )

        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1
        self.inplanes = 512 * block.expansion

        base_channels = 64 * block.expansion
        stage2_channels = base_channels * 2
        stage3_channels = base_channels * 4
        latent_channels = base_channels * 8

        self.deconv1 = deconv2x2(latent_channels, stage3_channels, stride=2)
        self.deconv2 = deconv2x2(stage3_channels, stage2_channels, stride=2)
        self.conv1 = conv3x3(stage3_channels * 2, stage3_channels)
        self.conv2 = conv3x3(stage2_channels * 2, stage2_channels)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)
                elif isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> List[Tensor]:
        feature_a = self.layer1(x)

        x = self.deconv1(x)
        feature_b = self.layer2(self.conv1(torch.cat([feature_a, x], dim=1)))

        x = self.deconv2(x)
        feature_c = self.layer3(self.conv2(torch.cat([feature_b, x], dim=1)))

        return [feature_c, feature_b, feature_a]


def _build_decoder(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> DecoderResNet:
    del progress
    if pretrained:
        raise ValueError("Pretrained decoder weights are not provided.")
    return DecoderResNet(block, layers, **kwargs)


def de_resnet18(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> DecoderResNet:
    return _build_decoder(BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def de_resnet34(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> DecoderResNet:
    return _build_decoder(BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_resnet50(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> DecoderResNet:
    return _build_decoder(Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_wide_resnet50_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> DecoderResNet:
    kwargs["width_per_group"] = 64 * 2
    return _build_decoder(Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def de_wide_resnet101_2(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> DecoderResNet:
    kwargs["width_per_group"] = 64 * 2
    return _build_decoder(Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
