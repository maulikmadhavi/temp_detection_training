import torch.nn as nn

# Modified from https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py.
# In this version, Relu6 is replaced with Relu to make it ONNX compatible.


class InvertedResidual(nn.Module):
    def __init__(
        self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False
    ):
        super(InvertedResidual, self).__init__()
        re_lu = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in {1, 2}

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = (
                nn.Sequential(
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    re_lu(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
                if use_batch_norm
                else nn.Sequential(
                    # dw
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        3,
                        stride,
                        1,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    re_lu(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
            )
        elif use_batch_norm:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                re_lu(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                re_lu(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                re_lu(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                ),
                re_lu(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)
