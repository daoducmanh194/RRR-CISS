import torch.nn as nn


class _SeparableConv(nn.Module):

    def __init__(self, *args, **kwargs):
        super(_SeparableConv, self).__init__()

        self.dwconv = None
        self.dwconv_normalization = None
        self.dwconv_activation = None

        self.pwconv = None
        self.pwconv_normalization = None
        self.pwconv_activation = None

    def forward(self, x):
        assert self.dwconv is not None and self.pwconv is not None, (
            "Depthwise Convolution and/or Pointwise Convolution is/are not "
            "implemented yet."
        )

        x  = self.dwconv(x)

        if self.dwconv_normalization is not None:
            x = self.dwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.dwconv_activation(x)

        x = self.pwconv(x)

        if self.pwconv_normalization is not None:
            x = self.pwconv_normalization(x)

        if self.pwconv_activation is not None:
            x = self.pwconv_activation(x)

        return x
