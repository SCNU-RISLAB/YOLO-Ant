import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DSLK(nn.Module):
    #
    def __init__(self, c1, c2, n=1, expansion=2, add_identity=True, large_kernel=9, small_kernel=3, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(CMBlock(c_, c_, large_kernel, small_kernel, expansion, add_identity) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class LargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 small_kernel):
        super(LargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        self.large_conv = nn.Sequential(
            # nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding='same'),
            nn.Conv2d(in_channels,in_channels, kernel_size=kernel_size, groups=in_channels, padding=(kernel_size-1)//2),
            nn.GELU(),
            nn.BatchNorm2d(in_channels))
        if small_kernel is not None:
            assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
            self.small_conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=small_kernel, groups=in_channels, padding=(small_kernel-1)//2),
            nn.GELU(),
            nn.BatchNorm2d(in_channels))

    def forward(self, inputs):
        out = self.large_conv(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        else:
            out = out + inputs
        return out

class CMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, large_kernel, small_kernel, expansion=2,add_identity=True):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.large_kernel = LargeKernelConv(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=large_kernel, small_kernel=small_kernel)
        self.cv1 = nn.Sequential(
            nn.Conv2d(out_channels, hidden_channels,kernel_size=1),
            nn.GELU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.large_kernel(x)
        out = self.cv1(out)
        out = self.cv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


