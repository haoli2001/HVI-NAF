import torch
import torch.nn as nn
import torch.nn.functional as F

from net.HVI_transform import RGB_HVI


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2, drop_out_rate=0.0):
        super().__init__()
        dw_channels = channels * dw_expand

        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels, bias=True)
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, padding=0, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, padding=0, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channels = ffn_expand * channels
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, padding=0, bias=True)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class CrossInteraction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.i_to_hv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)
        self.hv_to_i = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, hv_feat, i_feat):
        hv_in = hv_feat
        i_in = i_feat

        gate_i_to_hv = torch.sigmoid(self.i_to_hv(i_in))
        gate_hv_to_i = torch.sigmoid(self.hv_to_i(hv_in))

        hv_out = hv_in + hv_in * gate_i_to_hv
        i_out = i_in + i_in * gate_hv_to_i
        return hv_out, i_out


class HVIDualNAF(nn.Module):
    def __init__(
        self,
        width=32,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
    ):
        super().__init__()

        self.trans = RGB_HVI()

        self.hv_intro = nn.Conv2d(2, width, kernel_size=3, padding=1, bias=True)
        self.i_intro = nn.Conv2d(1, width, kernel_size=3, padding=1, bias=True)

        self.hv_ending = nn.Conv2d(width, 2, kernel_size=3, padding=1, bias=True)
        self.i_ending = nn.Conv2d(width, 1, kernel_size=3, padding=1, bias=True)

        self.hv_encoders = nn.ModuleList()
        self.i_encoders = nn.ModuleList()
        self.hv_decoders = nn.ModuleList()
        self.i_decoders = nn.ModuleList()
        self.hv_downs = nn.ModuleList()
        self.i_downs = nn.ModuleList()
        self.hv_ups = nn.ModuleList()
        self.i_ups = nn.ModuleList()
        self.cims = nn.ModuleList()

        channels = width
        for num in enc_blk_nums:
            self.hv_encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
            self.i_encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
            self.cims.append(CrossInteraction(channels))
            self.hv_downs.append(nn.Conv2d(channels, 2 * channels, kernel_size=2, stride=2))
            self.i_downs.append(nn.Conv2d(channels, 2 * channels, kernel_size=2, stride=2))
            channels *= 2

        self.hv_middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])
        self.i_middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.hv_ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            self.i_ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.hv_decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))
            self.i_decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num)]))

        self.padder_size = 2 ** len(self.hv_encoders)

    def _pad_input(self, hv, i):
        _, _, h, w = hv.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            hv = F.pad(hv, (0, mod_pad_w, 0, mod_pad_h))
            i = F.pad(i, (0, mod_pad_w, 0, mod_pad_h))
        return hv, i, h, w

    def forward(self, x):
        hvi = self.trans.HVIT(x)
        hv = hvi[:, :2, :, :]
        i = hvi[:, 2:3, :, :]

        hv, i, orig_h, orig_w = self._pad_input(hv, i)
        hv_inp = hv
        i_inp = i

        hv = self.hv_intro(hv)
        i = self.i_intro(i)

        hv_skips = []
        i_skips = []

        for hv_enc, hv_down, i_enc, i_down, cim in zip(
            self.hv_encoders, self.hv_downs, self.i_encoders, self.i_downs, self.cims
        ):
            hv = hv_enc(hv)
            i = i_enc(i)
            hv, i = cim(hv, i)
            hv_skips.append(hv)
            i_skips.append(i)
            hv = hv_down(hv)
            i = i_down(i)

        hv = self.hv_middle(hv)
        i = self.i_middle(i)

        for hv_dec, hv_up, hv_skip, i_dec, i_up, i_skip in zip(
            self.hv_decoders, self.hv_ups, reversed(hv_skips), self.i_decoders, self.i_ups, reversed(i_skips)
        ):
            hv = hv_up(hv)
            hv = hv + hv_skip
            hv = hv_dec(hv)

            i = i_up(i)
            i = i + i_skip
            i = i_dec(i)

        hv = self.hv_ending(hv) + hv_inp
        i = self.i_ending(i) + i_inp

        hvi_out = torch.cat([hv, i], dim=1)
        out = self.trans.PHVIT(hvi_out)
        return out[:, :, :orig_h, :orig_w]

    def HVIT(self, x):
        return self.trans.HVIT(x)
