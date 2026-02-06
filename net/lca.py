"""轻量级交叉注意力模块 (Lightweight Cross Attention)

移植自 HVI-CIDNet，用于 HV 和 I 双分支之间的跨分支信息交互。
包含以下核心组件:
- CAB: 跨注意力块，使用多头注意力机制实现跨分支特征融合
- IEL: 亮度增强层，使用门控机制的前馈网络
- HV_LCA / I_LCA: 分别用于 HV 分支和 I 分支的 LCA 封装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):
    """支持 channels_first 和 channels_last 两种数据格式的 LayerNorm。

    Args:
        normalized_shape: 归一化维度大小
        eps: 数值稳定性参数
        data_format: 数据格式，channels_first (B,C,H,W) 或 channels_last (B,H,W,C)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        # channels_first: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CAB(nn.Module):
    """跨注意力块 (Cross Attention Block)。

    使用归一化的 Q/K 和可学习温度参数的多头通道注意力机制。
    Q 来自当前分支，K/V 来自另一个分支，实现跨分支信息提取。

    Args:
        dim: 输入通道数
        num_heads: 注意力头数
        bias: 是否使用偏置
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias
        )
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
            groups=dim * 2, bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: 当前分支特征，作为 Query 来源
            y: 另一个分支特征，作为 Key/Value 来源
        """
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        # 归一化 Q 和 K，使注意力更加稳定
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    """亮度增强层 (Intensity Enhancement Layer)。

    使用 Tanh 门控机制的前馈网络，通过双路径深度卷积和
    元素乘法实现非线性特征增强。

    Args:
        dim: 输入通道数
        ffn_expansion_factor: 隐藏层扩展因子
        bias: 是否使用偏置
    """

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1,
            padding=1, groups=hidden_features * 2, bias=bias,
        )
        self.dwconv1 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1,
            padding=1, groups=hidden_features, bias=bias,
        )
        self.dwconv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1,
            padding=1, groups=hidden_features, bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.tanh(self.dwconv1(x1)) + x1
        x2 = self.tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class HV_LCA(nn.Module):
    """HV 分支的轻量级交叉注意力模块。

    结构: CrossAttention(norm(hv), norm(i)) + IEL(norm(hv))
    注意: IEL 的输出不使用残差连接（与 I_LCA 不同）。

    Args:
        dim: 输入通道数
        num_heads: 注意力头数
        bias: 是否使用偏置
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        """
        Args:
            x: HV 分支特征 (Query)
            y: I 分支特征 (Key/Value)
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    """I 分支的轻量级交叉注意力模块。

    结构: CrossAttention(norm(i), norm(hv)) + 残差IEL(norm(i))
    注意: IEL 的输出使用残差连接（与 HV_LCA 不同）。

    Args:
        dim: 输入通道数
        num_heads: 注意力头数
        bias: 是否使用偏置
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        """
        Args:
            x: I 分支特征 (Query)
            y: HV 分支特征 (Key/Value)
        """
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x
