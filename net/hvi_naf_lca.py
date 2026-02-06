"""HVI-NAF-LCA: 结合 NAFNet 主干和 CIDNet LCA 跨分支交互的双分支网络。

核心特点:
1. 使用 NAFBlock 作为编码/解码主干网络，保持高效的特征提取能力
2. 使用 CIDNet 的真实 LCA (Lightweight Cross Attention) 模块进行跨分支交互
3. HV 分支输入完整的 3 通道 HVI（包含亮度信息），而不仅仅是 HV 2 通道
4. 在编码器、瓶颈、解码器的所有层级都加入 LCA 交互
5. 使用双线性上采样，减少棋盘格效应
6. 采用 CIDNet 风格的残差连接: output = trans(concat(hv_out, i_out) + hvi_input)

架构层级:
    Encoder: 4 levels with channels [width, 2*width, 4*width, 8*width]
        enc_blk_nums: [2, 2, 4, 8] blocks per level
        LCA at each level after encoder blocks

    Middle: 12 NAFBlocks at bottleneck
        LCA in the middle of middle blocks

    Decoder: 4 levels
        dec_blk_nums: [2, 2, 2, 2] blocks per level
        LCA at each level before decoder blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.HVI_transform import RGB_HVI
from net.hvi_dual_naf import NAFBlock
from net.lca import HV_LCA, I_LCA


class HVINAF_LCA(nn.Module):
    """HVI-NAF-LCA 双分支低光照增强网络。

    Args:
        width: 初始通道数，默认 32
        enc_blk_nums: 编码器每级 NAFBlock 数量，默认 [2, 2, 4, 8]
        middle_blk_num: 瓶颈 NAFBlock 数量，默认 12
        dec_blk_nums: 解码器每级 NAFBlock 数量，默认 [2, 2, 2, 2]
        lca_heads: LCA 注意力头数配置，None 则自动计算
    """

    def __init__(
        self,
        width=32,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
        lca_heads=None,
    ):
        super().__init__()

        self.trans = RGB_HVI()
        self.num_levels = len(enc_blk_nums)

        # HV 分支输入完整的 3 通道 HVI (H, V, I)，与 CIDNet 保持一致
        self.hv_intro = nn.Conv2d(3, width, kernel_size=3, padding=1, bias=True)
        self.i_intro = nn.Conv2d(1, width, kernel_size=3, padding=1, bias=True)

        self.hv_ending = nn.Conv2d(width, 2, kernel_size=3, padding=1, bias=True)
        self.i_ending = nn.Conv2d(width, 1, kernel_size=3, padding=1, bias=True)

        # 计算每级的通道数和注意力头数
        self.channels = []
        self.heads = []
        ch = width
        for i in range(self.num_levels):
            self.channels.append(ch)
            if lca_heads is None:
                # 自动计算头数: 每 32 通道 1 个头，最大 8 个头
                head = max(1, min(8, ch // 32))
                if ch % head != 0:
                    head = 1
            else:
                head = lca_heads[i] if i < len(lca_heads) else lca_heads[-1]
            self.heads.append(head)
            ch *= 2

        # ===== 编码器 =====
        # 每级: NAFBlocks -> LCA -> Downsample
        self.hv_encoders = nn.ModuleList()
        self.i_encoders = nn.ModuleList()
        self.hv_enc_lcas = nn.ModuleList()
        self.i_enc_lcas = nn.ModuleList()
        self.hv_downs = nn.ModuleList()
        self.i_downs = nn.ModuleList()

        for num, ch, head in zip(enc_blk_nums, self.channels, self.heads):
            self.hv_encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.i_encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.hv_enc_lcas.append(HV_LCA(ch, head, bias=False))
            self.i_enc_lcas.append(I_LCA(ch, head, bias=False))
            self.hv_downs.append(nn.Conv2d(ch, 2 * ch, kernel_size=2, stride=2))
            self.i_downs.append(nn.Conv2d(ch, 2 * ch, kernel_size=2, stride=2))

        # ===== 瓶颈 =====
        bottleneck_ch = self.channels[-1] * 2
        if lca_heads is None:
            bottleneck_head = max(1, min(8, bottleneck_ch // 32))
            if bottleneck_ch % bottleneck_head != 0:
                bottleneck_head = 1
        else:
            bottleneck_head = lca_heads[-1] if len(lca_heads) >= self.num_levels else lca_heads[-1]

        # 将瓶颈分为两半，中间插入 LCA
        half_middle = middle_blk_num // 2
        self.hv_middle_1 = nn.Sequential(*[NAFBlock(bottleneck_ch) for _ in range(half_middle)])
        self.i_middle_1 = nn.Sequential(*[NAFBlock(bottleneck_ch) for _ in range(half_middle)])
        self.hv_middle_lca = HV_LCA(bottleneck_ch, bottleneck_head, bias=False)
        self.i_middle_lca = I_LCA(bottleneck_ch, bottleneck_head, bias=False)
        self.hv_middle_2 = nn.Sequential(*[NAFBlock(bottleneck_ch) for _ in range(middle_blk_num - half_middle)])
        self.i_middle_2 = nn.Sequential(*[NAFBlock(bottleneck_ch) for _ in range(middle_blk_num - half_middle)])

        # ===== 解码器 =====
        # 每级: Upsample -> SkipAdd -> LCA -> NAFBlocks
        self.hv_ups = nn.ModuleList()
        self.i_ups = nn.ModuleList()
        self.hv_dec_lcas = nn.ModuleList()
        self.i_dec_lcas = nn.ModuleList()
        self.hv_decoders = nn.ModuleList()
        self.i_decoders = nn.ModuleList()

        for num, ch, head in zip(dec_blk_nums, reversed(self.channels), reversed(self.heads)):
            self.hv_ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=True),
                )
            )
            self.i_ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(ch * 2, ch, kernel_size=3, padding=1, bias=True),
                )
            )
            self.hv_dec_lcas.append(HV_LCA(ch, head, bias=False))
            self.i_dec_lcas.append(I_LCA(ch, head, bias=False))
            self.hv_decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))
            self.i_decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(num)]))

        self.padder_size = 2 ** self.num_levels

    def _pad_input(self, hvi):
        """输入填充到 2^n 倍数。"""
        _, _, h, w = hvi.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            hvi = F.pad(hvi, (0, mod_pad_w, 0, mod_pad_h))
        return hvi, h, w

    def forward(self, x):
        """
        Args:
            x: RGB 输入图像 (B, 3, H, W)

        Returns:
            out: 增强后的 RGB 图像 (B, 3, orig_h, orig_w)
        """
        # RGB -> HVI 变换
        hvi = self.trans.HVIT(x)
        hvi_padded, orig_h, orig_w = self._pad_input(hvi)
        hvi_input = hvi_padded  # 保存用于最终的残差连接

        # HV 分支: 输入完整的 3 通道 HVI (与 CIDNet 一致)
        hv = hvi_padded  # (B, 3, H, W)
        i = hvi_padded[:, 2:3, :, :]  # (B, 1, H, W)

        # 初始卷积
        hv = self.hv_intro(hv)
        i = self.i_intro(i)

        hv_skips = []
        i_skips = []

        # ===== 编码器 =====
        for hv_enc, hv_enc_lca, hv_down, i_enc, i_enc_lca, i_down in zip(
            self.hv_encoders, self.hv_enc_lcas, self.hv_downs,
            self.i_encoders, self.i_enc_lcas, self.i_downs
        ):
            hv = hv_enc(hv)
            i = i_enc(i)
            # 跨分支交互
            hv = hv_enc_lca(hv, i)
            i = i_enc_lca(i, hv)

            hv_skips.append(hv)
            i_skips.append(i)

            hv = hv_down(hv)
            i = i_down(i)

        # ===== 瓶颈 =====
        hv = self.hv_middle_1(hv)
        i = self.i_middle_1(i)
        # 瓶颈 LCA 交互
        hv = self.hv_middle_lca(hv, i)
        i = self.i_middle_lca(i, hv)
        hv = self.hv_middle_2(hv)
        i = self.i_middle_2(i)

        # ===== 解码器 =====
        for hv_up, hv_dec_lca, hv_dec, hv_skip, i_up, i_dec_lca, i_dec, i_skip in zip(
            self.hv_ups, self.hv_dec_lcas, self.hv_decoders, reversed(hv_skips),
            self.i_ups, self.i_dec_lcas, self.i_decoders, reversed(i_skips)
        ):
            # 上采样
            hv = hv_up(hv)
            hv = hv + hv_skip
            i = i_up(i)
            i = i + i_skip

            # 跨分支交互
            hv = hv_dec_lca(hv, i)
            i = i_dec_lca(i, hv)

            # NAFBlocks
            hv = hv_dec(hv)
            i = i_dec(i)

        # 输出投影
        hv = self.hv_ending(hv)
        i = self.i_ending(i)

        # 拼接 HV 和 I 输出
        hvi_out = torch.cat([hv, i], dim=1)

        # CIDNet 风格的残差连接: 直接在 HVI 空间添加残差
        hvi_out = hvi_out + hvi_input

        # HVI -> RGB 逆变换
        out = self.trans.PHVIT(hvi_out)
        return out[:, :, :orig_h, :orig_w]

    def HVIT(self, x):
        """获取输入的 HVI 表示，用于损失计算。"""
        return self.trans.HVIT(x)


def count_parameters(model):
    """统计模型参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = HVINAF_LCA(
        width=32,
        enc_blk_nums=(2, 2, 4, 8),
        middle_blk_num=12,
        dec_blk_nums=(2, 2, 2, 2),
    )
    model.cuda()

    # 打印参数量
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 256, 256).cuda()
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # 测试 HVIT
    hvi = model.HVIT(x)
    print(f"HVI shape: {hvi.shape}")
