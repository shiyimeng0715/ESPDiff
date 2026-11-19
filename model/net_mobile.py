import os
import warnings
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torch.nn import Module, Conv2d
from mmcv.cnn import ConvModule

from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock

from model.MobileViT_mobile import MobileViT
from model.model_config import get_config


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(Module):
    # MobileViT decoder. IRAM (Fre_enhance) and IPM (mask_pixel) have been removed.
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim
        # 这里的conv是自己写的函数（见line442），是两个conv+relu层
        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        #352->88
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )
        # 88->88
        self.down_mobile = ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=3, padding=1,
                                 norm_cfg=dict(type='BN', requires_grad=True))

        # 88->88
        self.down_1 = ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                                 norm_cfg=dict(type='BN', requires_grad=True))
        #88->44
        self.down_2 = ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=2,
                   norm_cfg=dict(type='BN', requires_grad=True))
        # 44->22
        self.down_3 = ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=2,
                                 norm_cfg=dict(type='BN', requires_grad=True))
        # 22->11
        self.down_4 = ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=2,
                                 norm_cfg=dict(type='BN', requires_grad=True))

        # 11->22
        self.up_4 = nn.Sequential(
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim, embedding_dim, factor=2),
        )
        # 22->44
        self.up_3 = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
                       Upsample(embedding_dim, embedding_dim, factor=2),
        )
        # 44->88
        self.up_2 = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim, embedding_dim, factor=2),
        )
        # 88->352
        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            # ConvModule(in_channels=embedding_dim//8+1, out_channels=embedding_dim//8, kernel_size=1,
            #            norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

        self.pred_mobile = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            Conv2d(embedding_dim, self.num_classes, kernel_size=3, padding=1)
        )

        # compose 模块：对 _c1~_c4 各挂一个（通道均为 embedding_dim=256），
        # 每个 compose 输出 f_ci（替代后面 cat 的 _ci）以及边界/骨架预测 f_bi/f_si。
        self.compose_c1 = Compose(embedding_dim)
        self.compose_c2 = Compose(embedding_dim)
        self.compose_c3 = Compose(embedding_dim)
        self.compose_c4 = Compose(embedding_dim)

        # inject 模块：用上一级信息与边界/骨架信息增强 f_ci_i，增强后的 f_ci 替代原 f_ci 去 cat。
        # inject_1 的上一级是 cat([去噪特征 x, 粗糙显著图 x1_mobile])，inject_2/3/4 的上一级是去噪下采样特征 x2/x3/x4（与 down 交叉）。
        self.inject_1 = Inject(embedding_dim, pre_dim=embedding_dim * 2)
        self.inject_2 = Inject(embedding_dim, pre_dim=embedding_dim)
        self.inject_3 = Inject(embedding_dim, pre_dim=embedding_dim)
        self.inject_4 = Inject(embedding_dim, pre_dim=embedding_dim)

    def forward(self, inputs, timesteps, x, mode):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4_copy = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3_copy = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2_copy = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        c1_copy = _c1

        # compose：由 _c1~_c4 得到 f_ci（替代 cat 的 _ci）以及边界/骨架预测
        f_ci_1, f_bi_1, f_si_1 = self.compose_c1(_c1)
        f_ci_2, f_bi_2, f_si_2 = self.compose_c2(_c2)
        f_ci_3, f_bi_3, f_si_3 = self.compose_c3(_c3)
        f_ci_4, f_bi_4, f_si_4 = self.compose_c4(_c4)
        f_bi_list = [f_bi_1, f_bi_2, f_bi_3, f_bi_4]   # 各分辨率边界 logits
        f_si_list = [f_si_1, f_si_2, f_si_3, f_si_4]   # 各分辨率骨架 logits

        # saliency map generation
        x_cat = torch.cat([c1_copy, c2_copy, c3_copy, c4_copy], dim=1)
        x_mobile_pred = self.pred_mobile(x_cat)
        x_mobile = F.interpolate(x_mobile_pred, scale_factor=4, mode='bilinear', align_corners=False)
        ##############################################
        # Multi-Prior Guidance Denoising Network
        _x = [x]
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)
        # x->x1 88->88
        x1_mobile = self.down_mobile(x_mobile_pred)

        # inject 与去噪下采样交叉：inject_1 接 cat([x, x1_mobile])，其余 inject_i 接去噪特征 x_i
        new_f_ci_1 = self.inject_1(f_ci_1, f_bi_1, f_si_1, f_ci_prev=torch.cat([x, x1_mobile], dim=1))
        x1 = self.down_1(new_f_ci_1)                 # 88->88
        x2 = self.down_2(x1)                         # 88->44
        new_f_ci_2 = self.inject_2(f_ci_2, f_bi_2, f_si_2, f_ci_prev=x2)
        x3 = self.down_3(new_f_ci_2)                 # 44->22
        new_f_ci_3 = self.inject_3(f_ci_3, f_bi_3, f_si_3, f_ci_prev=x3)
        x4 = self.down_4(new_f_ci_3)                 # 22->11
        new_f_ci_4 = self.inject_4(f_ci_4, f_bi_4, f_si_4, f_ci_prev=x4)

        # fusion x_feat and x then transposed conv
        # new_f_ci_4 作为上采样起点 11->22
        x_4 = self.up_4(new_f_ci_4)
        # x_4->x_3 22->44
        x_3 = self.up_3(torch.cat([x3, x_4], dim=1))
        # x_3->x_2 44->88
        x_2 = self.up_2(torch.cat([x2, x_3], dim=1))

        x_1 = torch.cat([x1, x_2], dim=1)
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x_1 = blk(x_1, t)
            else:
                x_1 = blk(x_1)
        # x = self.pred(torch.cat([x, _x.pop(-1)], dim=1))
        x_noise = self.pred(x_1)
        # SimpleDiffSef.py的line 272计算loss
        return x_noise, x_mobile, c1, c2, c3, c4, f_bi_list, f_si_list


class MobileViTBackbone(nn.Module):
    """MobileViT backbone, returning the last four stage features.

    MobileViT outputs 5 features at H/2, H/4, H/8, H/16, H/32; we drop the first
    one and return the last four H/4..H/32 features to the decoder.
    """

    def __init__(self, model_name='xx_small', pretrained_path=None, mask_chans=0, **kwargs):
        super().__init__()
        self.model = MobileViT(get_config(model_name), num_classes=1000, mask_chans=mask_chans)
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'[MobileViTBackbone] missing keys ({len(missing)}): {missing[:5]}')
        if unexpected:
            print(f'[MobileViTBackbone] unexpected keys ({len(unexpected)}): {unexpected[:5]}')

    def forward(self, x, timesteps=None, cond_img=None):
        # inject the noisy mask and timestep token.
        masp = self.model(cond_img, timesteps=timesteps, mask=x)
        return masp[1:]  # drop H/2 feature, keep last four


class net_mobilevit(nn.Module):
    _ckpt_name = {'xx_small': 'mobilevit_xxs.pt', 'x_small': 'mobilevit_xs.pt', 'small': 'mobilevit_s.pt'}

    def __init__(self, class_num=2, mask_chans=0, model_name='xx_small', pretrained_path=None, **kwargs):
        super(net_mobilevit, self).__init__()
        self.class_num = class_num
        mcfg = get_config(model_name)
        dims = [mcfg[k]['out_channels'] for k in ('layer2', 'layer3', 'layer4', 'layer5')]
        if pretrained_path is None:
            pretrained_path = os.path.join('checkpoints', self._ckpt_name[model_name])
        self.backbone = MobileViTBackbone(model_name=model_name, pretrained_path=pretrained_path, mask_chans=mask_chans)
        self.decode_head = Decoder(dims=dims, dim=256, class_num=class_num, mask_chans=mask_chans)

    def forward(self, x, timesteps, cond_img, mode):
        features = self.backbone(x, timesteps, cond_img)
        features, features_mobile, layer1, layer2, layer3, layer4, f_bi_list, f_si_list = self.decode_head(
            features, timesteps, x, mode)
        # 返回 4 个值：前两个是扩散输出/显著图，后两个是边界/骨架多分辨率预测（仅训练时用于辅助损失）
        return features, features_mobile, f_bi_list, f_si_list

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img, mode):
        # 采样路径只需前两个输出（扩散预测 + 显著图），丢弃边界/骨架预测
        features, features_mobile, _, _ = self.forward(x, timesteps, cond_img, mode)
        return features, features_mobile

    def extract_features(self, cond_img):
        # do nothing
        return cond_img


class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass


class DWConv(nn.Module):
    """深度可分离卷积(depthwise convolution)。

    每个通道独立卷积(groups=dim)，不改变通道数，用 padding 保持分辨率不变。
    """

    def __init__(self, dim, kernel_size, dilation=1):
        super().__init__()
        # padding = dilation*(k-1)/2，保证输出分辨率与输入一致
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=padding,
                              dilation=dilation, groups=dim, bias=False)

    def forward(self, x):
        return self.conv(x)


class Compose(nn.Module):
    """compose 模块：对输入特征 fi 提取多尺度上下文，输出边界预测 f_bi 与骨架预测 f_si。

    结构：
      1) fi -> DWConv5x5(d=1) -> a
         a 按通道对半拆成两段：
           前半 -> DWConv5x5(d=2) -> a1
           后半 -> DWConv7x7(d=1) -> a2
         cat(a, a1, a2) -> 1x1 conv + SiLU -> f_ci（通道/分辨率与 fi 一致）
      2) f_ci 分两路：
           1x1 -> b1、1x1 -> b2（两个不同的 1x1）
           cat(b1, b2) -> 1x1 -> c
           d1 = b1 + c，d2 = b2 + c
           d1 -> 1x1 -> 3x3 -> f_bi（边界 logits，1 通道）
           d2 -> 1x1 -> 3x3 -> f_si（骨架 logits，1 通道；与边界头参数不共享）

    返回 (f_ci, f_bi, f_si)：f_ci 与 fi 同通道/同分辨率，用于替代后续 cat 的 _ci；
    f_bi/f_si 均为 1 通道 logits（未 sigmoid），由 compose_loss 用 BCE 监督。
    注意：fi 通道数 dim 需为偶数（a 对半拆），MobileViT 各 stage 通道均满足。
    """

    def __init__(self, dim):
        super().__init__()
        # ---- 第一部分：多尺度 DWConv + 1x1 融合 ----
        self.dw_d1 = DWConv(dim, kernel_size=5, dilation=1)        # fi -> a
        self.dw_d5 = DWConv(dim // 2, kernel_size=5, dilation=2)   # a 前半 -> a1（感受野 9，11×11 上不再溢出）
        self.dw_d3 = DWConv(dim // 2, kernel_size=7, dilation=1)   # a 后半 -> a2（感受野 7）
        # cat(a, a1, a2) 通道 = dim + dim/2 + dim/2 = 2*dim，经 1x1 压回 dim
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False),
            nn.SiLU(),
        )

        # ---- 第二部分：双分支边界 / 骨架 ----
        self.conv_b1 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)     # f_ci -> b1
        self.conv_b2 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)     # f_ci -> b2
        self.conv_c = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)  # cat(b1,b2) -> c

        # 边界预测头：d1 -> 1x1 -> 3x3 -> 1 通道 logits
        self.boundary_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
        )
        # 骨架预测头：d2 -> 1x1 -> 3x3 -> 1 通道 logits
        self.skeleton_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, fi):
        # 第一部分：多尺度深度卷积 + 融合
        a = self.dw_d1(fi)
        a1 = self.dw_d5(a[:, :a.shape[1] // 2])
        a2 = self.dw_d3(a[:, a.shape[1] // 2:])
        f_ci = self.fuse(torch.cat([a, a1, a2], dim=1))

        # 第二部分：双分支
        b1 = self.conv_b1(f_ci)
        b2 = self.conv_b2(f_ci)
        c = self.conv_c(torch.cat([b1, b2], dim=1))
        d1 = b1 + c
        d2 = b2 + c
        f_bi = self.boundary_head(d1)   # (B,1,H,W) 边界 logits
        f_si = self.skeleton_head(d2)   # (B,1,H,W) 骨架 logits
        return f_ci, f_bi, f_si


def compose_loss(f_bi_list, f_si_list, edge_gt, skeleton_gt):
    """边界 + 骨架的多分辨率监督损失。

    Args:
        f_bi_list: list[Tensor] 每个分辨率一个边界预测 logits，shape (B,1,H_i,W_i)
        f_si_list: list[Tensor] 每个分辨率一个骨架预测 logits，shape (B,1,H_i,W_i)
        edge_gt: Tensor 全分辨率边界真值 (B,1,H,W)，取值 0/1
        skeleton_gt: Tensor 全分辨率骨架真值 (B,1,H,W)，取值 0/1

    Returns:
        Tensor 标量：所有分辨率下（边界 BCE + 骨架 BCE）之和。

    说明：
      - 每个分辨率的真值由全分辨率真值用 adaptive max pooling 下采样得到，
        以保留稀疏的边界/骨架细线（避免下采样后细线消失）。
      - BCE 与 structure_loss 中显著图的加权 BCE 完全一致（带 weit 边缘权重）。
    """

    def _wbce(pred, mask):
        # 与 structure_loss 中显著图一致的加权 BCE
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
        return (weit * bce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    total = 0.0
    for f_bi, f_si in zip(f_bi_list, f_si_list):
        h, w = f_bi.shape[2:]
        edge_ds = F.adaptive_max_pool2d(edge_gt, (h, w))
        skel_ds = F.adaptive_max_pool2d(skeleton_gt, (h, w))
        total = total + _wbce(f_bi, edge_ds) + _wbce(f_si, skel_ds)
    return total.mean()


class ChannelAttention(nn.Module):
    """通道注意力（CBAM 式，GMP + GAP 双分支）。

    GMP -> 1x1 -> ReLU -> 1x1 与 GAP -> 1x1 -> ReLU -> 1x1 两路相加，再过 sigmoid，
    得到 (B,C,1,1) 的通道权重。两条分支的 1x1 卷积参数不共享。
    """

    def __init__(self, dim):
        super().__init__()
        self.mlp_max = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.mlp_avg = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        gmp = F.adaptive_max_pool2d(x, 1)   # (B,C,1,1) 全局最大池化
        gap = F.adaptive_avg_pool2d(x, 1)   # (B,C,1,1) 全局平均池化
        return torch.sigmoid(self.mlp_max(gmp) + self.mlp_avg(gap))


class SpatialAttention(nn.Module):
    """空间注意力（CBAM 式）：沿通道做 max/avg 池化 -> cat(2 通道) -> 7x7 卷积 + sigmoid，
    得到 (B,1,H,W) 的空间权重图。
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # (B,1,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True)     # (B,1,H,W)
        out = torch.cat([max_out, avg_out], dim=1)       # (B,2,H,W)
        return torch.sigmoid(self.conv(out))             # (B,1,H,W)


class Inject(nn.Module):
    """inject 模块：用边界/骨架信息增强 f_ci_i。

    输入 f_ci_i(256 通道) 及对应的边界/骨架预测 f_bi/f_si(1 通道 logits)。
    - 可选地融合上一级信息（先下采样到当前分辨率，cat 后 1x1 压回 256），得到"新的 f_ci"。
    - 边界路得到 A，骨架路得到 B。
    - 输出 = 1x1(cat([new_f_ci, A, B]))，无额外可学习权重。
    输出保持 256 通道，替换原 f_ci_i 去 cat。
    """

    def __init__(self, dim, pre_dim=None):
        super().__init__()
        self.pre_dim = pre_dim
        if pre_dim is not None:
            # cat([上一级 inject 输出, 当前 f_ci]) -> 1x1 压回 dim
            self.fuse_pre = nn.Conv2d(pre_dim + dim, dim, kernel_size=1, bias=False)

        # ---- 边界路 -> A ----
        self.cha = ChannelAttention(dim)                              # 通道注意力
        self.sa = SpatialAttention()                                  # 空间注意力（权重图 (B,1,H,W)）
        self.bd_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # e+a -> A

        # ---- 骨架路 -> B ----
        self.sk_si_conv = nn.Conv2d(1, dim, kernel_size=1, bias=False)             # f_si(1通道概率) -> dim 通道映射
        self.sk_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)   # b -> c
        self.sk_conv11 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)            # 1x1（第一条支路）
        self.sk_conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)  # 3x3（第一条支路）
        self.sk_bn = nn.BatchNorm2d(dim)                                           # GAP 后 BN
        self.sk_gelu = nn.GELU()
        self.sk_conv12 = nn.Conv2d(dim, dim, kernel_size=1, bias=False)            # c -> h
        self.sk_out = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=False)           # cat(k,l) -> B

        # ---- 融合：cat([new_f_ci, A, B]) -> 1x1 ----
        self.fuse_out = nn.Conv2d(3 * dim, dim, kernel_size=1, bias=False)

    def forward(self, f_ci, f_bi, f_si, f_ci_prev=None):
        # 1) 构造"新的 f_ci"（256 通道）
        if self.pre_dim is not None:
            f_ci_prev = F.interpolate(f_ci_prev, size=f_ci.shape[2:], mode='bilinear', align_corners=False)
            new_f_ci = self.fuse_pre(torch.cat([f_ci_prev, f_ci], dim=1))
        else:
            new_f_ci = f_ci

        # f_si 先 sigmoid 成 0~1 概率图（骨架路输入调制）；f_bi 也先 sigmoid 到 0~1（边界路，与空间注意力 d 同量纲）
        f_si = torch.sigmoid(f_si)
        f_bi = torch.sigmoid(f_bi)

        # 2) 边界路 -> A
        # a -> 通道注意力 b -> c=a*b -> 空间注意力 d -> e=(d+f_bi)*c -> e+a -> 1x1
        a = new_f_ci
        b = self.cha(a)                        # (B,C,1,1)
        c = a * b
        d = self.sa(c)                         # (B,1,H,W) 空间权重图
        e = (d + f_bi) * c                     # f_bi(1通道,0~1) 广播
        A = self.bd_out(e + a)

        # 3) 骨架路 -> B（加性引导：f_si 先 1×1 扩通道再加，每个通道学习不同骨架响应）
        b_s = new_f_ci + self.sk_si_conv(f_si)
        c_s = self.sk_conv(b_s)
        f = c_s + self.sk_conv3(self.sk_conv11(c_s))   # 第一条支路：a + 3x3(1x1(a))
        g = self.sk_gelu(self.sk_bn(F.adaptive_avg_pool2d(c_s, 1)))  # (B,C,1,1)
        h = self.sk_conv12(c_s)
        k = f * g
        l = h * g
        B = self.sk_out(torch.cat([k, l], dim=1))

        # 4) 融合：cat([new_f_ci, A, B]) -> 1x1
        return self.fuse_out(torch.cat([new_f_ci, A, B], dim=1))
