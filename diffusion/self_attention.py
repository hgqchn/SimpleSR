import torch
import torch.nn as nn
import math
from diffusion.common_utils import conv_nd, group_norm32, zero_module


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels,
                 num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # 组归一化，组数量为32
        self.norm = group_norm32(channels)
        # qkv矩阵,采用1x1卷积，
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        # b,c,hw
        x = x.reshape(b, c, -1)
        # b,3c,hw
        qkv = self.qkv(self.norm(x))
        # b,c,hw
        h = self.attention(qkv)
        h = self.proj_out(h)
        return h.reshape(b, c, *spatial)

class AttentionBlock_Residual(AttentionBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        return x + super().forward(x)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * Heads * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (Heads * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        # head_ch
        ch = width // (3 * self.n_heads)
        # q, k, v:  b x (heads*ch) x length
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # N,T,T
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # N,C,T
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        out = a.reshape(bs, -1, length)

        return out

class ImageSelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm)
        
        # 先拆分Q,K,V (B, 3*C, H, W) -> 3 * (B, C, H, W)
        # 假设Conv输出按 Q, K, V 顺序排列，这是标准实现
        query, key, value = qkv.chunk(3, dim=1)

        # 再变形为多头 (B, n_head, head_dim, H, W)
        query = query.view(batch, n_head, head_dim, height, width)
        key = key.view(batch, n_head, head_dim, height, width)
        value = value.view(batch, n_head, head_dim, height, width)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(head_dim)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


if __name__ == '__main__':
    pass
