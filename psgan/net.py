import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
from torch.nn.parameter import Parameter
from .fpt_style import FPT_style
from .fpt_content import FPT_content
import matplotlib.pyplot as plt
import cv2
import numpy as np
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='in', activation='lrelu', pad_type='zero', alpha=0.2):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        # initialize padding 
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)  # nn.layernorm???
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(alpha, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):  # 得到有多少个窗，每个窗的高宽通道数
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # [B*num_windows, Mh, Mw, C]


def window_reverse(windows, window_size, H, W):
    """
    将一个个window还原成feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C] 求出B的数量，然后因为本来B*num_windows就是由B, H//Mh, W//Mw,得到的，所以按照原来这个形式就可以还原回去
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] ->B, H//Mh, W//Mw,
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x  # B, H//Mh, W//Mw,


class WindowAttention(nn.Module):  # 实现W-MSA/SW-MSA，SW-MSA实现部分功能（因为挪上面的一些行和左面的一些列在之前SwinTransformerBlock挪过了）
    r""" Window based multi-head self attention (W-MSA) module with relative position bias. #官方这里写的就是实现W-MSA的方法
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 得到每个head的dimension
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # num_heads说明每个head使用的relative_position_bias_table不一样

        # get pair-wise relative position index for each token inside the window 生成relative position index
        coords_h = torch.arange(self.window_size[0])  # 0，1
        coords_w = torch.arange(self.window_size[1])  # 0，1
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww 创建网格，他以行和列表示，meshgrid会返回两个tensor,所以用stack拼接
        coords_flatten = torch.flatten(coords, 1)  # 构建绝对索引，2, Wh*Ww   #从第一维开始展平 就是，第一行是每个像素对应的行标，第二行是每个像素对应的列标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 得到相对位置索引，2, Wh*Ww, Wh*Ww 利用None新增维度
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0 一种计算方法
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(
            -1)  # Wh*Ww, Wh*Ww 在最后一个维度求和，就是行标和列标相加 所以relative_position_index就是构建好的相对位置索引
        self.register_buffer("relative_position_index",
                             relative_position_index)  # 利用register_buffer方法将relative_position_index放入缓存，这个值是固定的值

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 和最早的attention计算方式一样
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化相对位置编码table
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 通过index去table里取对应参数
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(
            0)  # 在patch_embedding加上relative_position_bias.unsqueeze(0)相对位置编码，差一个batch维度所以用unsqueeze
        # qk自注意力机制算完得到attention map后加上相对位置编码，然后用上mask把qk不合适的地方置为0，再和v相乘
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:  # 如果mask为None直接用softmax处理，如果不为None
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # input_resolution=(patches_resolution[0],patches_resolution[1]) 264，184
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # shift_size=0 if (i % 2 == 0) else window_size // 2, 按照for循环
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:  ## 如果输入分辨率小于M，就直接一个window attention就好
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:  # 0的时候，就不用mask，1的时候就用mask # 对SW-MSA，生成attention mask
            attn_mask = self.calculate_mask(self.input_resolution)  # 264，184    #初始化的时候按照输入特征图的大小制作mask
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    # mask每次都是按照已经移动好了来计算的，也就是说不是单纯的从上到下从左到右3*3是同一种数字，而是按照已经第一行到了最后一行，最左面一列到了最右面，然后在silce的时候切成大小不一的mask,不都是3*3，3*3只有4个,然后是各个大小，每个块（mask）里面的数字都不同
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍,为了支持多尺度而设定的
        # Hp = int(np.ceil(H / self.window_size)) * self.window_size
        # Wp = int(np.ceil(W / self.window_size)) * self.window_size
        H, W = x_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition，window_partition也是这样(1, H, W, 1)顺序 patch_embedding后，[B,HW,C]
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # 只是赋值了，没有切块，还是一张图img_mask
                cnt += 1

        mask_windows = window_partition(img_mask,
                                        self.window_size)  # 【nW, window_size, window_size, 1 】==[B*num_windows, Mh, Mw, C]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)  # [nW, Mh*Mw*1] 【窗的数量，每个窗的大小展平】二维，每一行都是一个窗的展平
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # mask_windows是3*3的窗口，但是窗口里的值不一样
        # [nW, Mh*Mw*1]第1号位置加一个维度，[nW,1,Mh*Mw*1]二号位置加一个维度[nW, Mh*Mw*1,1]，再广播机制复制
        # [nW,1,Mh*Mw*1]->[nW,Mh*Mw*1,Mh*Mw*1]一个窗口是一行，每行复制那一行
        # [nW,Mh*Mw*1,1]->[nW,Mh*Mw*1,Mh*Mw*1]一个窗口是一列，每列复制那一列
        # 广播后的矩阵相减，就能计算每个窗口的所有点关于每一个点的相对位置。 mask_windows.unsqueeze(2)每一行是同一个数，窗口的一个位置的数 mask_windows.unsqueeze(1)每一行是一个窗口的各个值
        # 可以计算每个点和其他点是否是相邻位置
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # 相减得到的结果，同一区域的减完了就都是0，不是0的就不是相邻位置，计算窗口内Shift window后做多头注意力机制的部分
        return attn_mask

    def forward(self, x, x_size):  # 图像特征图进来，是patch_embedding以后的  x_size是经过浅层特征的输出，x自己的HW在input_resolution中
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        ## pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        # 代码没写，多尺度处理部分

        # cyclic shift
        if self.shift_size > 0:  # 从上往下移，从左往右移
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        mask = self.calculate_mask(x_size)
        # partition windows 移动完特征图后划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:  # input_resolution输入分辨率，x_size不是特征图的高和宽， x_size是经过浅层特征的输出，那么直接用初始化的mask 264，184
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:  # 否则就按照x_size（x_size是经过浅层特征的输出）做mask,然后把274行分好的窗口以及mask送进WindowAttention做窗口内多头注意力机制
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows view:计算完窗口中的多头注意力，274行还原回去，HW分开 window-reverse把窗口拼回去，得到feature map
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift 把特征图还原回去
        if self.shift_size > 0:  # 从下往上，从右往左
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)  # 有shortcut说明尺寸不变，所以atten_mask可以写在外面BasicLayer
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.InstanceNorm2d, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim  # 每个SwinTransformerBlock的多头注意力机制有6个头
        self.input_resolution = input_resolution  # input_resolution=(patches_resolution[0],patches_resolution[1])
        self.depth = depth  # SwinTransformerBlock有6个
        self.use_checkpoint = use_checkpoint

        # build blocks，SwinTransformerBlock是最基础的模块，就是swin的核心
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])  # 一个stage多少个block

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    # 在这里就把atten mask写出来的话，只需要计算一遍，因为每个stage的N个block，都用的相同大小的mask，要不然进blocks里每次都创建一遍
    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.InstanceNorm2d, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        # 这里的residual_grop,BsicLayer传入的参数和原论文SwinTransformer类传入BasicLayer是一样的，BasicLayer是一个stage模块类，stage中的每个block是SwinTransformerBlock类，这个才是最基础的
        self.residual_group = BasicLayer(dim=dim,  # resudyak_group相比原论文是再封装一层，因为这个类是用来写残差的，具体的内容在BasicLayer类中
                                         input_resolution=input_resolution,  # 输入分辨率
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         # 下采样downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
        # 用上了，self.conv(self.patch_unembed
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory 降低计算量参数和节省内存
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    # x_size是经过浅层特征的输出，在patch_size之前，x的HW在它自己类属性input_resolution中，
    def forward(self, x, x_size):  # patch_unembedding要传入x_size因为要恢复成能用卷积的特征图，HW要分开，patch_embed不需要，因为它要把HW合在一起
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size),
                                                             x_size))) + x  # 先patch_embed再残差相加，因为最开始进入RSTB的时候已经是打平的了，后面要还原，再加上打平的x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution  # input_resolution=(patches_resolution[0],patches_resolution[1])
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=60 * 4, norm_layer=None):  # patch_size=1
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[
            1]]  # //，在python中，这个叫“地板除”，3//2=1  /，这是传统的除法，3/2=1.5  %，这个是取模操作，也就是区余数，4%2=0，5%2=1
        self.img_size = img_size
        self.patch_size = patch_size  # 1 不做下采样
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # 分成多少个patch 264,184

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        '''
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        '''
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # flatten:[B,C,H,W]->[B,C,HW]
        # transpose:[B,C,HW]->[B,HW,C]
        # 缺少卷积下采样，只是linear embeding了
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)  # 使用layernorm在channel维度做处理 LayerNorm是把一张图的所有C的HW弄一起，算C*H*W的均值
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


################编码器模块
class FaceEncoder(nn.Module):  # FIEnc
    def __init__(self, input_dim=3, dim=60,
                 img_size=64, patch_size=1, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.InstanceNorm2d, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 norm='in', activ='lrelu', pad_type='reflect', rfb=False, vis=False,
                 **kwargs):
        super(FaceEncoder, self).__init__()

        self.window_size = window_size  # Window size. 8
        self.img_range = img_range
        self.num_layers = len(depths)  # Depth of each Swin Transformer layer.
        self.embed_dim = dim  # Patch embedding dimension=256/384
        self.ape = ape  # If True, add absolute position embedding to the patch embedding. Default: False
        self.patch_norm = patch_norm  # If True, add normalization after patch embedding.
        self.num_features = dim * 4  # Patch embedding dimension. 60
        self.mlp_ratio = mlp_ratio  # Ratio of mlp hidden dim to embedding dim. Default: 4

        self.conv_layer1 = ConvBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ,
                                     pad_type=pad_type)  # dim=64 image_size=256*25
        # downsampling blocks   image_size=256-->128-->64 embed_dim=64-->128-->256
        self.conv_layer2 = ConvBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        self.conv_layer3 = ConvBlock(2 * dim, 4 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)

        dim *= 4  # 384

        self.patch_embed = PatchEmbed(
            # patch_size=1,embed_dim=256,img_size=64*64 没必要128单独拿出来做，因为那样的话就是128-》64-》32-》32之前试过，会爆内存。现在就是64-》64-》64-》6
            # 之前dim是96-》192-》384-》384 现在是256-》256-》256-》256  #也可以dim用96开始，那么96-》192-》384-》（384-》384-》384-》384）
            img_size=img_size // 4, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches  # 经过patch embedding得到多少个patch，由patches_resolution[0] * patches_resolution[1]得到
        patches_resolution = self.patch_embed.patches_resolution  # 经过patch embedding后，高的方向有几个块，宽的方向有几个块
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,  # patch_size=1
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth  在transformer block中他们采用的dropout rate从0慢慢增长到1，根据depths定
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 一共4个stage，通过for循环之后创建了所有的layer（statge）并把layer(stage)放到modulelist之中
            layer = RSTB(dim=dim,
                         # 本来这里接的是basiclayer，现在因为有残差块，所以basiclayer在RSTB中使用，原文是dim=int(embed_dim*2**i_layer)，这里是一直是embed_dim
                         input_resolution=(patches_resolution[0],  # 多
                                           patches_resolution[1]),
                         depth=depths[i_layer],  # 在每个stage堆叠6次swin transformer block
                         num_heads=num_heads[i_layer],  # 每个swin transformer block有6个头
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,  # qk_scale多
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         # downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                         # patch merging和patch embedding一样，是用来下采样和channel增加的，原论文是宽高减半4-->2，channel变为原来两倍，经过window_size/2分为的窗口，然后同样位置上的像素进行拼接-->channel维度做concat-->layernorm-->linear
                         downsample=None,
                         # 本次stage后是否有patch merging 本代码没用patch merging why? downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,  # 多
                         patch_size=patch_size,  # 多
                         resi_connection=resi_connection  # 多
                         )
            self.layers.append(layer)
        self.norm = norm_layer(dim)  # 加一个layernorm，如果是分类模型，再在这后面加自适应的全局平均池化和linear线性映射改变输出节点个数

        self.apply(self._init_weights)

    def _init_weights(self, m):  # swin的参数初始化 那么ASFF和conv层的参数初始化呢？ssat的network.py的397行 在G后init，因为原代码的生成器不止一个类还有好多其他部分
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):  # 只有多个RSTL，没有多个RSTL后的conv
        # 这个x_size就是在原论文中本来在embed中传出的H,W
        raw_x = x
        x_size = (
            x.shape[2], x.shape[3])  # 经过浅层特征提取self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1) 后得到的图，大小和原来一样
        x = self.patch_embed(x)  # split image into non-overlapping patches
        if self.ape:
            x = x + self.absolute_pos_embed  # 绝对位置编码
        x = self.pos_drop(x)  # patch embedding后面加上一个dropout

        for layer in self.layers:  # 4层        #到这里是，patch partition+linear embedding都像原论文一样，先处理了，后面每一个layer都是他自己stage的SwinTransformerBlock和下一个stage的patch embedding结合
            x = layer(x, x_size)  # 由801行得知，和原论文直接传入H,W是不一样的 x_size是经过浅层特征的输出，原文的HW是已经封装在x的input_resolution了

        x = self.norm(x)  # B L C           #因为不是分类网络所以不需要自适应全局平均池化和线性映射
        x = self.patch_unembed(x, x_size)  # 恢复成卷积能做的特征图，原来经过patch_embedding已经把hw打平放一起了
        x = raw_x + x

        return x

    def forward(self, x):
        results = [x]
        # 卷积提取和下采样
        x = self.conv_layer1(x)  # 3  64 256
        results.append(x)
        x = self.conv_layer2(x)  # 128
        results.append(x)
        x = self.conv_layer3(x)  # 64
        results.append(x)
        # print("x_convlayer3:", x_convlayer3.shape)

        x = self.check_image_size(x)  # pad操作，根据window_size，不想pad需要整除，这里需要是8
        # print("pad_x_convlayer3:", x_convlayer3.shape)
        # 残差swinTransformer,多个RSTL
        x = self.forward_features(x)  # 4层 for layer in self.layers
        results.append(x)
        # ASFF还没加，只到swinIR的部分结束

        return results


class Content_SA(nn.Module):
    def __init__(self, in_dim):
        super(Content_SA, self).__init__()
        # self.fpt = FPT(feature_dim=in_dim)
        self.fpt = FPT_content(feature_dim=in_dim)
        # self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        # self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        # self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        # self.softmax = nn.Softmax(dim=-1)
        # self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, style_feat):
        res1, res2, res3, res4 = style_feat[1], style_feat[2], style_feat[3], style_feat[4]
        # print("res2:",res2.shape)
        # print("res3:", res3.shape)
        # print("res5:", res6.shape)

        # style_feat = self.fpt(res1, res2, res3, res4)
        out = self.fpt(res1, res2, res4)
        # B, C, H, W = style_feat.size()
        # F_Fc_norm = self.f(style_feat).view(B, -1, H * W)
        #
        # B, C, H, W = style_feat.size()
        # G_Fs_norm = self.g(style_feat).view(B, -1, H * W).permute(0, 2, 1)  # C*C
        #
        # energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        # attention = self.softmax(energy)
        #
        # H_Fs = self.h(style_feat).view(B, -1, H * W)
        # out = torch.bmm(attention.permute(0, 2, 1), H_Fs)
        #
        # out = out.view(B, C, H, W)
        # out = self.out_conv(out)
        # out += style_feat

        return out



class ASFF(nn.Module):
    # level值可能为0,1,2
    # 0是第一个分支的尺度（尺度最大，图片大小最小）
    # 1是第二个分支的尺度（图片大小是上面的两倍）
    # 2是第三个分支的尺度（图片大小是上面的两倍）
    def __init__(self, dim=256, rfb=False, vis=False):
        super(ASFF, self).__init__()
        # 每个level的通道数
        self.inter_dim = dim
        # self.expand就是改变通道数
        # self.compress_level_0就是特征图大小不变，只是加了一层卷积
        # self.stride_level_1/2就是每次使得特征图大小减为原来1/2
        # F.max_pool2d用在特征图需要缩小为4倍的那层，因为一个卷积只缩小2倍
        # self.stride_level_1/2除了需要缩小4倍的要加个F.max_pool2d，其他的都是直接加上，用来缩小2倍就行了
        # F.interpolate可以插值2倍，可以插值4倍
        self.stride_level_1 = ConvBlock(dim//2, self.inter_dim, 3, 2, 1, norm='in', activation='lrelu',
                                        pad_type='reflect',
                                        alpha=0.1)  # add_conv(in_ch, out_ch, ksize, stride, leaky=True): ksize=3,pad=1,stride=2
        self.stride_level_2 = ConvBlock(dim // 4, self.inter_dim, 3, 2, 1, norm='in', activation='lrelu',
                                        pad_type='reflect',
                                        alpha=0.1)  # stride=2,特征图缩小一半
        self.expand = ConvBlock(self.inter_dim, self.inter_dim, 3, 1, 1, norm='in', activation='lrelu',
                                pad_type='reflect',
                                alpha=0.1)  # 特征图大小不变，(W-ksize+2pad)/stride +1=不变，改变通道数

        compress_c = 8  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = ConvBlock(self.inter_dim, compress_c, 1, 1, 0, norm='in', activation='lrelu',
                                        pad_type='reflect',
                                        alpha=0.1)  # 特征图大小不变，这一层卷积是做权重系数，三个weight_level把输出通道数弄成一样的好相加
        self.weight_level_1 = ConvBlock(self.inter_dim, compress_c, 1, 1, 0, norm='in', activation='lrelu',
                                        pad_type='reflect', alpha=0.1)
        self.weight_level_2 = ConvBlock(self.inter_dim, compress_c, 1, 1, 0, norm='in', activation='lrelu',
                                        pad_type='reflect', alpha=0.1)
        ########################################
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)  # 因为是3个特征图拼接所以输入特征图x3
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):  # x_level_0是最小的特征图，x_level_2是最大的特征图  #dim=256,128,64
        # 每次都有一个特征图什么都不做，直接赋值到最后×权重

        # 第一个分支是最后一层的输出
        level_0_resized = x_level_0  # 第三个卷积没拿出来加入是因为本身最后就有个RSTL最后加上第三层卷积的结果

        # 使第二个分支的特征形状与第一个分支相同，注意这个卷积会缩小特征大小到一半，一共缩小一次，变成1/2
        level_1_resized = self.stride_level_1(x_level_1)  # 第二个卷积

        # 使第三个分支的特征形状与第一个分支相同，pool缩小一半，卷积再缩小一半，一共缩小两次，变成1/4
        level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)  # kernel=3   #第一个卷积
        level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        # 计算每个level特征的权重

        # 先把每个level的特征过卷积，然后拼接起来再过卷积
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)  # 一层卷积，把concat的三个特征图重新洗牌得到新的channel维数的特征图

        # 略
        levels_weight = F.softmax(levels_weight, dim=1)

        # 加权求和
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        # 加权求和后再过个卷积改变通道数
        out = self.expand(fused_out_reduced)  # 所有的都会过expand所以写在最外面就行

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class Style_SA(nn.Module):
    def __init__(self, in_dim):
        super(Style_SA, self).__init__()
        self.fpt = FPT_style(feature_dim=in_dim//2)
        self.asff = ASFF(in_dim)

    def forward(self, style_feat):
        res1, res2, res3, res4 = style_feat[1], style_feat[2], style_feat[3], style_feat[4]
        # print("res2:",res2.shape)
        # print("res3:", res3.shape)
        # print("res5:", res6.shape)

        res2 = self.fpt(res1, res2, res3)
        out = self.asff(res4, res2, res1)
        return out


class GetMatrix(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


# class NONLocalBlock2D(nn.Module):
#     def __init__(self):
#         super(NONLocalBlock2D, self).__init__()
#         self.g = nn.Conv2d(in_channels=1, out_channels=1,
#                            kernel_size=1, stride=1, padding=0)
#
#     def forward(self, source, weight):
#         """(b, c, h, w)
#         src_diff: (3, 136, 32, 32)  # weight(3, HW, HW)
#         """
#         # print("source:",source.size())
#         batch_size = source.size()[0]
#         channel = source.size()[1]
#
#         g_source = source.view(batch_size, channel, -1)  # (N, C, H*W)
#         g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)
#         # print("weight.size()",weight.size())
#         # print("g_source.size()",g_source.size())
#         y = torch.bmm(weight.to_dense(), g_source)
#         y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
#         y = y.view(batch_size, channel, *source.size()[2:])
#         return y


# 面部特征和妆容特征融合，注意力Non-local模块
class Multi_Adaptation_Module(nn.Module):
    def __init__(self, in_dim):  # 192个通道
        super(Multi_Adaptation_Module, self).__init__()
        #####################扭曲妆容模块
        # self.CA = SymmetryAttention()
        # ######################分别通过各自的图提取内容特征和妆容特征
        # self.CSA = Content_SA(in_dim)   #######已完成，按照ACMMM2020保留白化操作进一步分离内容特征和妆容特征，保证提的内容特征很纯粹 里面的conv都是1*1卷积，non-local，dim不变 384 32*32->64*64
        # self.SSA = Style_SA(in_dim)     #######FPT 32*32->64*64 dim=192

        # Bottleneck. All bottlenecks share the same attention module
        # self.atten_bottleneck_g = NONLocalBlock2D()
        # self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetMatrix(in_dim, in_dim)  # get the makeup matrix

        self.n_blocks = 6
        for i in range(self.n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaLINBlock(in_dim, use_bias=True, use_se=False))

    def atten_feature(self, mask_s, weight, gamma_s, beta_s):
        """

        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        # print("mask_s_re",mask_s_re.shape)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w)
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        B_g, C_g, H_g, W_g = gamma_s.shape[0], gamma_s.shape[1], gamma_s.shape[2], gamma_s.shape[3]
        B_b, C_b, H_b, W_b = beta_s.shape[0], beta_s.shape[1], beta_s.shape[2], beta_s.shape[3]

        gamma = torch.bmm(gamma_s.view(B_g,C_g,H_g*W_g), weight)  # n*HW*1
        beta = torch.bmm(beta_s.view(B_b, C_b, H_b*W_b), weight)
        gamma = gamma.view(B_g,C_g,H_g,W_g)
        beta = beta.view(B_b, C_b, H_b, W_b)
        # gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)   gama复制三份，weight的batchsize也是3，眼睛嘴巴脸  # (3, HW, HW)
        # beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) combine the three parts
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    def get_weight(self, mask_c, mask_s, fea_c, fea_s):  ##mask(3, 1, 256, 256)  diff（3，136，64，64）
        """  s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 64, 64)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None  # fea_s when i==3
        # get 3 part fea using mask

        B, channel_num,H,W = fea_s.shape[0], fea_s.shape[1],fea_s.shape[2],fea_s.shape[3]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(1, channel_num, 1,
                                                          1)  ##mask(3, 1, 256, 256)-->线性插值 (3, 1, 64, 64)-->(3, 192, 64, 64)
        # fea_c = normal(fea_c)
        fea_c = fea_c - fea_c.mean(dim=(2, 3), keepdim=True)
        fea_c = fea_c.view(B, channel_num, HW)
        fea_c = fea_c / torch.norm(fea_c, dim=1, keepdim=True)  # 把对应通道的对应值平方在开根号
        fea_c=fea_c.view(B, channel_num, H,W)
        fea_c = fea_c.repeat(3, 1, 1, 1)  # (1, 384, 64, 64)-->(3, c, h, w)
        fea_c = fea_c * mask_c_re  # (3, c, h, w) 3 stands for 3 parts   *逐像素相乘

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        # fea_s = normal(fea_s)
        fea_s = fea_s - fea_s.mean(dim=(2, 3), keepdim=True)
        fea_s = fea_s.view(B, channel_num, HW)
        fea_s = fea_s / torch.norm(fea_s, dim=1, keepdim=True)  # 把对应通道的对应值平方在开根号
        fea_s = fea_s.view(B, channel_num, H, W)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re  # 分眼睛嘴巴脸，分别算注意力
        theta_target = fea_c.view(batch_size, -1, HW)  # (N, C+136, H*W)
        #theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C+136)
        phi_source = fea_s.view(batch_size, -1, HW)  # (N, C+136, H*W)
        phi_source = phi_source.permute(0, 2, 1)  # (N, H*W, C+136)

        #weight = torch.bmm(theta_target, phi_source)  # (3, HW, HW)
        weight = torch.bmm(phi_source,theta_target)  # (3, HW, HW)
        weight *= 200
        # with torch.no_grad():
        #     v = weight.detach().nonzero().long().permute(1, 0)
        #     # This clone is required to correctly release cuda memory.
        #     weight_ind = v.clone()
        #     del v
        #     # torch.cuda.empty_cache()
        #
        # weight *= 200  # hyper parameters for visual feature
        # weight = F.softmax(weight, dim=-1)
        # weight = weight[weight_ind[0], weight_ind[1], weight_ind[2]]
        # ret = torch.sparse.FloatTensor(weight_ind, weight, torch.Size([3, HW, HW]))  ####################
        

        weight = F.softmax(weight, dim=2)


        return weight

    def forward(self, mask_non_makeup, mask_makeup, z_non_makeup_c, z_makeup_c,z_non_makeup_s, z_makeup_s):  # 输入两张图片
        s, gamma, beta = self.simple_spade(z_makeup_s)
        weight = self.get_weight(mask_non_makeup, mask_makeup, z_non_makeup_c, z_makeup_c)
        #print("weight:",weight.size())
        # gamma, beta = self.atten_feature(mask_makeup, weight, gamma, beta, self.atten_bottleneck_g,
        #                                  self.atten_bottleneck_b)
        gamma, beta = self.atten_feature(mask_makeup, weight, gamma, beta)

        for i in range(self.n_blocks):
            z_non_makeup_c = getattr(self, 'UpBlock1_' + str(i + 1))(z_non_makeup_c, gamma, beta)

        return z_non_makeup_c  # 给没妆的上妆结果，给有妆的卸妆结果


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.encode_with_intermediate = FaceEncoder(in_chans=3,
                                                    patch_size=2,  # 下采样倍率
                                                    window_size=8,  # MLP
                                                    dim=64,
                                                    depths=(6, 6, 6, 6,6),
                                                    num_heads=(8, 8, 8, 8,8),
                                                    norm_layer=nn.LayerNorm,
                                                    )

        self.in_dim = 256

        ######################分别通过各自的图提取内容特征和妆容特征
        self.CSA = Content_SA(
            self.in_dim)  #######已完成，按照ACMMM2020保留白化操作进一步分离内容特征和妆容特征，保证提的内容特征很纯粹 里面的conv都是1*1卷积，non-local，dim不变 384 32*32->64*64
        self.SSA = Style_SA(self.in_dim)  #######FPT 32*32->64*64 dim=192

        # transform,warp_makeup+content
        self.ma_module = Multi_Adaptation_Module(self.in_dim)  # 384通道

        # Down-Sampling
        n_downsampling=2
        curr_dim = 256
        # ngf = 64
        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            UpBlock2 += [
                         nn.ConvTranspose2d(curr_dim, curr_dim// 2, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.InstanceNorm2d(curr_dim // 2, affine=True),
                         nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2


        UpBlock2 += [
                     nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
                     nn.Tanh()
                     ]

        self.decoder = nn.Sequential(*UpBlock2)


    def forward(self, mask_c, mask_s, non_makeup, makeup,phase="train"):  # 输入两张素颜图，两张化妆图，一共四张图片
        mask_c, mask_s, non_makeup, makeup = [x.squeeze(0) if x.ndim == 5 else x
                                                                                      for x in
                                                                                      [mask_c, mask_s, non_makeup, makeup]]
        
        if phase=="test":
            z_non_makeup = self.encode_with_intermediate(non_makeup)  # 1张图但是后面也需要复制8遍
            z_makeup = self.encode_with_intermediate(makeup)

            # 分别提内容特征和妆容特征
            z_non_makeup_c = self.CSA(z_non_makeup)
            z_makeup_c = self.CSA(z_makeup)

            z_non_makeup_s = self.SSA(z_non_makeup)
            z_makeup_s = self.SSA(z_makeup)

            # 给没妆的上妆结果，给有妆的卸妆结果
            z_makeup_warp = self.ma_module(mask_c, mask_s, z_non_makeup_c, z_makeup_c, z_non_makeup_s,z_makeup_s)
            z_non_makeup_warp = self.ma_module(mask_s, mask_c, z_makeup_c, z_non_makeup_c, z_makeup_s,z_non_makeup_s)
            # 解码器恢复图片
            p_transfer = self.decoder(z_makeup_warp)  # c1s1
            p_removal = self.decoder(z_non_makeup_warp)  # s1c1
            
            return p_transfer, p_removal
            
        ####################
            
        # 两组图片提基础特征，只用于解纠缠
        z_non_makeup = self.encode_with_intermediate(non_makeup)  # 1张图但是后面也需要复制8遍
        z_makeup = self.encode_with_intermediate(makeup)

        # 分别提内容特征和妆容特征
        z_non_makeup_c = self.CSA(z_non_makeup)
        z_makeup_c = self.CSA(z_makeup)

        z_non_makeup_s = self.SSA(z_non_makeup)
        z_makeup_s = self.SSA(z_makeup)

        # 给没妆的上妆结果，给有妆的卸妆结果
        z_makeup_warp = self.ma_module(mask_c, mask_s, z_non_makeup_c, z_makeup_c, z_non_makeup_s,z_makeup_s)
        z_non_makeup_warp = self.ma_module(mask_s, mask_c, z_makeup_c, z_non_makeup_c, z_makeup_s,z_non_makeup_s)
        # 解码器恢复图片
        p_transfer = self.decoder(z_makeup_warp)  # c1s1
        p_removal = self.decoder(z_non_makeup_warp)  # s1c1

        z_rec_non_makeup_warp = self.ma_module(mask_c, mask_c, z_non_makeup_c, z_non_makeup_c,z_non_makeup_s, z_non_makeup_s)
        z_rec_makeup_warp = self.ma_module(mask_s, mask_s, z_makeup_c, z_makeup_c, z_makeup_s, z_makeup_s)

        # rec恢复图像
        p_rec_non_makeup = self.decoder(z_rec_non_makeup_warp)
        p_rec_makeup = self.decoder(z_rec_makeup_warp)

        z_removal = self.encode_with_intermediate(p_removal)  # 1张图
        z_transfer = self.encode_with_intermediate(p_transfer)

        # 分别提内容特征和妆容特征
        z_removal_c = self.CSA(z_removal)
        z_transfer_c = self.CSA(z_transfer)

        z_removal_s = self.SSA(z_removal)
        z_transfer_s = self.SSA(z_transfer)

        z_transfer_warp = self.ma_module(mask_s, mask_c, z_removal_c, z_transfer_c,z_removal_s, z_transfer_s)
        z_removal_warp = self.ma_module(mask_c, mask_s, z_transfer_c, z_removal_c,z_transfer_s, z_removal_s)
        # 解码器恢复图片
        p_cycle_makeup = self.decoder(z_transfer_warp)
        p_cycle_non_makeup = self.decoder(z_removal_warp)
        
        
            

        return p_transfer, p_removal, p_rec_non_makeup, p_rec_makeup, p_cycle_makeup, p_cycle_non_makeup


# conv + (spectral) + (instance) + leakyrelu
class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# 妆容迁移模块
class ResnetAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias, use_se=False):
        super(ResnetAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaLIN(dim)
        self.use_se = use_se
        if use_se:
            self.se = ChannelSpatialSELayer(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        #print(out.shape)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        if self.use_se:
            out = self.se(out)
        return out + x

class AdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, in_x, gamma, beta):
        in_mean, in_var = torch.mean(in_x, dim=[2, 3], keepdim=True), torch.var(in_x, dim=[2, 3], keepdim=True)
        out_in = (in_x - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(in_x, dim=[1, 2, 3], keepdim=True), torch.var(in_x, dim=[1, 2, 3], keepdim=True)
        out_ln = (in_x - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(in_x.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(in_x.shape[0], -1, -1, -1)) * out_ln
        #out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        # out=out_in
        out = out * gamma + beta
        return out

class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks,
        MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=4):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        attention = self.cSE(input_tensor) + self.sSE(input_tensor)
        return attention


class ChannelSELayer(nn.Module):
    def __init__(self, in_size, reduction=4, min_hidden_channel=8):
        super(ChannelSELayer, self).__init__()

        hidden_channel = max(in_size // reduction, min_hidden_channel)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, hidden_channel, kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, in_size, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.se(x) * x


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
        return squeeze_tensor * input_tensor

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

