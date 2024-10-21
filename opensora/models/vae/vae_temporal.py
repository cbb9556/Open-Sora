from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint

from .utils import DiagonalGaussianDistribution


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


def exists(v):
    return v is not None


class CausalConv3d(nn.Module):
    """
    3D 因果卷积层。

    该层通过对输入数据在时间维度上进行特定的填充，确保在时刻 t 的输出仅依赖于时刻 t 及之前的输入，
    实现单向依赖性。这在处理序列数据时特别有用，例如视频中的帧序列。

    参数:
    - chan_in (int): 输入通道数。
    - chan_out (int): 输出通道数。
    - kernel_size (int | Tuple[int, int, int]): 卷积核的大小，可以是整数或包含三个整数的元组，
      分别表示时间、高度和宽度维度的卷积核大小。
    - pad_mode (str, optional): 填充模式，默认为 'constant'，表示用零填充。
    - strides (Tuple[int, int, int], optional): 步幅大小，默认为 None，表示使用卷积层的默认步幅。
    - **kwargs: 其他传递给 nn.Conv3d 的参数。

    注意:
    - 该类继承自 nn.Module，是 PyTorch 的 3D 卷积层的包装。
    - 使用时，输入数据应具有形状 (batch_size, channels, time, height, width)。
    """

    def __init__(
            self,
            chan_in,
            chan_out,
            kernel_size: Union[int, Tuple[int, int, int]],
            pad_mode="constant",
            strides=None,  # allow custom stride
            **kwargs,
    ):
        super().__init__()
        # 将 kernel_size 转换为三元组形式
        kernel_size = cast_tuple(kernel_size, 3)

        # 分别获取时间、高度和宽度维度的卷积核大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度维度的卷积核大小为奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 获取膨胀率，默认为 1，感受野：随着膨胀率的增加，卷积核能够覆盖更大的输入区域，从而增加其感受野
        # 空洞卷积
        dilation = kwargs.pop("dilation", 1)
        # 获取步幅，默认为 1
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        self.pad_mode = pad_mode

        # 计算在时间维度上的填充量
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        # 计算在高度和宽度维度上的填充量
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        # 定义时间因果填充，确保输出仅依赖于当前及之前的输入
        self.time_causal_padding = (width_pad, width_pad,
                                    height_pad, height_pad, time_pad, 0)

        # 设置卷积层的步幅和膨胀率
        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)

        # 初始化卷积层，3d卷积
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        """
        前向传播函数。

        对输入数据进行填充，然后应用卷积操作。

        参数:
        - x (Tensor): 输入张量，形状为 (batch_size, channels, time, height, width)。

        返回:
        - Tensor: 经过因果卷积后的输出张量。
        """
        # 对输入数据进行填充
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        # 应用卷积操作
        x = self.conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        # 组正则化， 组内计算均值和方差，正则化组内的数值
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        # CausalConv3d 就是 conv_fn通过参数 dict指针传递
        self.conv1 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
        self.norm2 = nn.GroupNorm(num_groups, self.filters)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), bias=False)

        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
            else:
                # 直接路径，恒等映射，输出维度与输入维度一致：缓解梯度消失问题、保持信息传递：
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(1, 1, 1), bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # SiLU门控激活函数
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            # conv3 对残差 x进行 卷积操作 还是恒等映射
            residual = self.conv3(residual)
        return x + residual


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation == "swish":
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


class Encoder(nn.Module):
    """Encoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
            bias=False,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 1
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters, filters, kernel_size=(3, 3, 3), strides=(t_stride, s_stride, s_stride)
                        )
                    )
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    # # Identity层的作用是在网络中传递输入而不进行任何变换，常用于残差网络中的跳过连接
                    self.conv_blocks.append(nn.Identity(prev_filters))  # Identity
                    prev_filters = filters  # update in_channels

        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)

        self.conv2 = self.conv_fn(prev_filters, self.embedding_dim, kernel_size=(1, 1, 1), padding="same")

    def forward(self, x):
        # 3d卷积
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(self.embedding_dim, filters, kernel_size=(3, 3, 3), bias=True)

        # last layer res block
        self.res_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters, **self.block_args))

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.ModuleList([])
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters, prev_filters * t_stride * self.s_stride * self.s_stride, kernel_size=(3, 3, 3)
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(prev_filters),
                    )

        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


@MODELS.register_module()
class VAE_Temporal(nn.Module):
    """
    定义一个处理时间数据的变分自编码器(VAE)类。
    这个类继承自nn.Module，包含了编码器和解码器，用于处理三维数据（如视频），并使用对角高斯分布作为潜在空间的后验分布。
    """

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        """
        初始化VAE_Temporal类。

        参数:
        - in_out_channels (int): 输入和输出的通道数。
        - latent_embed_dim (int): 潜在空间的维度。
        - embed_dim (int): 嵌入空间的维度。
        - filters (int): 卷积层的滤波器数量。
        - num_res_blocks (int): 残差块的数量。
        - channel_multipliers (tuple): 通道倍增器的元组，用于不同尺度的卷积层。
        - temporal_downsample (tuple): 时间下采样的元组，表示是否在每个尺度上进行时间维度的下采样。
        - num_groups (int): 组归一化的组数。
        - activation_fn (str): 激活函数的名称。
        """
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        # self.time_padding = self.time_downsample_factor - 1
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels

        # NOTE: following cc, conv in bias=False in encoder first conv
        # 对于 3D VAE，我们采用Magvit-v2中的 VAE 结构
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )

    def get_latent_size(self, input_size):
        """
        根据输入尺寸计算潜在空间的尺寸。

        参数:
        - input_size (tuple): 输入数据的尺寸。

        返回:
        - latent_size (list): 潜在空间的尺寸。
        """
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - input_size[i] % self.time_downsample_factor
                )
                lsize = (input_size[i] + time_padding) // self.patch_size[i]
            else:
                lsize = input_size[i] // self.patch_size[i]
            latent_size.append(lsize)
        return latent_size

    def encode(self, x):
        """
        对输入数据进行编码。

        此函数首先根据时间维度的下采样因子计算需要在时间维度上填充的量，以确保时间维度的大小能够被下采样因子整除。
        然后在时间维度上对输入数据进行相应的填充。接着，使用编码器对填充后的数据进行编码，生成编码后的特征。
        最后，通过全连接层对编码后的特征进行处理，生成高斯分布的参数，用于构建对角高斯分布的后验分布，并将其返回。

        参数:
        - x (Tensor): 输入的数据张量。

        返回:
        - DiagonalGaussianDistribution: 生成的对角高斯分布的后验分布。
        """
        # 计算时间维度的填充量，以确保时间维度的大小能够被时间下采样因子整除，nt = T/f
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        # 在时间维度上对输入数据进行相应的填充
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        # 使用编码器对填充后的数据进行编码，生成编码后的特征
        encoded_feature = self.encoder(x)
        # 通过全连接层对编码后的特征进行处理，生成高斯分布的参数
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        # 使用生成的高斯分布参数构建对角高斯分布的后验分布，并将其返回
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, num_frames=None):
        """
        对潜在变量进行解码。

        参数:
        - z (Tensor): 潜在变量的张量。
        - num_frames (int): 帧的数量，用于计算时间维度的填充量。

        返回:
        - x (Tensor): 解码后的数据张量。
        """
        # 计算时间维度的填充量
        # 如果帧数能被时间维度的下采样因子整除，则不需要额外的填充，填充量为0
        # 否则，计算需要多少帧才能达到下一个最接近的、能被时间维度下采样因子整除的帧数
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )

        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    # 在 VAE 的重参数化技巧中，通过引入外部噪声源 ϵ，潜在变量
    # z 通过一个确定性的函数从均值 μ、方差 σ 和 ϵ 计算得到，即 z = μ + σ ⊙ ϵ。
    # 这样的随机化处理使得模型中的所有操作都变成了可微分的，从而可以使用基于梯度的优化方法进行训练
    # sample_posterior就是 对 encoder的 输出进行重参数化（实现从latent采样），使得 出现玄月
    def forward(self, x, sample_posterior=True):
        """
        VAE_Temporal类的前向传播函数。

        参数:
        - x (Tensor): 输入的数据张量。
        - sample_posterior (bool): 是否从后验分布中采样。

        返回:
        - recon_video (Tensor): 重建的视频数据张量。
        - posterior (DiagonalGaussianDistribution): 对角高斯分布的后验分布。
        - z (Tensor): 潜在变量的张量。
        """
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z, num_frames=x.shape[2])
        return recon_video, posterior, z


@MODELS.register_module("VAE_Temporal_SD")
def VAE_Temporal_SD(from_pretrained=None, **kwargs):
    model = VAE_Temporal(
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        **kwargs,
    )
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
