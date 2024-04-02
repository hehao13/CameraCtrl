# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

from einops import rearrange, repeat
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.activations import get_activation
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm


class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class InflatedGroupNorm(nn.GroupNorm):
    def forward(self, x):
        # return super().forward(x)

        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


class FusionBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="default",  # default, scale_shift, ada_group, spatial
            kernel=None,
            output_scale_factor=1.0,
            use_in_shortcut=None,
            up=False,
            down=False,
            conv_shortcut_bias: bool = True,
            conv_2d_out_channels: Optional[int] = None,

            zero_init=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True

        in_channels = in_channels * 2
        self.in_channels = in_channels

        out_channels = in_channels * 3 if out_channels is None else out_channels * 3
        self.out_channels = out_channels

        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = torch.nn.Conv2d(out_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0, bias=conv_shortcut_bias
            )

        conv_out = torch.nn.Conv2d(
            conv_2d_out_channels, conv_2d_out_channels, kernel_size=1, stride=1, padding=0,
        )
        self.conv_out = zero_module(conv_out) if zero_init else conv_out

    def forward(self, init_hidden_state, post_hidden_states, temb):
        # init_hidden_state:  b c   1   h w
        # post_hidden_states: b c (f-1) h w

        video_length = post_hidden_states.shape[2]
        repeated_init_hidden_state = repeat(init_hidden_state, "b c f h w -> b c (n f) h w", n=video_length)

        hidden_states = torch.cat([repeated_init_hidden_state, post_hidden_states], dim=1)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        input_tensor = hidden_states

        if temb.shape[0] != input_tensor.shape[0]:
            temb = repeat(temb, "b c -> (b n) c", n=input_tensor.shape[0] // temb.shape[0])
        assert temb.shape[0] == input_tensor.shape[0], f"{temb.shape}, {input_tensor.shape}"

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        output_tensor = self.conv_out(output_tensor)

        output_tensor = rearrange(output_tensor, "(b f) c h w -> b c f h w", f=video_length)
        scale_1, scale_2, shift = output_tensor.chunk(3, dim=1)

        # output_tensor = (1 + scale_1) * repeated_init_hidden_state + scale_2 * post_hidden_states + shift
        output_tensor = scale_1 * repeated_init_hidden_state + (1 + scale_2) * post_hidden_states + shift

        return output_tensor

class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # if self.use_conv:
        #     if self.name == "conv":
        #         hidden_states = self.conv(hidden_states)
        #     else:
        #         hidden_states = self.Conv2d_0(hidden_states)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="swish",
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, temb):
        # input: b c f h w

        hidden_states = input_tensor

        video_length = hidden_states.shape[2]
        emb = repeat(emb, "b c -> (b f) c", f=video_length)

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))