import torch
import torch.nn as nn
import torch.nn.init as init
import logging
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.attention import Attention
from diffusers.utils import USE_PEFT_BACKEND
from typing import Optional

from einops import rearrange

logger = logging.getLogger(__name__)


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
            scale: float = 1.0,
            pose_feature=None
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRAAttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(
            self,
            hidden_size=None,
            cross_attention_dim=None,
            rank=4,
            network_alpha=None,
            lora_scale=1.0,
    ):
        super().__init__()

        self.rank = rank
        self.lora_scale = lora_scale

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            pose_feature=None,
            scale=None
    ):
        lora_scale = self.lora_scale if scale is None else scale
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + lora_scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)

    def forward(self,
                attn,
                hidden_states,
                pose_feature,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=None,):
        assert pose_feature is not None
        pose_embedding_scale = (scale or self.scale)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature.ndim == 5:
            pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
        elif pose_feature.ndim == 4:
            pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
        else:
            assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.query_condition and self.key_value_condition:  # only self attention
            query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = query_hidden_state
        elif self.query_condition:
            query_hidden_state = self.q_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = encoder_hidden_states
        else:
            key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * pose_embedding_scale + encoder_hidden_states
            query_hidden_state = hidden_states

        # original attention
        query = attn.to_q(query_hidden_state)
        key = attn.to_k(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LORAPoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0,
                 # lora keywords
                 rank=4,
                 network_alpha=None,
                 lora_scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)
        # lora
        self.rank = rank
        self.lora_scale = lora_scale
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(self,
                 attn,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None,
                 scale=1.0,
                 pose_feature=None,
                 ):
        assert pose_feature is not None
        lora_scale = self.lora_scale if scale is None else scale
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature.ndim == 5:
            pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
        elif pose_feature.ndim == 4:
            pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
        else:
            assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.query_condition and self.key_value_condition:  # only self attention
            query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * self.scale + hidden_states
            key_value_hidden_state = query_hidden_state
        elif self.query_condition:
            query_hidden_state = self.q_merge(hidden_states + pose_feature) * self.scale + hidden_states
            key_value_hidden_state = encoder_hidden_states
        else:
            key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * self.scale + encoder_hidden_states
            query_hidden_state = hidden_states

        # original attention
        query = attn.to_q(query_hidden_state) + lora_scale * self.to_q_lora(query_hidden_state)
        key = attn.to_k(key_value_hidden_state) + lora_scale * self.to_k_lora(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state) + lora_scale * self.to_v_lora(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
