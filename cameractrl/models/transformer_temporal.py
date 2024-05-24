import torch
import torch.nn as nn

from typing import Optional
from diffusers.models.transformer_temporal import TransformerTemporalModelOutput
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.resnet import AlphaBlender
from cameractrl.models.attention import TemporalPoseCondTransformerBlock


class TransformerSpatioTemporalModelPoseCond(nn.Module):
    """
        A Transformer model for video-like data.

        Parameters:
            num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
            attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
            in_channels (`int`, *optional*):
                The number of channels in the input and output (specify if the input is **continuous**).
            out_channels (`int`, *optional*):
                The number of channels in the output (specify if the input is **continuous**).
            num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
            cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        """

    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: int = 320,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalPoseCondTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,            # [bs * frame, c, h, w]
        encoder_hidden_states: Optional[torch.Tensor] = None,        # [bs * frame, 1, c]
        image_only_indicator: Optional[torch.Tensor] = None,         # [bs, frame]
        pose_feature: Optional[torch.Tensor] = None,                # [bs, c, frame, h, w]
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a
                plain tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states        # [bs * frame, 1, c]
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]     # [bs, frame, c]
        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        )           # [bs, h*w, 1, c]
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1])    # [bs * h * w, 1, c]

        residual = hidden_states

        hidden_states = self.norm(hidden_states)        # [bs * frame, c, h, w]
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)  # [bs * frame, h * w, c]
        hidden_states = self.proj_in(hidden_states)     # [bs * frame, h * w, c]

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)       # [bs, frame]
        num_frames_emb = num_frames_emb.reshape(-1)     # [bs * frame]
        t_emb = self.time_proj(num_frames_emb)          # [bs * frame, c]

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]       # [bs * frame, 1, c]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,          # [bs * frame, h * w, c]
                    encoder_hidden_states=encoder_hidden_states,    # [bs * frame, 1, c]
                )               # [bs * frame, h * w, c]

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,      # [bs * frame, h * w, c]
                num_frames=num_frames,
                encoder_hidden_states=time_context,     # [bs * h * w, 1, c]
                pose_feature=pose_feature
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
