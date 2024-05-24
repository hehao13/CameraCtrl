import torch
from typing import Optional
from diffusers.models.attention import TemporalBasicTransformerBlock, _chunked_feed_forward
from diffusers.utils.torch_utils import maybe_allow_in_graph


@maybe_allow_in_graph
class TemporalPoseCondTransformerBlock(TemporalBasicTransformerBlock):
    def forward(
        self,
        hidden_states: torch.FloatTensor,   # [bs * num_frame, h * w, c]
        num_frames: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # [bs * h * w, 1, c]
        pose_feature: Optional[torch.FloatTensor] = None,       # [bs, c, n_frame, h, w]
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)  # [bs * h * w, frame, c]

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        pose_feature = pose_feature.permute(0, 3, 4, 2, 1).reshape(batch_size * seq_length, num_frames, -1)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None, pose_feature=pose_feature)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states, pose_feature=pose_feature)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states