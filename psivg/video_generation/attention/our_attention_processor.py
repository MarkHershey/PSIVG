
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


from diffusers.models.attention import Attention


class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # print("IN OUR OWN COGVIDEOX ATTN PROCESSOR")

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # print("hidden states shape", hidden_states.shape) ## torch.Size([1, 17776, 3072])

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # print("attn.to_q", attn.to_q) # lora.Linear, wrapping around a nn.Linear module
        # print("attn.to_k", attn.to_k) # lora.Linear, wrapping around a nn.Linear module
        # print("attn.to_v", attn.to_v) # lora.Linear, wrapping around a nn.Linear module

        ### lora.Linear(
        #   (base_layer): Linear(in_features=3072, out_features=3072, bias=True)
        #   (lora_dropout): ModuleDict(
        #     (default): Identity()
        #   )
        #   (lora_A): ModuleDict(
        #     (default): Linear(in_features=3072, out_features=2048, bias=False)
        #   )
        #   (lora_B): ModuleDict(
        #     (default): Linear(in_features=2048, out_features=3072, bias=False)
        #   )
        #   (lora_embedding_A): ParameterDict()
        #   (lora_embedding_B): ParameterDict()
        #   (lora_magnitude_vector): ModuleDict()
        # )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        ### 3072 =  48* 64. we are splitting it into smaller parts
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # print("query, key and value shape:", query.shape, key.shape, value.shape) ## all are torch.Size([1, 48, 17776, 64])
        # print("attn.norm_q is:",attn.norm_q) ## LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        # print("attn.norm_k is:",attn.norm_k) ## LayerNorm((64,), eps=1e-06, elementwise_affine=True)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        # print("image rotary emb", image_rotary_emb[0].shape, image_rotary_emb[1].shape) ## tuple of torch.Size([17550, 64]) torch.Size([17550, 64])
        if image_rotary_emb is not None:
            # from .embeddings import apply_rotary_emb
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            # print("attn.is_cross_attention", attn.is_cross_attention)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # print("hidden_states shape before splitting", hidden_states.shape)  ## torch.Size([1, 17776, 3072])

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        # print("hidden_states shape after splitting", hidden_states.shape) ## torch.Size([1, 17550, 3072])
        # print("encoder_hidden_states shape after splitting", encoder_hidden_states.shape) ## torch.Size([1, 226, 3072])

        return hidden_states, encoder_hidden_states
