from typing import List, Optional, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer




### Our function to get the prompt embeddings.
def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds





def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds


#### Used when we do not optimize the prompt.
def compute_prompt_embeddings(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds


#### Besides getting the prompt encoding, we also get the indices of where the foreground tokens are
def compute_prompt_embeddings_with_fgindices_output(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: str,
    foreground_prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
):
    if requires_grad:
        prompt_embeds, sequence_idx = encode_prompt_with_fgindices_output(
            tokenizer,
            text_encoder,
            prompt,
            foreground_prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds, sequence_idx = encode_prompt_with_fgindices_output(
                tokenizer,
                text_encoder,
                prompt,
                foreground_prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds, sequence_idx



#### Helper function to get the indices of where the foreground  tokens are.
def encode_prompt_with_fgindices_output(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    foreground_prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    foreground_prompt = [foreground_prompt] if isinstance(foreground_prompt, str) else foreground_prompt

    batch_size = len(prompt)

    if batch_size > 1:
        raise NotImplementedError("The encoding of prompts with fgindices is not implemented for a batch size bigger than 1. Just need to add a for-loop in this case, to get the indices.")

    if tokenizer is not None:

        fg_text_inputs = tokenizer(
            foreground_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids

        # For batch size 1 (extend to batch if needed)
        fg_ids = fg_text_inputs.input_ids
        fg_mask = fg_text_inputs.attention_mask
        main_ids = text_inputs.input_ids[0]

        # Get the sequence of tokens where attention_mask == 1
        fg_active_ids = fg_ids[fg_mask == 1]
        fg_active_ids = fg_active_ids[:-1] ## remove the end of sequence token

        # Find where this sequence appears in main_ids
        sequence_idx = find_subsequence_indices(main_ids, fg_active_ids)

    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, sequence_idx



def find_subsequence_indices(main_ids, sub_ids):
    """Find the start index of sub_ids in main_ids. Returns -1 if not found."""
    n, m = len(main_ids), len(sub_ids)
    for i in range(n - m + 1):
        
        if torch.equal(main_ids[i:i+m], sub_ids):
            return list(range(i, i+m))
        
    raise ValueError("Foreground tokens not found in the main token sequence!!!")
    return -1


