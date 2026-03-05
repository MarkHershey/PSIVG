import gc
import inspect
from typing import Optional, Tuple, Union

import torch
from accelerate.logging import get_logger
from diffusers.models.embeddings import get_3d_rotary_pos_embed

import torch.nn.functional as F

logger = get_logger(__name__)


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay,
        )

    if use_8bit and use_4bit:
        raise ValueError("Cannot set both `use_8bit` and `use_4bit` to True.")

    if (use_torchao and (use_8bit or use_4bit)) or use_cpu_offload_optimizer:
        try:
            import torchao

            torchao.__version__
        except ImportError:
            raise ImportError(
                "To use optimizers from torchao, please install the torchao library: `USE_CPP=0 pip install torchao`."
            )

    if not use_torchao and use_4bit:
        raise ValueError("4-bit Optimizers are only supported with torchao.")

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy", "came"]
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if (use_8bit or use_4bit) and optimizer_name not in ["adam", "adamw"]:
        raise ValueError("`use_8bit` and `use_4bit` can only be used with the Adam and AdamW optimizers.")

    if use_8bit:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer_name == "adamw":
        if use_torchao:
            from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

            optimizer_class = AdamW8bit if use_8bit else AdamW4bit if use_4bit else torch.optim.AdamW
        else:
            optimizer_class = bnb.optim.AdamW8bit if use_8bit else torch.optim.AdamW

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }


    elif optimizer_name == "adam":
        if use_torchao:
            from torchao.prototype.low_bit_optim import Adam4bit, Adam8bit

            optimizer_class = Adam8bit if use_8bit else Adam4bit if use_4bit else torch.optim.Adam
        else:
            optimizer_class = bnb.optim.Adam8bit if use_8bit else torch.optim.Adam

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        init_kwargs = {
            "lr": learning_rate,
            "betas": (beta1, beta2),
            "beta3": beta3,
            "eps": epsilon,
            "weight_decay": weight_decay,
            "decouple": prodigy_decouple,
            "use_bias_correction": prodigy_use_bias_correction,
            "safeguard_warmup": prodigy_safeguard_warmup,
        }

    elif optimizer_name == "came":
        try:
            import came_pytorch
        except ImportError:
            raise ImportError("To use CAME, please install the came-pytorch library: `pip install came-pytorch`")

        optimizer_class = came_pytorch.CAME

        init_kwargs = {
            "lr": learning_rate,
            "eps": (1e-30, 1e-16),
            "betas": (beta1, beta2, beta3),
            "weight_decay": weight_decay,
        }

    if use_cpu_offload_optimizer:
        from torchao.prototype.low_bit_optim import CPUOffloadOptimizer

        if "fused" in inspect.signature(optimizer_class.__init__).parameters:
            init_kwargs.update({"fused": True})

        optimizer = CPUOffloadOptimizer(
            params_to_optimize, optimizer_class=optimizer_class, offload_gradients=offload_gradients, **init_kwargs
        )
    else:
        optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


def get_gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


def reset_memory(device: Union[str, torch.device]) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)


def print_memory(device: Union[str, torch.device]) -> None:
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    print(f"{memory_allocated=:.3f} GB")
    print(f"{max_memory_allocated=:.3f} GB")
    print(f"{max_memory_reserved=:.3f} GB")




def get_all_tile_pixel_regions(
    latent_height, latent_width, tile_latent_min_height, tile_latent_min_width,
    overlap_height, overlap_width, downsample_factor, video_height, video_width
):
    """
    Returns a list of (tile_h_start, tile_h_end, tile_w_start, tile_w_end, pixel_h_start, pixel_h_end, pixel_w_start, pixel_w_end)
    for all tiles.
    """
    regions = []
    for i in range(0, latent_height, overlap_height):
        for j in range(0, latent_width, overlap_width):
            tile_h_start = i
            tile_h_end = min(i + tile_latent_min_height, latent_height)
            tile_w_start = j
            tile_w_end = min(j + tile_latent_min_width, latent_width)
            pixel_h_start, pixel_h_end, pixel_w_start, pixel_w_end = get_video_pixel_region_for_tile(
                tile_h_start, tile_h_end, tile_w_start, tile_w_end, downsample_factor, video_height, video_width
            )
            regions.append({
                "tile_h": (tile_h_start, tile_h_end),
                "tile_w": (tile_w_start, tile_w_end),
                "pixel_h": (pixel_h_start, pixel_h_end),
                "pixel_w": (pixel_w_start, pixel_w_end),
            })
    return regions


def get_video_pixel_region_for_tile(
    tile_h_start, tile_h_end, tile_w_start, tile_w_end, downsample_factor, video_height, video_width
):
    """
    Returns the pixel region in the video corresponding to a tile in latent space.
    """
    pixel_h_start = tile_h_start * downsample_factor
    pixel_h_end = min(tile_h_end * downsample_factor, video_height)
    pixel_w_start = tile_w_start * downsample_factor
    pixel_w_end = min(tile_w_end * downsample_factor, video_width)
    return pixel_h_start, pixel_h_end, pixel_w_start, pixel_w_end



def compute_fg_percentages(masks, regions, threshold=0.5):
    """
    Compute the percentage of foreground pixels in the mask for each region.

    Args:
        masks: torch.Tensor, shape (B, C, T, H, W)
        regions: list of dicts, each with keys 'pixel_h' and 'pixel_w'
        threshold: float, value above which a pixel is considered foreground

    Returns:
        fg_percentages: list of floats, one per region
    """
    B, C, T, H, W = masks.shape
    fg_percentages = []

    # For each region, compute the foreground percentage
    for region in regions:
        pixel_h_start, pixel_h_end = region["pixel_h"]
        pixel_w_start, pixel_w_end = region["pixel_w"]

        # Extract the region from the mask (all batches, channels, and frames)
        mask_region = masks[..., pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end]

        # Compute foreground mask
        fg_mask = (mask_region == 1).float()
        fg_pixels = fg_mask.sum().item()
        total_pixels = mask_region.numel()
        fg_percentage = fg_pixels / (total_pixels)
        fg_percentages.append(fg_percentage)

    return fg_percentages


def pad_tile_to_shape(tile: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """
    Pads the input tile tensor to the target height and width with zeros if necessary.
    Args:
        tile (torch.Tensor): The input tensor of shape (N, C, T, H, W).
        target_height (int): The desired height.
        target_width (int): The desired width.
    Returns:
        torch.Tensor: The padded tensor of shape (N, C, T, target_height, target_width).
    """
    _, _, _, h, w = tile.shape
    pad_h = target_height - h
    pad_w = target_width - w
    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # (w_left, w_right, h_left, h_right)
        tile = F.pad(tile, padding)
    return tile