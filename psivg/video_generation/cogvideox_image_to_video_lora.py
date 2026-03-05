# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import logging
import math
import os
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from mpmath import fac

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers import (
    # AutoencoderKLCogVideoX, ## we need more functionality
    CogVideoXDPMScheduler,
    # CogVideoXImageToVideoPipeline, ## we need more functionality
    # CogVideoXPipeline,  ## we need more functionality
    # CogVideoXTransformer3DModel, # we need more functionality
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel


from args import get_args  # isort:skip
from dataset import (
    VideoDatasetWithResizingAndTTT,
)  # isort:skip
from text_encoder import compute_prompt_embeddings  # isort:skip
from utils import (
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    get_all_tile_pixel_regions,
    compute_fg_percentages,
    pad_tile_to_shape
)  # isort:skip

import rp
import rp.r_iterm_comm as ric

import numpy as np
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from psivg.video_generation.dataset import get_downtemp_noise

rp.r._pip_import_autoyes=True #Automatically install missing packages
rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import rp.git.CommonSource.noise_warp as nw

import torch.utils.checkpoint
import torch.nn.functional as F

from psivg.video_generation.vae.our_vae import AutoencoderKLCogVideoX
from psivg.video_generation.text_encoder.text_encoder import compute_prompt_embeddings_with_fgindices_output #, _get_t5_prompt_embeds_original
from psivg.video_generation.transformer.our_transformer import CogVideoXTransformer3DModel


logger = get_logger(__name__)


def log_validation(
    accelerator: Accelerator,
    pipe,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    epoch,
    is_final_validation: bool = False,
    output_dir_videoid = None,
):
    
    if args.optimize_prompt:
        logger.info(
            f"Running validation... \n Generating {args.num_validation_videos} videos with prompt ebmbedding."
        )
    elif not args.optimize_prompt:
        logger.info(
            f"Running validation... \n Generating {args.num_validation_videos} videos with prompt embedding."
        )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(output_dir_videoid, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    return videos



class CollateFunction:
    def __init__(self, weight_dtype: torch.dtype) -> None:
        self.weight_dtype = weight_dtype

    def __call__(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        prompts = [x["prompt"] for x in data]

        images = [x["image"] for x in data]
        images = torch.stack(images).to(dtype=self.weight_dtype, non_blocking=True)

        videos = [x["video"] for x in data]
        videos = torch.stack(videos).to(dtype=self.weight_dtype, non_blocking=True)

        masks = [x["masks"] for x in data]
        masks = torch.stack(masks).to(dtype=self.weight_dtype, non_blocking=True)

        output = {
            "images": images,
            "videos": videos,
            "prompts": prompts,
            "masks": masks,
        }

        noises = [x["noise"] for x in data]
        noises = torch.stack(noises).to(dtype=self.weight_dtype, non_blocking=True)
        noises_downtemp = [x["noise_downtemp"] for x in data]
        noises_downtemp = torch.stack(noises_downtemp).to(dtype=self.weight_dtype, non_blocking=True)
        output |= {
            "noises": noises,
            "noises_downtemp" : noises_downtemp,
        }

        return output




def main(args):

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if args.setting_name:
        setting_name = args.setting_name

    expt_name_parts = [
        "expt_LRcosine",
        f"lr{args.learning_rate}",
        f"iter{args.max_train_steps}",
    ]
    expt_name = "_".join(expt_name_parts)
    expt_name_videoid = f"{expt_name}/{args.video_id}"
    output_dir_videoid = os.path.join(args.output_dir, setting_name, expt_name_videoid)
    print("Saving the outputs of this setting and video ID in:", output_dir_videoid)

    logging_dir = Path(args.output_dir, setting_name, expt_name_videoid, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=output_dir_videoid, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir_videoid is not None:
            os.makedirs(output_dir_videoid, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(output_dir_videoid).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing() 
        vae.enable_gradient_checkpointing() 
        print("Enabled gradient checkpointing for Transformer and VAE")

    # Now we will add the Go-with-the-Flow pre-trained LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)


    #######  Load the Go-with-the-Flow pre-trained LoRA weights
    lora_weights_path = os.path.join(PROJECT_ROOT, args.lora_weights_path)
    
    from diffusers.loaders import LoraLoaderMixin
    lora_state_dict = LoraLoaderMixin.lora_state_dict(lora_weights_path)[0]
    transformer_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith("transformer.")}
    transformer_state_dict = {k.replace("transformer.", ""): v for k, v in transformer_state_dict.items()}
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")

    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            logger.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    print("Loaded pre-trained LoRA weights from:", lora_weights_path)


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            CogVideoXImageToVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(model).__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    for name, param in transformer.named_parameters():
        if param.requires_grad:  ### All lora layers from 42 transformer blocks
            param.requires_grad = False

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)


    if args.use_TTCO and args.optimize_prompt:
        prompt_file = os.path.join(args.data_root, str(args.video_id), args.caption_column)
        prompt_fg_file = os.path.join(args.data_root, str(args.video_id), args.caption_fg_column)


        ### logic for loading the prompts
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        input_prompt = first_line
                        print(f"Found input prompt: {input_prompt}")
            except Exception as e:
                print(f"Error reading prompt.txt: {e}")

        if os.path.exists(prompt_fg_file):
            try:
                with open(prompt_fg_file, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        foreground_prompt = first_line
                        print(f"Found foreground prompt: {foreground_prompt}")
            except Exception as e:
                print(f"Error reading prompt_fg.txt: {e}")

        ### To compute the indices of the foreground prompt in the main prompt
        prompt_embeds, sequence_idx = compute_prompt_embeddings_with_fgindices_output(
            tokenizer,
            text_encoder,
            input_prompt,
            foreground_prompt,
            226, # model_config.max_text_seq_length,
            accelerator.device,
            weight_dtype,
            requires_grad=False,
        )

        prompt_embeds.requires_grad_(False)

        # Create optimizable tensor for foreground indices
        embed_dim = prompt_embeds.shape[-1]
        num_fg_tokens = len(sequence_idx)
        fg_embeds_delta = torch.zeros((num_fg_tokens, embed_dim), device=prompt_embeds.device, dtype=prompt_embeds.dtype, requires_grad=True)
        prompt_embeds_opt = prompt_embeds.clone()  # [1, 226, 4096]
        prompt_embeds_opt[0, sequence_idx, :] = prompt_embeds_opt[0, sequence_idx, :] + fg_embeds_delta

        ### Initialize the other optimizable tokens for intermediate layers here.
        if args.optimize_prompt_each_layer:
            num_layers = 42
            attn_embed_dim = 3072

            layer_wise_fg_embeds_delta = torch.zeros(
                (num_layers, num_fg_tokens, attn_embed_dim),
                device=prompt_embeds.device,
                dtype=prompt_embeds.dtype,
                requires_grad=True,
            )
            layer_wise_fg_embeds_delta.requires_grad_(True)

    else:
        
        fg_embeds_delta = None ## Dont need this here. Just initialize to None.

    if args.use_TTCO and args.optimize_prompt: 
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    params_to_optimize = []
    num_trainable_parameters_prompt = 0
    if args.use_TTCO and args.optimize_prompt:
        params_to_optimize.append(fg_embeds_delta)
        if args.optimize_prompt_each_layer:
            params_to_optimize.append(layer_wise_fg_embeds_delta)
        num_trainable_parameters_prompt = sum(param.numel() for param in params_to_optimize)

    if not args.use_TTCO: ### for not TTCO, have a dummy parameter here to make it run.
        dummy_optimizer_param = None
        if len(params_to_optimize) == 0:
            # Keep optimizer/scheduler initialization valid for no-training flows (e.g. use_TTCO=False).
            dummy_optimizer_param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
            params_to_optimize.append(dummy_optimizer_param)

    print("num_trainable_parameters_prompt:", num_trainable_parameters_prompt)
    num_trainable_parameters = num_trainable_parameters_prompt


    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )


    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )


    ### Added to select the correct video's folder from the dataset
    data_root = os.path.join(args.data_root, str(args.video_id))

    dataset_init_kwargs = {
        "data_root": data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": False,
        "random_flip": args.random_flip,
        "image_to_video": True,
        "noises_column": args.noises_column,
        "masks_column":args.masks_column,
        "degradation": args.degradation, ## 0.5
        "use_moving_camera": args.use_moving_camera,
    }


    rp.fansi_print(f'dataset_init_kwargs={dataset_init_kwargs}','green','bold')

    train_dataset = VideoDatasetWithResizingAndTTT(**dataset_init_kwargs)

    collate_fn = CollateFunction(weight_dtype)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        shuffle=True, ## added
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )


    # Prepare everything with `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = args.resume_from_checkpoint
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir_videoid)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            rp.fansi_print(f"Resuming from checkpoint {path}", 'yellow','bold','black')
            accelerator.load_state(path)

            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)


    ### To save space, set it to None, so we compute it on the first iteration only
    if not args.optimize_prompt:
        prompt_embeds = None

    regions = get_all_tile_pixel_regions(
        latent_height=60,
        latent_width=90,
        tile_latent_min_height=30,
        tile_latent_min_width=45,
        overlap_height=25, # 25
        overlap_width=36, # 36
        downsample_factor=8,
        video_height=480,
        video_width=720,
    )

    ### manually load the masks first
    mask_data_path = Path(train_dataset.masks_paths[0])

    _, fg_masks, _ = train_dataset._preprocess_mask_video(mask_data_path)

    #### if we decode fully the video, we will face memory issues. So we selectively decode only the regions with the highest foreground percentages.
    fg_masks = fg_masks.unsqueeze(0)
    fg_masks = fg_masks.permute(0, 2, 1, 3, 4)
    if fg_masks.shape[1] == 1:
        fg_masks = fg_masks.expand(-1, 3, -1, -1, -1)

    #### computing the indices of the regions to selectively decode here. Using regions with the highest foreground percentages.
    fg_percentages = compute_fg_percentages(fg_masks, regions)

    # Finding the top 5 indices, or all non-zero areas (if there are less than 5 non-zero regions).
    fg_percentages_tensor = torch.tensor(fg_percentages)
    nonzero_indices = (fg_percentages_tensor > 0).nonzero(as_tuple=True)[0]

    if len(nonzero_indices) >= 5:
        _, top5_indices_in_nonzero = torch.topk(fg_percentages_tensor[nonzero_indices], 5)
        top5_indices = nonzero_indices[top5_indices_in_nonzero]
        selected_indices = top5_indices.tolist()
    else:
        selected_indices = nonzero_indices.tolist()

    # If TTCO is disabled, run a single one-off validation with the original prompt and exit.
    if not args.use_TTCO:
        if accelerator.is_main_process:
            torch.cuda.synchronize(accelerator.device)

            import glob
            if args.use_moving_camera:
                noises_dir = os.path.join(args.data_root, str(args.video_id), "merged_noises")
            elif not args.use_moving_camera:
                noises_dir = os.path.join(args.data_root, str(args.video_id), "noises")

            validation_noises = None
            if os.path.exists(noises_dir):
                noise_files = glob.glob(os.path.join(noises_dir, "*"))
                if noise_files:
                    validation_noises = noise_files[0]
                    print(f"Found validation noises: {validation_noises}")
            if validation_noises is None:
                raise FileNotFoundError(f"No validation noises found in {noises_dir}")

            validation_noise = np.load(validation_noises)
            validation_noise = torch.tensor(validation_noise).permute(0, 3, 1, 2).to(weight_dtype)
            validation_noise = get_downtemp_noise(validation_noise, args.noise_downtemp_interp)
            validation_noise = validation_noise[None]
            validation_noise = nw.mix_new_noise(validation_noise, args.degradation)

            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=unwrap_model(transformer),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            if args.enable_tiling:
                pipe.vae.enable_tiling()
            pipe = pipe.to(accelerator.device)

            input_image_dir = os.path.join(args.data_root, str(args.video_id), "input_image")
            validation_image = None
            if os.path.exists(input_image_dir):
                input_image_files = glob.glob(os.path.join(input_image_dir, "*"))
                if input_image_files:
                    validation_image = input_image_files[0]
                    print(f"Found validation image: {validation_image}")
            if validation_image is None:
                raise FileNotFoundError(f"No validation input image found in {input_image_dir}")

            prompt_file = os.path.join(args.data_root, str(args.video_id), args.caption_column)
            validation_prompt = ""
            if os.path.exists(prompt_file):
                with open(prompt_file, "r") as f:
                    validation_prompt = f.readline().strip()
            if not validation_prompt:
                # Fallback to dataset prompt if prompt file is missing/empty.
                validation_prompt = train_dataset[0]["prompt"]

            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            pipeline_args = {
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "latents": validation_noise,
                "num_inference_steps": 30,
                "image": load_image(validation_image),
            }

            with torch.no_grad():
                video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]

            base_name = os.path.splitext(os.path.basename(validation_image))[0]
            val_epoch_path = os.path.join(output_dir_videoid, "validation_once")
            os.makedirs(val_epoch_path, exist_ok=True)
            mp4_path = os.path.join(val_epoch_path, base_name + ".mp4")
            export_to_video(video, mp4_path, fps=8)
            print(f"Saved one-off validation video as mp4 to {mp4_path}")

            reset_memory(accelerator.device)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(accelerator.device)

        accelerator.wait_for_everyone()
        accelerator.end_training()
        print("Code successfully completed!")
        return
        
    ### TTCO is enabled
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                
                images = batch["images"].to(accelerator.device, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                prompts = batch["prompts"]
                noises_downtemp = batch["noises_downtemp"].to(accelerator.device, non_blocking=True)
                masks = batch["masks"].to(accelerator.device, non_blocking=True)

                # Encode videos
                images = images.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                image_noise_sigma = torch.normal(
                    mean=-3.0, std=0.5, size=(images.size(0),), device=accelerator.device, dtype=weight_dtype
                )
                image_noise_sigma = torch.exp(image_noise_sigma)
                noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]

                image_latent_dist = vae.encode(noisy_images).latent_dist

                videos = videos.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
                latent_dist = vae.encode(videos).latent_dist

                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)


                # Encode prompts
                if args.optimize_prompt: ## the prompt is already generated. just need to put it in

                    prompt_embeds_opt = prompt_embeds.clone()  # [1, 226, 4096]
                    prompt_embeds_opt[0, sequence_idx, :] = prompt_embeds_opt[0, sequence_idx, :] + fg_embeds_delta
                    prompt_embeds_opt = prompt_embeds_opt.to(dtype=weight_dtype)


                elif not args.optimize_prompt: ## just generate the prompt in the first iteration
                    if prompt_embeds is None: # first time computing the prompt embeddings. If have memory, we can also compute it everytime.
                        prompt_embeds = compute_prompt_embeddings(
                            tokenizer,
                            text_encoder,
                            prompts,
                            model_config.max_text_seq_length,
                            accelerator.device,
                            weight_dtype,
                            requires_grad=False,
                        )

                    prompt_embeds.requires_grad_(False)

                    del text_encoder
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(accelerator.device)

                # Sample noise that will be added to the latents
                noise = noises_downtemp.to(video_latents.dtype).to(video_latents.device) # using warped noise
                batch_size, num_frames, num_channels, height, width = video_latents.shape

                #### randomly sample
                if args.timestep_sampling=="uniform_random":
                    timesteps = torch.randint(
                        0,
                        scheduler.config.num_train_timesteps,
                        (batch_size,),
                        dtype=torch.int64,
                        device=accelerator.device,
                    )

                ### to schedule and focus on modelling the starting noisy parts, so that we can steer it in the right direction first.
                elif args.timestep_sampling=="noisy_steps":
                    timesteps = torch.randint(
                        args.noise_step_thresh,
                        1000,
                        (batch_size,),
                        dtype=torch.int64,
                        device=accelerator.device,
                    )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=num_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)

                if IS_I2V:
                    noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                else:
                    assert IS_T2V
                    noisy_model_input = noisy_video_latents

                transformer_args = {
                    "hidden_states": noisy_model_input,
                    "timestep": timesteps,
                    "image_rotary_emb": image_rotary_emb,
                    "return_dict": False,
                }

                if args.optimize_prompt:
                    transformer_args["encoder_hidden_states"] = prompt_embeds_opt

                    if args.optimize_prompt_each_layer:
                        transformer_args["optimize_prompt_each_layer"] = args.optimize_prompt_each_layer
                        transformer_args["layer_wise_fg_embeds_delta"] = layer_wise_fg_embeds_delta
                        transformer_args["sequence_idx"] = sequence_idx

                elif not args.optimize_prompt:
                    transformer_args["encoder_hidden_states"] = prompt_embeds

                model_output = transformer(**transformer_args)[0]
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                model_pred = model_pred.to(dtype=weight_dtype)

                # To decode to video, and apply the loss in the pixel space, using the masks
                latents = model_pred.permute(0, 2, 1, 3, 4) / VAE_SCALING_FACTOR

                masks = masks.permute(0, 2, 1, 3, 4)
                if masks.shape[1] == 1:
                    masks = masks.expand(-1, videos.shape[1], -1, -1, -1)

                selected_regions = tuple([regions[i] for i in selected_indices])                
                model_pred_decoded = vae.decode_selected(latents, selected_regions).sample


                # To select the patches to apply the loss to 
                video_decoded_selected = []
                masks_selected = []
                for region in selected_regions:
                    pixel_h_start, pixel_h_end = region["pixel_h"]
                    pixel_w_start, pixel_w_end = region["pixel_w"]
                    # Extract the region for all batches, frames, and channels
                    cropped = videos[..., pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end]
                    cropped = pad_tile_to_shape(cropped, 240, 360)
                    video_decoded_selected.append(cropped.squeeze(0))
                    
                    cropped_mask = masks[..., pixel_h_start:pixel_h_end, pixel_w_start:pixel_w_end]
                    cropped_mask = pad_tile_to_shape(cropped_mask, 240, 360)
                    masks_selected.append(cropped_mask.squeeze(0))

                video_decoded_selected = torch.stack(video_decoded_selected, dim=0)                
                masks_selected = torch.stack(masks_selected, dim=0)


                loss = 0.0

                import torch.nn.functional as F

                # To be resistant to noise, we remove a ring of pixels. Find foreground areas (value of 1) and remove 2-pixel ring around outside
                foreground_areas = (masks_selected == 1)

                # Create a kernel for erosion to shrink the foreground by 2 pixels from inside
                erosion_kernel_size = 5  # 2*2 + 1 to get 2-pixel erosion
                erosion_padding = 2

                # Erode the foreground areas to remove the inner 2-pixel ring
                # Use avg_pool2d and threshold to perform erosion
                eroded_foreground = F.avg_pool2d(
                    foreground_areas.float().view(-1, 1, foreground_areas.shape[-2], foreground_areas.shape[-1]), 
                    kernel_size=erosion_kernel_size, 
                    stride=1, 
                    padding=erosion_padding
                ).view(foreground_areas.shape)

                # Convert to boolean - only keep pixels that are fully surrounded by foreground
                eroded_foreground_bool = (eroded_foreground > 0.99)  # Threshold to ensure full kernel coverage

                # Update masks_selected to remove the 2-pixel ring from inside foreground
                # The ring to remove is the original foreground minus the eroded foreground
                ring_to_remove = foreground_areas & ~eroded_foreground_bool
                rgb_mask_with_ring = masks_selected.clone()
                rgb_mask_with_ring[ring_to_remove] = 0

                pixel_loss = (model_pred_decoded - video_decoded_selected) ** 2
                masked_pixel_loss = pixel_loss * rgb_mask_with_ring
                loss_masked = masked_pixel_loss.sum() / (rgb_mask_with_ring.sum() + 1e-8)

                loss += loss_masked * weights.item() * args.TTCO_loss_lambda 
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer: #called
                    lr_scheduler.step()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir_videoid)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir_videoid, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)


                        save_path = os.path.join(output_dir_videoid, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True) ## added

                        save_dict = {}
                        if args.optimize_prompt:

                            save_dict["fg_embeds_delta"] = fg_embeds_delta.detach().cpu()
                            save_dict["sequence_idx"] = sequence_idx

                        if args.optimize_prompt_each_layer:

                            save_dict["layer_wise_fg_embeds_delta"] = layer_wise_fg_embeds_delta.detach().cpu()

                        torch.save(save_dict, os.path.join(save_path, "fg_embeds_delta_and_indices.pt"))


            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs = {"loss": loss.detach().item(), "lr": last_lr}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)


            if global_step >= args.max_train_steps:
                break


        # Validation
        if accelerator.is_main_process:
            if (epoch + 1) % args.validation_epochs == 0:
                torch.cuda.synchronize(accelerator.device)


                import glob
                if args.use_moving_camera:
                    noises_dir = os.path.join(args.data_root, str(args.video_id), "merged_noises")
                elif not args.use_moving_camera:
                    noises_dir = os.path.join(args.data_root, str(args.video_id), "noises")

                if os.path.exists(noises_dir):
                    noise_files = glob.glob(os.path.join(noises_dir, "*"))
                    if noise_files:
                        validation_noises = noise_files[0]  # Take the first (and should be only) file
                        print(f"Found validation noises: {validation_noises}")


                ## Added: to get the validation noise
                validation_noise = np.load(validation_noises)
                validation_noise = torch.tensor(validation_noise).permute(0,3,1,2).to(weight_dtype)
                validation_noise = get_downtemp_noise(validation_noise, args.noise_downtemp_interp)
                validation_noise = validation_noise[None]
                validation_noise = nw.mix_new_noise(validation_noise, args.degradation)


                pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    # scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                if args.enable_tiling:
                    pipe.vae.enable_tiling()

                input_image_dir = os.path.join(args.data_root, str(args.video_id), "input_image")
                if os.path.exists(input_image_dir):
                    input_image_files = glob.glob(os.path.join(input_image_dir, "*"))
                    if input_image_files:
                        validation_image = input_image_files[0]  # Take the first (and should be only) file
                        print(f"Found validation image: {validation_image}")

                pipeline_args = {
                    "prompt_embeds": prompt_embeds,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                    "latents": validation_noise,
                    "num_inference_steps": 30,
                }


                if args.optimize_prompt:
                    # Put the optimized prompt_embeds_opt in the pipeline here
                    prompt_embeds_opt = prompt_embeds.clone()  # [1, 226, 4096]
                    prompt_embeds_opt[0, sequence_idx, :] = prompt_embeds_opt[0, sequence_idx, :] + fg_embeds_delta
                    prompt_embeds_opt = prompt_embeds_opt.to(dtype=weight_dtype)
                    pipeline_args["prompt_embeds"] = prompt_embeds_opt


                    if args.optimize_prompt_each_layer:
                        pipeline_args["optimize_prompt_each_layer"] = args.optimize_prompt_each_layer
                        pipeline_args["layer_wise_fg_embeds_delta"] = layer_wise_fg_embeds_delta
                        pipeline_args["sequence_idx"] = sequence_idx

                if IS_I2V:
                    pipeline_args.update({"image": load_image(validation_image)})

                with torch.no_grad():                    
                    video = log_validation(
                        accelerator=accelerator,
                        pipe=pipe,
                        args=args,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                        output_dir_videoid=output_dir_videoid
                    )


                # Save the video to the output directory
                video = video[0]
                base_name = os.path.splitext(os.path.basename(validation_image))[0]
                val_epoch_path = os.path.join(output_dir_videoid, f"validation_epoch{epoch}")
                os.makedirs(val_epoch_path, exist_ok=True)

                # Also save as mp4 for visualization
                mp4_path = os.path.join(val_epoch_path, base_name + ".mp4")
                export_to_video(video, mp4_path, fps=8)
                print(f"Saved validation video as mp4 to {mp4_path}")

                reset_memory(accelerator.device)

                del pipe
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(accelerator.device)

    accelerator.wait_for_everyone()
    accelerator.end_training()

    save_dict = {}
    if args.optimize_prompt:
        save_dict["fg_embeds_delta"] = fg_embeds_delta.detach().cpu()
        save_dict["sequence_idx"] = sequence_idx

    if args.optimize_prompt_each_layer:
        save_dict["layer_wise_fg_embeds_delta"] = layer_wise_fg_embeds_delta.detach().cpu()

    torch.save(save_dict, os.path.join(output_dir_videoid, "fg_embeds_delta_and_indices.pt"))
    print("Code successfully completed!")

if __name__ == "__main__":
    args = get_args()

    ric.process_args.update(args.__dict__) #Is updated in the ...lora.py script
    IS_I2V = True

    from psivg.video_generation.pipeline.our_pipeline import CogVideoXImageToVideoPipeline
    

    main(args)

