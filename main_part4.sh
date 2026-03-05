#!/bin/bash -l

mamba activate PSIVG_env3


export LORA_WEIGHTS_PATH="./pretrained_models/I2V5B_final_i38800_nearest_lora_weights.safetensors"
export BASE_MODEL_NAME="./pretrained_models/CogVideoX-5b-I2V"

export output_dir="./outputs"

export TIMESTEP_SAMPLING="noisy_steps"
export NOISE_STEP_THRESH="700"
export DEGRADATION="0.5"

export LR_SCHEDULES="cosine"
export OPTIMIZERS="adamw"
export LEARNING_RATES="2e-4"  
export TTCO_LOSS_LAMBDA="10"


export MAX_TRAIN_STEPS="50"
export VALIDATION_EPOCHS="50"
export CHECKPOINTING_STEPS="10"

# export USE_TTCO="true"
export USE_TTCO="false"

export USE_MOVING_CAMERA="true"
# export USE_MOVING_CAMERA="false"



export DATA_ROOT="data_root/datasets/generated_data_example"


export VIDEOS="0009"


export SETTING_NAME="03032026_testingnewenv_merged"




./psivg/video_generation/video_gen_i2v.sh






#### for interactive session, for a100 and for h100
### srun -p gpu22 --pty --gres gpu:a100:1 -t 1:00:00  /bin/bash
### srun -p gpu24 --pty --gres gpu:h100:1 -t 1:00:00  /bin/bash

