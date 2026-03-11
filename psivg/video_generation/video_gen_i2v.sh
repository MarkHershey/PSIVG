export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0

GPU_IDS="0"

mkdir -p $output_dir

# Single GPU uncompiled training
ACCELERATE_CONFIG_FILE="configs/uncompiled_1.yaml" #Default: "configs/uncompiled_1.yaml"


CAPTION_COLUMN="prompt.txt"
CAPTION_FG_COLUMN="prompt_fg.txt"
VIDEO_COLUMN="videos.txt"
NOISES_COLUMN="noises.txt"
MASKS_COLUMN="masks.txt"

RANK=2048
LORA_ALPHA=$RANK

NUM_DATAWORKERS=0 #0 means on main thread. 
GRADIENT_ACCUMULATION_STEPS=1 #Default: 1

# Launch experiment 
IFS=',' read -ra VIDEO_IDS <<< "$VIDEOS"
# Iterate over each video ID
for video_id in "${VIDEO_IDS[@]}"; do

    cmd="accelerate launch --config_file $ACCELERATE_CONFIG_FILE --gpu_ids $GPU_IDS psivg/video_generation/cogvideox_image_to_video_lora.py \
    $( if [[ -n $RESUME_FROM_CHECKPOINT ]]; then echo "--resume_from_checkpoint $RESUME_FROM_CHECKPOINT"; fi ) \
    --pretrained_model_name_or_path $BASE_MODEL_NAME \
    --data_root $DATA_ROOT \
    --caption_column $CAPTION_COLUMN \
    --video_column $VIDEO_COLUMN \
    --height_buckets 480 \
    --width_buckets 720 \
    --frame_buckets 49 \
    --dataloader_num_workers $NUM_DATAWORKERS \
    --pin_memory \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_epochs $VALIDATION_EPOCHS \
    --seed 1 \
    --rank $RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_weights_path $LORA_WEIGHTS_PATH \
    --mixed_precision bf16 \
    --output_dir $output_dir \
    --max_num_frames 49 \
    --train_batch_size 1 \
    --max_train_steps $MAX_TRAIN_STEPS \
    --checkpointing_steps $CHECKPOINTING_STEPS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing \
    --learning_rate $LEARNING_RATES \
    --lr_scheduler $LR_SCHEDULES \
    --lr_warmup_steps 0 \
    --lr_num_cycles 0.5 \
    --noised_image_dropout 0.05 \
    --optimizer $OPTIMIZERS \
    --enable_slicing \
    --enable_tiling \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.001 \
    --max_grad_norm 1.0 \
    --allow_tf32 \
    --nccl_timeout 1800 \
    --noise_downtemp_interp "nearest" \
    --noises_column $NOISES_COLUMN \
    --masks_column $MASKS_COLUMN \
    --degradation $DEGRADATION \
    --video_id $video_id \
    $( if [[ "$USE_TTCO" == "true" ]]; then echo "--use_TTCO"; fi ) \
    --TTCO_loss_lambda $TTCO_LOSS_LAMBDA \
    --timestep_sampling  $TIMESTEP_SAMPLING \
    --noise_step_thresh  $NOISE_STEP_THRESH  \
    --setting_name $SETTING_NAME  \
    --optimize_prompt   \
    --optimize_prompt_each_layer  \
    --caption_fg_column $CAPTION_FG_COLUMN \
    $( if [[ "$USE_MOVING_CAMERA" == "true" ]]; then echo "--use_moving_camera"; fi ) 
    "


    echo "Running command for video_id $video_id: $cmd"
    eval $cmd
    echo -ne "-------------------- Finished executing script for video_id $video_id --------------------\n\n"
done


