#!/bin/bash -l

mamba activate PSIVG_env3

### This script processes the data from the perception pipeline to the dataset for videogen

export USE_MOVING_CAMERA="true"
# export USE_MOVING_CAMERA="false"

FOLDER_NAME="data_root"

#### prepared inputs
SELECTED_VIDS="${FOLDER_NAME}/OUT_Flow/selected/selected_examples.txt"
PROMPT_FILE="${FOLDER_NAME}/INPUT_DATA/Prompts/0009.txt"  
PROMPT_FG_FILE="${FOLDER_NAME}/INPUT_DATA/Prompts/0009_fg.txt"  


### TODO. check if this path to the first frame is correct
IMAGE_FOLDER="${FOLDER_NAME}/INPUT_DATA/Frames"
RENDER_DATA_FOLDER="${FOLDER_NAME}/OUT_Rendering"


TEMPLATE_VIDEO_FOLDER="${FOLDER_NAME}/INPUT_DATA/Videos"

#### output directories
OUTPUT_DIR="${FOLDER_NAME}/OUT_Flow/computed_noises" 
MASK_FIRSTFRAME_FOLDER="${FOLDER_NAME}/OUT_Flow/segmaps_firstframe/masks_npy"
OUTPUT_SEGMENTATION_DIR_IMAGES="${FOLDER_NAME}/OUT_Flow/segmaps_firstframe"

OUTPUT_DATASET_DIR="${FOLDER_NAME}/datasets/generated_data_example"

# To segment static images (first frames) using LangSAM (filter by selected videos and prompts)
mamba deactivate
mamba activate langsam
IMAGE_DIR="${IMAGE_FOLDER}"
SELECTED_VIDEOS_FILE="${SELECTED_VIDS}"

echo "Starting image segmentation..."
echo "Image dir: ${IMAGE_DIR}"
echo "Output dir: ${OUTPUT_SEGMENTATION_DIR_IMAGES}"
python psivg/utils/segment_frames.py \
  --image_dir ${IMAGE_DIR} \
  --output_dir ${OUTPUT_SEGMENTATION_DIR_IMAGES} \
  --selected_videos ${SELECTED_VIDEOS_FILE} \
  --text_prompts ${PROMPT_FG_FILE}
echo "Image segmentation completed!"




# here, need to warp the noise with the optical flow and generate the noise.npy. store it
mamba deactivate
mamba activate PSIVG_env3

python psivg/utils/make_warped_noise.py \
  --selected_vids_file ${SELECTED_VIDS} \
  --input_folder ${RENDER_DATA_FOLDER} \
  --output_folder ${OUTPUT_DIR} \
  --first_frame_folder ${IMAGE_FOLDER} \
  --mask_firstframe_folder ${MASK_FIRSTFRAME_FOLDER}
echo "Warped noise completed!"


#to generate the pixel correspondences
python psivg/utils/process_pixel_correspondences.py \
    --selected_vids_file ${SELECTED_VIDS} \
    --input_folder ${RENDER_DATA_FOLDER} \
    --output_folder ${OUTPUT_DIR} \
    --first_frame_folder ${IMAGE_FOLDER} \
    --mask_firstframe_folder ${MASK_FIRSTFRAME_FOLDER}
echo "Pixel correspondences completed!"




# To get the masks and the flow for the background, to handle the moving camera 
if [ "$USE_MOVING_CAMERA" = "true" ]; then
  echo "Using moving camera, calculating background flow and masks"


    python psivg/utils/make_warped_noise_background.py \
      --input_folder_templatevideo ${TEMPLATE_VIDEO_FOLDER} \
      --output_dir ${OUTPUT_DIR} \
      --selected_vids_file ${SELECTED_VIDS} 
    echo "Background flow and masks completed!"


    # next, to get the masks for the template videos with moving camera
    mamba deactivate
    mamba activate langsam

    python psivg/utils/segment_video_frames.py \
      --input_folder ${OUTPUT_DIR} \
      --text_prompt ${PROMPT_FG_FILE} \
      --frame_rate 8  \
      --output_dir ${OUTPUT_DIR}  \
      --selected_vids_file ${SELECTED_VIDS} 
    echo "Masks for the template videos with moving camera completed!"


    # next, to merge the flows together using segmentation masks
    mamba deactivate
    mamba activate PSIVG_env3


    python psivg/utils/merge_flows_noises.py \
      --input_dir ${OUTPUT_DIR} \
      --flow_threshold 2.0  \
      --selected_vids_file ${SELECTED_VIDS}  
    echo "Merged flows and generated merged noise completed!"

fi



# then, we organize the data and transfer it to the dataset format
TRANSFER_FLAGS=""
if [ "$USE_MOVING_CAMERA" = "true" ]; then
  TRANSFER_FLAGS="${TRANSFER_FLAGS} --with_merged_noises"
fi

python psivg/utils/transfer_to_dataset.py  \
  --input_dir ${OUTPUT_DIR}  \
  --output_dataset_dir ${OUTPUT_DATASET_DIR}  \
  --prompt_file ${PROMPT_FILE}  \
  --prompt_fg_file ${PROMPT_FG_FILE}  \
  --selected_vids_file ${SELECTED_VIDS}  \
  --image_folder ${IMAGE_FOLDER}  \
  --with_correspondences  \
  ${TRANSFER_FLAGS}
echo "Transfer outputs to dataset completed!"



#### for interactive session, for a100 and for h100
### srun -p gpu22 --pty --gres gpu:1 -t 1:00:00  /bin/bash
### srun -p gpu22 --pty --gres gpu:a100:1 -t 1:00:00  /bin/bash
### srun -p gpu24 --pty --gres gpu:h100:1 -t 1:00:00  /bin/bash








