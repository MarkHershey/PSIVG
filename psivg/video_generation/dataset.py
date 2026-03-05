# import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
# import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset #, Sampler
from torchvision import transforms
# from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
# from torchvision.io import read_image

import rp
import einops

rp.r._pip_import_autoyes=True #Automatically install missing packages
rp.git_import('CommonSource') #If missing, installs code from https://github.com/RyannDaGreat/CommonSource
import rp.git.CommonSource.noise_warp as nw

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

HEIGHT_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
WIDTH_BUCKETS = [256, 320, 384, 480, 512, 576, 720, 768, 960, 1024, 1280, 1536]
FRAME_BUCKETS = [16, 24, 32, 48, 64, 80]


def get_downtemp_noise(noise, noise_downtemp_interp):
    assert noise_downtemp_interp in {'nearest', 'blend', 'blend_norm', 'randn'}, noise_downtemp_interp
    if   noise_downtemp_interp == 'nearest'    : return                  rp.resize_list(noise, 13)
    elif noise_downtemp_interp == 'blend'      : return                   downsamp_mean(noise, 13)
    elif noise_downtemp_interp == 'blend_norm' : return normalized_noises(downsamp_mean(noise, 13))
    elif noise_downtemp_interp == 'randn'      : return torch.randn_like(rp.resize_list(noise, 13)) #Basically no warped noise, just r
    else: assert False, 'impossible'



def downsamp_mean(x, l=13):
    return torch.stack([rp.mean(u) for u in rp.split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])


import rp.r_iterm_comm as ric
ric.process_args = {} #Is updated in the ...lora.py script


######## Our Own!!! To edit the call function according to the new dataset structure
class VideoDatasetWithResizingAndTTT(Dataset):


    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
        noises_column: str = "noises",
        masks_column: str = "masks",
        degradation: float = "0.5",
        use_moving_camera: bool = False,
    ) -> None:
        rp.fansi_print("dataset.VideoDatasetWithResizingAndTTT: INITIALIZING!",'green','bold')
        Dataset.__init__(self)

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = id_token or ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.masks_column = masks_column
        self.degradation = degradation

        self.use_moving_camera = use_moving_camera
        if self.use_moving_camera:
            self.noises_column = "merged_noises.txt" # hard coded for now
        elif not self.use_moving_camera:
            self.noises_column = noises_column



        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]


        if dataset_file is None:

            (
                self.prompts,
                self.video_paths,
                self.noises_paths,
                self.masks_paths,
                self.video_original_path,
                self.video_original_masks_path,
                self.layered_masks_paths,
            ) = self._load_dataset_from_local_path()
        else:
            raise NotImplementedError("Currently only implemented for local path loading.")

        if len(self.video_paths) != len(self.prompts):
            raise ValueError(
                f"Expected length of prompts and videos to be the same but found {len(self.prompts)=} and {len(self.video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )

        self.video_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(random_flip)
                if random_flip
                else transforms.Lambda(self.identity_transform),
                transforms.Lambda(self.scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_preprocessed_latents_and_embeds(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        filename_without_ext = path.name.split(".")[0]
        pt_filename = f"{filename_without_ext}.pt"

        image_latents_path = path.parent.parent.joinpath("image_latents")
        video_latents_path = path.parent.parent.joinpath("video_latents")
        embeds_path = path.parent.parent.joinpath("prompt_embeds")

        if (
            not video_latents_path.exists()
            or not embeds_path.exists()
            or (self.image_to_video and not image_latents_path.exists())
        ):
            raise ValueError(
                f"When setting the load_tensors parameter to `True`, it is expected that the `{self.data_root=}` contains two folders named `video_latents` and `prompt_embeds`. However, these folders were not found. Please make sure to have prepared your data correctly using `prepare_data.py`. Additionally, if you're training image-to-video, it is expected that an `image_latents` folder is also present."
            )

        if self.image_to_video:
            image_latent_filepath = image_latents_path.joinpath(pt_filename)
        video_latent_filepath = video_latents_path.joinpath(pt_filename)
        embeds_filepath = embeds_path.joinpath(pt_filename)

        if not video_latent_filepath.is_file() or not embeds_filepath.is_file():
            if self.image_to_video:
                image_latent_filepath = image_latent_filepath.as_posix()
            video_latent_filepath = video_latent_filepath.as_posix()
            embeds_filepath = embeds_filepath.as_posix()
            raise ValueError(
                f"The file {video_latent_filepath=} or {embeds_filepath=} could not be found. Please ensure that you've correctly executed `prepare_dataset.py`."
            )

        images = (
            torch.load(image_latent_filepath, map_location="cpu", weights_only=True) if self.image_to_video else None
        )
        latents = torch.load(video_latent_filepath, map_location="cpu", weights_only=True)
        embeds = torch.load(embeds_filepath, map_location="cpu", weights_only=True)

        return images, latents, embeds



    def __getitem__(self, index: int) -> Dict[str, Any]:

        if isinstance(index, list): ## deprecated
            return index

        else:

            image, video, _ = self._preprocess_video(self.video_paths[index])
            
            _, masks, _ = self._preprocess_mask_video(self.masks_paths[index])

            # Loading and processing the noise
            noise_downtemp_interp = "nearest"
            noise_data = np.load(self.noises_paths[index])
            noise_data = torch.Tensor(noise_data).float()

            noise_data = einops.rearrange(noise_data, 'T H W C -> T C H W') ### torch.Size([49, 16, 60, 90])
            noise_downtemp = get_downtemp_noise(noise_data, noise_downtemp_interp)
            noise_downtemp = nw.mix_new_noise(noise_downtemp, self.degradation)

            output = {
                "prompt": self.id_token + self.prompts[index],
                "image": image,
                "video": video,
                "video_metadata": {
                    "num_frames": video.shape[0],
                    "height": video.shape[2],
                    "width": video.shape[3],
                    "noise_downtemp_interp": noise_downtemp_interp
                },
                "noise": noise_data,
                "noise_downtemp": noise_downtemp,
                "masks": masks,
            }

            return output



    def _load_dataset_from_local_path(self) -> Tuple[List[str], List[str]]:
        if not self.data_root.exists():
            raise ValueError("Root folder for videos does not exist")

        prompt_path = self.data_root.joinpath(self.caption_column)
        video_path = self.data_root.joinpath(self.video_column)
        noises_path = self.data_root.joinpath(self.noises_column)
        masks_path = self.data_root.joinpath(self.masks_column)

        video_original_path = self.data_root.joinpath("video_original.mp4")
        video_original_masks_path = self.data_root.joinpath("original_video_masks_npy")

        video_original_path = None
        video_original_masks_path = None


        layered_masks_paths = None

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--data_root` containing line-separated paths to video data in the same directory."
            )
        if not noises_path.exists() or not noises_path.is_file():
            # print("noises_path:", noises_path)
            raise ValueError(
                "Expected `--noises_column` to be path to a file in `--data_root` containing line-separated paths to noise data in the same directory."
            )
        if not masks_path.exists() or not masks_path.is_file():
            raise ValueError(
                "Expected `--masks_column` to be path to a file in `--data_root` containing line-separated paths to mask data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            video_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(noises_path, "r", encoding="utf-8") as file:
            noises_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]
        with open(masks_path, "r", encoding="utf-8") as file:
            masks_paths = [self.data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0]

        if not self.load_tensors and any(not path.is_file() for path in video_paths):
            raise ValueError(
                f"Expected `{self.video_column=}` to be a path to a file in `{self.data_root=}` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return prompts, video_paths, noises_paths, masks_paths, video_original_path, video_original_masks_path, layered_masks_paths




    def _preprocess_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            video_reader = decord.VideoReader(uri=path.as_posix())
            video_num_frames = len(video_reader)
            nearest_frame_bucket = min(
                self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
            )

            frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

            frames = video_reader.get_batch(frame_indices)
            frames = frames[:nearest_frame_bucket].float()
            frames = frames.permute(0, 3, 1, 2).contiguous()

            nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
            frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)

            
            frames = torch.stack([self.video_transforms(frame) for frame in frames_resized], dim=0)

            image = frames[:1].clone() if self.image_to_video else None

            return image, frames, None
        
    def _preprocess_mask_video(self, path: Path) -> torch.Tensor:
        if self.load_tensors:
            return self._load_preprocessed_latents_and_embeds(path)
        else:
            # Check if path is a directory (for npy files) or a file (for video)
            if path.is_dir():
                return self._preprocess_mask_from_npy(path)
            else:
                # Original video loading logic
                video_reader = decord.VideoReader(uri=path.as_posix())
                video_num_frames = len(video_reader)
                nearest_frame_bucket = min(
                    self.frame_buckets, key=lambda x: abs(x - min(video_num_frames, self.max_num_frames))
                )

                frame_indices = list(range(0, video_num_frames, video_num_frames // nearest_frame_bucket))

                frames = video_reader.get_batch(frame_indices)
                frames = frames[:nearest_frame_bucket].float()
                frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

                nearest_res = self._find_nearest_resolution(frames.shape[2], frames.shape[3])
                frames_resized = torch.stack([resize(frame, nearest_res) for frame in frames], dim=0)

                # Convert to grayscale: average over channel dimension (assume RGB)
                masks_gray = frames_resized.mean(dim=1, keepdim=True)  # [T, 1, H, W]
                # Normalize to [0, 1] if needed (input is 0-255)
                masks_gray = masks_gray / 255.0
                # Threshold at 0.5 to get binary mask
                masks_bin = (masks_gray > 0.5).to(torch.int64)  # [T, 1, H, W], int64

                return None, masks_bin, None


    def _preprocess_mask_from_npy(self, npy_dir: Path) -> torch.Tensor:
        """
        Load mask data from .npy files in a directory.
        Each .npy file is named with a 4-digit integer starting from 0001.npy.
        """
        # Get all .npy files in the directory and sort them
        npy_files = sorted([f for f in npy_dir.glob("*.npy")], 
                          key=lambda x: int(x.stem))
        
        if not npy_files:
            raise ValueError(f"No .npy files found in directory: {npy_dir}")
        
        # Load all mask files
        masks_list = []
        for npy_file in npy_files:
            mask_data = np.load(npy_file)
            masks_list.append(mask_data)
        
        # Stack all masks into a single tensor
        masks_array = np.stack(masks_list, axis=0)  # [T, H, W] or [T, C, H, W]
        masks_tensor = torch.from_numpy(masks_array).float()
        
        # Ensure the tensor has the right shape [T, 1, H, W]
        if masks_tensor.dim() == 3:
            # If shape is [T, H, W], add channel dimension
            masks_tensor = masks_tensor.unsqueeze(1)  # [T, 1, H, W]
        elif masks_tensor.dim() == 4 and masks_tensor.shape[1] > 1:
            # If shape is [T, C, H, W] with multiple channels, take mean across channels
            masks_tensor = masks_tensor.mean(dim=1, keepdim=True)  # [T, 1, H, W]
        
        # Find nearest resolution and resize if needed
        nearest_res = self._find_nearest_resolution(masks_tensor.shape[2], masks_tensor.shape[3])
        if (masks_tensor.shape[2], masks_tensor.shape[3]) != nearest_res:
            masks_resized = torch.stack([resize(mask, nearest_res) for mask in masks_tensor], dim=0)
            masks_tensor = masks_resized
        
        # Ensure binary mask (0 or 1)
        masks_bin = (masks_tensor > 0.5).to(torch.int64)  # [T, 1, H, W], int64
        
        return None, masks_bin, None



    def _find_nearest_resolution(self, height, width):
        nearest_res = min(self.resolutions, key=lambda x: abs(x[1] - height) + abs(x[2] - width))
        return nearest_res[1], nearest_res[2]


