# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from rich import print

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (AssignAttributesProcessor, FrameAttribute,
                               MultiviewVideoList, ProcessedVideoStream,
                               StreamProcessor, VideoStream)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import (AdaptiveDepthProcessor, GeoCalibIntrinsicsProcessor,
                         GroundTruthMaskProcessor, TrackAnythingProcessor)

logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.out_path = Path(self.out_cfg.path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        
        # Check if ground truth masks are provided [IMPLEMENTED BY CURSOR]
        if hasattr(self.init_cfg, 'gt_masks') and self.init_cfg.gt_masks is not None:
            print(f"Using ground truth masks from: {self.init_cfg.gt_masks}")
            gt_masks, gt_phrases = self._load_ground_truth_masks(self.init_cfg.gt_masks, video_stream)
            init_processors.append(
                GroundTruthMaskProcessor(
                    gt_masks=gt_masks,
                    gt_phrases=gt_phrases,
                    add_sky=getattr(self.init_cfg.instance, 'add_sky', False) if self.init_cfg.instance else False,
                    mask_expand=getattr(self.init_cfg.instance, 'mask_expand', 5) if self.init_cfg.instance else 5,
                )
            )
        elif self.init_cfg.instance is not None:
            print(f"TrackAnythingProcessor is initialized with phrases: {self.init_cfg.instance.phrases}")
            init_processors.append(
                TrackAnythingProcessor(
                    mask_phrases=self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _load_ground_truth_masks(self, gt_config: DictConfig, video_stream: VideoStream) -> tuple[list[torch.Tensor], dict[int, str]]:
        """
        [IMPLEMENTED BY CURSOR]
        Load ground truth masks from various formats.
        
        Args:
            gt_config: Configuration containing mask path and format info
            video_stream: Video stream to get frame dimensions
            
        Returns:
            Tuple of (list of masks per frame, instance_id to phrase mapping)
        """
        mask_path = Path(gt_config.path)
        format_type = getattr(gt_config, 'format', 'npz')  # Default format
        
        gt_masks = []
        gt_phrases = {0: "background"}  # Always include background
        
        if format_type == 'zipped_png':
            raise NotImplementedError("Zipped PNG format is not supported for ground truth masks")
            # Load from zipped PNG files (same format as VIPE saves)
            import zipfile
            with zipfile.ZipFile(mask_path, 'r') as z:
                file_names = sorted(z.namelist())
                for file_name in file_names:
                    frame_idx = int(file_name.split('.')[0])
                    with z.open(file_name) as f:
                        mask_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                        mask = cv2.imdecode(mask_buffer, cv2.IMREAD_UNCHANGED)
                        gt_masks.append(torch.from_numpy(mask.copy()).byte())
            
            # Load phrases if available
            phrase_path = mask_path.parent / f"{mask_path.stem}_phrases.txt"
            if phrase_path.exists():
                with phrase_path.open('r') as f:
                    for line in f:
                        if ':' in line:
                            idx_str, phrase = line.strip().split(':', 1)
                            gt_phrases[int(idx_str)] = phrase.strip()
                            
        elif format_type == 'npz':
            # Load from NPZ file (V, N, H, W) format
            data = np.load(mask_path)
            if 'mask' in data:
                masks = data['mask']  # (V, N, H, W) boolean array
                V, N, H, W = masks.shape # where V is number of frames, N is number of objects
                
                # Convert to instance format: combine all objects into single mask
                for v in range(V):
                    instance_mask = np.zeros((H, W), dtype=np.uint8)
                    for n in range(N):
                        if np.any(masks[v, n]):
                            instance_mask[masks[v, n]] = n + 1  # Object IDs start from 1
                    gt_masks.append(torch.from_numpy(instance_mask).byte())
                
                # Create phrases from object indices
                for n in range(N):
                    gt_phrases[n + 1] = f"object_{n + 1}"
                    
        elif format_type == 'directory':
            raise NotImplementedError("Directory format is not supported for ground truth masks")
            # Load from directory of PNG files
            mask_files = sorted(mask_path.glob("*.png"))
            for mask_file in mask_files:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                gt_masks.append(torch.from_numpy(mask.copy()).byte())
                
        else:
            raise ValueError(f"Unsupported ground truth mask format: {format_type}")
        
        # Ensure we have masks for all frames
        expected_frames = len(video_stream)
        if len(gt_masks) < expected_frames:
            # Pad with empty masks if needed
            empty_mask = torch.zeros(video_stream.frame_size(), dtype=torch.uint8)
            gt_masks.extend([empty_mask] * (expected_frames - len(gt_masks)))
        elif len(gt_masks) > expected_frames:
            # Truncate if too many masks
            gt_masks = gt_masks[:expected_frames]
            
        return gt_masks, gt_phrases

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
