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

from pathlib import Path

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging
from vipe.utils.PSIVG_export import main as export_main
from vipe.utils.viser import run_viser


# @click.command()
# @click.argument("video", type=click.Path(exists=True, path_type=Path))
# @click.option(
#     "--output",
#     "-o",
#     type=click.Path(path_type=Path),
#     help="Output directory (default: current directory)",
#     default=Path.cwd() / "vipe_results",
# )
# @click.option(
#     "--pipeline",
#     "-p",
#     default="default",
#     help="Pipeline configuration to use (default: 'default')",
# )
# @click.option(
#     "--visualize",
#     "-v",
#     is_flag=True,
#     help="Enable visualization of intermediate results",
# )
# @click.option(
#     "--phrases",
#     default="",
#     help="Phrases separated by comma",
# )
# @click.option(
#     "--gt_masks",
#     default="",
#     help="Path to the ground truth masks npz file",
# )
def infer(
    video: Path,
    output: Path,
    pipeline: str,
    visualize: bool,
    phrases: str,
    gt_masks: str,
):
    """Run inference on a video file."""

    logger = configure_logging()

    overrides = [
        f"pipeline={pipeline}",
        f"pipeline.output.path={output}",
        "pipeline.output.save_artifacts=true",
    ]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")

    ### Added
    if gt_masks:
        gt_masks = Path(gt_masks).resolve()
        assert gt_masks.exists(), f"Ground truth masks file {gt_masks} does not exist"
        gt_masks = str(gt_masks)

    if gt_masks:
        print(f"Provided ground truth masks file: {gt_masks}")
        overrides.append(f"+pipeline.init.gt_masks.path={gt_masks}")
        overrides.append("+pipeline.init.gt_masks.format=npz")
    ### Added End

    with hydra.initialize_config_dir(
        config_dir=str(get_config_path()), version_base=None
    ):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {video}...")
    if phrases:
        phrases = [x.strip() for x in phrases.split(",")]
    else:
        phrases = []
    vipe_pipeline = make_pipeline(args.pipeline, phrases=phrases)

    # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
    video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(
        desc="Reading video stream"
    )

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


# @click.command()
# @click.argument(
#     "data_path",
#     type=click.Path(exists=True, path_type=Path),
#     default=Path.cwd() / "vipe_results",
# )
# @click.option(
#     "--port",
#     "-p",
#     default=20540,
#     type=int,
#     help="Port for the visualization server (default: 20540)",
# )
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# @click.command()
# @click.argument(
#     "data_path",
#     type=click.Path(exists=True, path_type=Path),
#     default=Path.cwd() / "vipe_results",
# )
# @click.option(
#     "--export_base_path",
#     type=click.Path(path_type=Path),
#     default=Path.cwd() / "vipe_results" / "mark_exported",
#     help="Export base path (default: VIPE_EXPORT_DIR)",
# )
# @click.option(
#     "--sample_id",
#     default="2_basketball",
#     type=str,
#     help="Sample ID to export (default: 2_basketball)",
# )
# @click.option(
#     "--launch_viser",
#     "-v",
#     is_flag=True,
#     help="Launch viser server for visualization",
# )
# @click.option(
#     "--port",
#     "-p",
#     default=20541,
#     type=int,
#     help="Port for the visualization server (default: 20541)",
# )
def export(
    data_path: Path,
    export_base_path: Path,
    sample_id: str,
    launch_viser: bool,
    port: int,
):
    export_main(
        base_path=data_path,
        export_base_path=export_base_path,
        sample_id=sample_id,
        launch_viser=launch_viser,
        port=port,
    )


# Add subcommands
# main.add_command(infer)
# main.add_command(visualize)
# main.add_command(export)

if __name__ == "__main__":
    main()
