from rich import print

import psivg.constants as C
from cli import command_line_entry_point
from psivg.helpers import print_section
from psivg.perception.perception_2d import main as perception_2d_main
from psivg.utils.extract_frames import extract_all_frames


def main(sample_id: str, overwrite: bool = False):
    input_video_path = C.INPUT_VIDEOS_DIR / f"{sample_id}.mp4"
    # The main function assumes that the video must exist in the input videos directory
    assert (
        input_video_path.exists()
    ), f"Sample ID '{input_video_path}' not found in {C.INPUT_VIDEOS_DIR}"
    # extract all frames from video
    print_section("Extracting frames from video")
    extract_all_frames(
        video_path=input_video_path,
        output_folder=C.INPUT_FRAMES_DIR / sample_id,
    )
    # run 2d perception pipeline on video frames
    perception_2d_main(sample_id=sample_id, overwrite=overwrite)

    print("Main Part 1 completed.")


if __name__ == "__main__":
    sample_id = command_line_entry_point()
    main(sample_id)
