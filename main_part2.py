import json
import time
from datetime import datetime
from pathlib import Path

from rich import print

import psivg.constants as C
from cli import command_line_entry_point
from psivg.perception.scale_refinement import run_object_scale_refinement
from psivg.perception.vipe.vipe.cli.main import infer
from psivg.perception.vipe.vipe.utils.PSIVG_export import main as vipe_export
from psivg.rendering.render_final import render_RGB_video
from psivg.rendering.render_flow import calculate_flow
from psivg.simulation.simulation import run_mpm_simulation


def is_infer_completed(video_path: str | Path) -> bool:
    sample_id = Path(video_path).stem
    vipe_depth_file = C.VIPE_RAW_DIR / "depth" / f"{sample_id}.zip"
    if vipe_depth_file.exists():
        return True
    return False


def run_vipe_infer(sample_id: str, overwrite: bool = False, gpu_idx: int = 0):
    video_path = C.INPUT_VIDEOS_DIR / f"{sample_id}.mp4"
    assert (
        video_path.exists()
    ), f"[red]✗[/red] Sample ID '{sample_id}' not found in {C.INPUT_VIDEOS_DIR}"
    print(f"\n[green]➜ Running ViPE inference[/green] {sample_id}")

    if is_infer_completed(video_path) and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {sample_id} because output video already exists"
        )
        return

    # get phrases from obj info file
    obj_info_file = C.INPUT_META_DIR / f"{sample_id}.json"
    if obj_info_file.exists():
        with open(obj_info_file, "r") as f:
            obj_info = json.load(f)
        primary = obj_info.get("primary")
        secondary = obj_info.get("secondary")
        if primary and secondary:
            phrases = f"{primary},{secondary}"
        elif primary:
            phrases = primary
        else:
            print(
                f"  [red]Warning:[red] No primary or secondary object info found for {sample_id}"
            )
            phrases = ""
    else:
        print(f"  [red]Warning:[red] No object info file found for {sample_id}")
        phrases = ""

    ### Run vipe infer
    start_time = time.time()
    infer(
        video=video_path,
        output=C.VIPE_RAW_DIR,
        phrases=phrases,
        pipeline="default",
        visualize=False,
        gt_masks="",
    )
    time_taken = time.time() - start_time
    print(f"  [green]✔[/green] Processed in {int(time_taken)} seconds")


def run_vipe_export(sample_id: str, overwrite: bool = False, gpu_idx: int = 0):
    ### Run vipe export (Custom PSIVG export)
    print(f"\n[green]Exporting[/green] {sample_id}")
    success_file = C.VIPE_EXPORT_DIR / sample_id / "success.txt"
    if success_file.exists() and not overwrite:
        print(
            f"  [yellow]✔ Skipped[/yellow] {sample_id} because output export already exists"
        )
        return

    start_time = time.time()
    vipe_export(sample_id=sample_id)
    time_taken = time.time() - start_time
    print(f"  [green]✔[/green] Exported in {int(time_taken)} seconds")


def main(sample_id: str, overwrite: bool = False, gpu_idx: int = 0):

    ### Run 4D reconstruction inference
    run_vipe_infer(sample_id=sample_id, overwrite=overwrite, gpu_idx=gpu_idx)

    ### Post-process 4D reconstruction results
    run_vipe_export(sample_id=sample_id, overwrite=overwrite, gpu_idx=gpu_idx)

    ### Refine object mesh size
    print(f"\n[green]Running object scale refinement[/green] {sample_id}")
    run_object_scale_refinement(sample_id, overwrite=overwrite)

    ### Run simulation
    print(f"\n[green]Running physical simulation[/green] {sample_id}")
    date_today = datetime.now().strftime("%Y-%m-%d")
    simulation_id = f"{date_today}_run"
    run_mpm_simulation(
        device="cpu",
        sample_id=sample_id,
        output_dir=C.OUT_SIMULATION_DIR / sample_id / simulation_id,
        fps_factor=4,
        overwrite=overwrite,
    )

    ### Render RGB video
    print(f"\n[green]Rendering RGB video[/green] {sample_id}")
    render_RGB_video(
        sample_id=sample_id, simulation_id=simulation_id, overwrite=overwrite
    )

    ### Calculate frame-to-frame flow & point correspondences
    print(f"\n[green]Calculating flow[/green] {sample_id}")
    calculate_flow(
        sample_id=sample_id, simulation_id=simulation_id, overwrite=overwrite
    )


if __name__ == "__main__":
    sample_id = command_line_entry_point()
    main(sample_id=sample_id)
