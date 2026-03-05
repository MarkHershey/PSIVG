from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
PRETRAINED_MODELS_DIR = PROJ_ROOT / "pretrained_models"
CONFIGS_DIR = PROJ_ROOT / "configs"

###############################################################################
### Change this to your own data root
DATA_ROOT = PROJ_ROOT / "data_root"
### Do not edit below this line
###############################################################################


def _create_sub_dir(base_dir: Path, sub_dir_name: str) -> Path:
    base_dir = Path(base_dir).resolve()
    sub_dir = base_dir / sub_dir_name
    sub_dir.mkdir(parents=True, exist_ok=True)
    return sub_dir


INPUT_DATA_DIR = _create_sub_dir(DATA_ROOT, "INPUT_DATA")
INPUT_VIDEOS_DIR = _create_sub_dir(INPUT_DATA_DIR, "Videos")
INPUT_FRAMES_DIR = _create_sub_dir(INPUT_DATA_DIR, "Frames")
INPUT_PROMPTS_DIR = _create_sub_dir(INPUT_DATA_DIR, "Prompts")
INPUT_MASKS_DIR = _create_sub_dir(INPUT_DATA_DIR, "Masks")
INPUT_META_DIR = _create_sub_dir(INPUT_DATA_DIR, "Metadata")

VIPE_RAW_DIR = _create_sub_dir(DATA_ROOT, "OUT_ViPE_Raw")
VIPE_EXPORT_DIR = _create_sub_dir(DATA_ROOT, "OUT_ViPE_Export")

OUT_PERCEPTION_DIR = _create_sub_dir(DATA_ROOT, "OUT_Perception")
OUT_SIMULATION_DIR = _create_sub_dir(DATA_ROOT, "OUT_Simulation")
OUT_RENDERING_DIR = _create_sub_dir(DATA_ROOT, "OUT_Rendering")
