import base64
import json
import os
from typing import Any, Dict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rich import print

from .physics_mapping import PhysicalParams, map_physical_params
from psivg.constants import INPUT_META_DIR, INPUT_FRAMES_DIR


def get_physics_data(sample_id: str) -> dict[str, PhysicalParams]:
    meta_json_path = INPUT_META_DIR / f"{sample_id}.json"
    first_frame_img_path = INPUT_FRAMES_DIR / sample_id / "00000.jpg"
    meta_data = json.load(open(meta_json_path)) if meta_json_path.exists() else {}
    primary_obj_name = meta_data.get("primary", "")
    # read cached physics data
    physics_data = meta_data.get("physics_data", {})
    if primary_obj_name and primary_obj_name in physics_data:
        # construct the PhysicalParams object from the serialized data
        physics_data = {k: PhysicalParams.from_dict(v) for k, v in physics_data.items()}
        # return cached physics data
        return physics_data

    if primary_obj_name and physics_data == {}:
        assert (
            first_frame_img_path.exists()
        ), f"First frame image not found: {first_frame_img_path}"

        physics_data[primary_obj_name] = infer_physical_params(
            image_path=first_frame_img_path,
            object_name=primary_obj_name,
        )
        physics_data["default"] = map_physical_params()

        # make a copy and serialize the PhysicalParams object
        physics_data_copy = physics_data.copy()
        physics_data_copy[primary_obj_name] = physics_data[primary_obj_name].to_dict()
        physics_data_copy["default"] = physics_data["default"].to_dict()
        # cache the physics data
        meta_data["physics_data"] = physics_data_copy
        # write to file
        json.dump(meta_data, open(meta_json_path, "w"), indent=4)

        return physics_data

    return {"default": map_physical_params()}


def infer_physical_params(
    image_path: str,
    object_name: str | None = None,
    model: str = "openai/gpt-4o",
) -> PhysicalParams:
    assert Path(image_path).exists(), f"Image file not found: {image_path}"
    client = make_openrouter_client()
    if client is None:
        return map_physical_params()
    else:
        result = query_object_semantic_attributes(
            image_path, object_name, model, client
        )
        params: PhysicalParams = map_physical_params(result)
        return params


# ---------------------------------------------------------------------------
# OpenRouter client setup
# ---------------------------------------------------------------------------


def make_openrouter_client() -> OpenAI | None:
    """
    Create an OpenAI-compatible client targeting OpenRouter.
    Requires OPENROUTER_API_KEY to be set (env or .env).
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "OpenRouter API key not found. "
            "Set OPENROUTER_API_KEY in your environment or .env file."
        )
        return None

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "Physics Semantic Attribute Extractor",
        },
    )
    return client


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def encode_image_to_base64(image_path: str) -> str:
    """
    Read a local image file and return a 'data:' URL suitable for
    OpenAI/OpenRouter multimodal input.
    """
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")

    # Best guess of mime-type from extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext in [".png"]:
        mime = "image/png"
    elif ext in [".webp"]:
        mime = "image/webp"
    else:
        mime = "image/png"  # fallback

    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a physics-aware vision assistant.
You are given an image that contains an object of interest.
Your task is to describe the object's material and contact behavior using
ONLY qualitative categories and discrete labels.

IMPORTANT CONSTRAINTS:
- DO NOT output any numeric physical values (such as density, Young's modulus,
  coefficient of restitution, or friction coefficients).
- Always use ONLY the label sets and formats specified below.
- Your final output MUST be a single valid JSON object with the exact keys,
  and no extra text before or after it.
""".strip()


def build_user_prompt_text(object_name: str | None = None) -> str:
    """
    Build the text portion of the user message that defines
    the label sets and JSON format.
    """
    _tmp = (
        "The object of interest is a " + object_name + "."
        if object_name is not None
        else ""
    )

    return (
        _tmp
        + """
You are given an image that contains a single or multiple objects.
The object of interest is the one that is highlighted by a bounding box in the image.

Based on visual appearance and common real-world knowledge,
estimate the following qualitative physical attributes of the object of interest.

1. Material category (choose ONE):
   - "metal"
   - "wood"
   - "hard_plastic"
   - "soft_plastic_or_rubber"
   - "glass_or_ceramic"
   - "fabric_or_textile"
   - "paper_or_cardboard"
   - "foam"
   - "stone_or_concrete"
   - "other"

2. Solidity / internal structure (choose ONE):
   - "solid"              (mostly solid, compact interior)
   - "mostly_solid"       (small hollow parts but mostly solid)
   - "hollow_thin_shell"  (like a thin plastic ball, can)
   - "layered_or_composite" (e.g., book, stacked layers)

3. Hardness (choose ONE):
   - "very_soft"   (easily deformable by hand, like sponge, pillow)
   - "soft"
   - "medium"
   - "hard"
   - "very_hard"   (rigid, difficult to deform, like steel, stone, thick glass)

4. Elasticity / bounce behavior in everyday use:
   Imagine the object is dropped from about 1 meter onto a hard floor.
   Choose ONE:
   - "almost_no_bounce"  (stays where it lands, heavy or very soft)
   - "low_bounce"        (barely bounces, like a wood block)
   - "medium_bounce"     (bounces but quickly loses height, like some plastics)
   - "high_bounce"       (bounces well, like a rubber ball)
   - "very_high_bounce"  (extremely lively, like a superball)

5. Surface roughness (integer 1-5):
   - 1 = very smooth / polished (glass, glossy plastic)
   - 2 = mostly smooth (painted metal, smooth wood)
   - 3 = slightly textured (matte plastic, unfinished wood)
   - 4 = rough (coarse wood, textured rubber)
   - 5 = very rough (abrasive, heavily textured)

6. Surface friction tendency (choose ONE):
   - "very_slippery"
   - "slippery"
   - "medium"
   - "grippy"
   - "very_grippy"

7. Thickness / size hint (choose ONE):
   This is to help estimate stiffness at the scale of the object.
   - "thin_and_small"  (e.g., thin plastic cup, small toy)
   - "thin_and_large"  (e.g., large but thin panel)
   - "thick_and_small" (compact, chunky object)
   - "thick_and_large" (large and bulky)

8. Short natural-language justification:
   Briefly explain how the texture, appearance, and context of the object
   led you to these choices.

OUTPUT FORMAT (VERY IMPORTANT):
Return ONLY a single JSON object with the following keys:
- "material_class"
- "solidity"
- "hardness_level"
- "bounce_category"
- "surface_roughness_level"  (integer 1-5)
- "friction_tendency"
- "size_thickness_hint"
- "justification"

Example of the required structure (values are just illustrative):
{
  "material_class": "hard_plastic",
  "solidity": "hollow_thin_shell",
  "hardness_level": "hard",
  "bounce_category": "medium_bounce",
  "surface_roughness_level": 2,
  "friction_tendency": "medium",
  "size_thickness_hint": "thin_and_small",
  "justification": "The object looks like a hollow plastic toy ball with a smooth, glossy surface."
}

AGAIN: Output ONLY the JSON object, with no extra commentary.
""".strip()
    )


# ---------------------------------------------------------------------------
# Core function: query semantic attributes
# ---------------------------------------------------------------------------


def query_object_semantic_attributes(
    image_path: str,
    object_name: str | None = None,
    model: str = "openai/gpt-4o",
    client: OpenAI = None,
) -> Dict[str, Any]:
    """
    Given a local image path, call a multimodal LVLM on OpenRouter
    to obtain the semantic physical attributes for the highlighted object.
    """
    if client is None:
        client = make_openrouter_client()

    image_data_url = encode_image_to_base64(image_path)
    user_prompt_text = build_user_prompt_text(object_name)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},  # force JSON
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            },
        ],
    )

    content = response.choices[0].message.content

    try:
        semantic = json.loads(content)
    except json.JSONDecodeError as e:
        # In case the model wraps JSON in extra text despite response_format,
        # try to heuristically extract the JSON block.
        print("Warning: JSON decode failed, trying to recover:", e)
        semantic = _try_extract_json(content)

    return semantic


def _try_extract_json(text: str) -> Dict[str, Any]:
    """
    Heuristic JSON extraction if model returns extra text around JSON.
    This is a fallback; ideally response_format avoids this.
    """
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("Could not find JSON object in model output.")
    json_str = text[first : last + 1]
    return json.loads(json_str)
