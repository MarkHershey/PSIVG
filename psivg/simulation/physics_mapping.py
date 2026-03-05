from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PhysicalParams:
    density: float  # kg/m^3
    young_modulus: float  # Pa
    coeff_restitution: float  # 0..1
    mu_static: float  # 0..~1.5
    mu_dynamic: float  # 0..~1.5

    def to_mpm(self) -> dict:
        youngs_modulus_MPa = self.young_modulus / 1e6
        return dict(
            E=youngs_modulus_MPa,
            rho=self.density,
            mu_static=self.mu_static,
            mu_dynamic=self.mu_dynamic,
            extra_rebound=0,
        )

    def to_dict(self) -> dict:
        return {
            "density": self.density,
            "young_modulus": self.young_modulus,
            "coeff_restitution": self.coeff_restitution,
            "mu_static": self.mu_static,
            "mu_dynamic": self.mu_dynamic,
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicalParams":
        return PhysicalParams(
            density=data.get("density", 0),
            young_modulus=data.get("young_modulus", 0),
            coeff_restitution=data.get("coeff_restitution", 0),
            mu_static=data.get("mu_static", 0),
            mu_dynamic=data.get("mu_dynamic", 0),
        )


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Base density by material class (approximate, bulk values)
BASE_DENSITY = {
    "metal": 7800.0,  # steel-ish
    "wood": 700.0,
    "hard_plastic": 1100.0,
    "soft_plastic_or_rubber": 900.0,
    "glass_or_ceramic": 2500.0,
    "fabric_or_textile": 300.0,
    "paper_or_cardboard": 800.0,
    "foam": 50.0,
    "stone_or_concrete": 2300.0,
    "other": 1000.0,  # fallback ~water
}

# Solidity factor: adjust effective density & stiffness
SOLIDITY_DENSITY_FACTOR = {
    "solid": 1.0,
    "mostly_solid": 0.8,
    "hollow_thin_shell": 0.3,
    "layered_or_composite": 0.9,
}

# Base Young's modulus by (material_class, hardness_level)
# All in Pascals (Pa). These are very rough orders of magnitude.
BASE_YOUNG_MODULUS = {
    "soft_plastic_or_rubber": {
        "very_soft": 1e5,  # soft foam rubber
        "soft": 5e5,
        "medium": 1e6,
        "hard": 5e6,
        "very_hard": 1e7,
    },
    "foam": {
        "very_soft": 5e4,
        "soft": 1e5,
        "medium": 5e5,
        "hard": 1e6,
        "very_hard": 5e6,
    },
    "fabric_or_textile": {
        "very_soft": 1e5,
        "soft": 5e5,
        "medium": 1e6,
        "hard": 5e6,
        "very_hard": 1e7,
    },
    "paper_or_cardboard": {
        "very_soft": 1e7,
        "soft": 5e7,
        "medium": 1e8,
        "hard": 5e8,
        "very_hard": 1e9,
    },
    "wood": {
        "very_soft": 5e8,
        "soft": 1e9,
        "medium": 5e9,
        "hard": 1e10,
        "very_hard": 2e10,
    },
    "hard_plastic": {
        "very_soft": 1e8,
        "soft": 5e8,
        "medium": 1e9,
        "hard": 2e9,
        "very_hard": 3e9,
    },
    "glass_or_ceramic": {
        "very_soft": 1e9,
        "soft": 2e9,
        "medium": 5e9,
        "hard": 5e10,
        "very_hard": 7e10,
    },
    "stone_or_concrete": {
        "very_soft": 1e9,
        "soft": 2e9,
        "medium": 3e10,
        "hard": 5e10,
        "very_hard": 7e10,
    },
    "metal": {
        "very_soft": 1e10,  # quite soft for metals
        "soft": 5e10,
        "medium": 1e11,
        "hard": 1.5e11,
        "very_hard": 2e11,
    },
    "other": {
        "very_soft": 1e6,
        "soft": 1e7,
        "medium": 1e8,
        "hard": 1e9,
        "very_hard": 1e10,
    },
}

# Thickness/size modifiers: adjust effective stiffness
SIZE_THICKNESS_MODIFIER = {
    "thin_and_small": 0.5,
    "thin_and_large": 0.3,
    "thick_and_small": 1.0,
    "thick_and_large": 1.2,
}

# Coefficient of restitution from bounce category
COEFF_RESTITUTION_FROM_BOUNCE = {
    "almost_no_bounce": 0.05,
    "low_bounce": 0.15,
    "medium_bounce": 0.3,
    "high_bounce": 0.6,
    "very_high_bounce": 0.85,
}

# Friction coefficients from friction_tendency & optional roughness
BASE_FRICTION_FROM_TENDENCY = {
    "very_slippery": (0.1, 0.05),
    "slippery": (0.2, 0.15),
    "medium": (0.4, 0.3),
    "grippy": (0.7, 0.5),
    "very_grippy": (1.0, 0.8),
}

# Roughness influence (multiplicative factor on friction)
# surface_roughness_level ∈ {1,2,3,4,5}
ROUGHNESS_FRICTION_FACTOR = {
    1: 0.8,  # very smooth
    2: 0.9,
    3: 1.0,
    4: 1.1,
    5: 1.2,  # very rough
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    if v is None:
        return default
    if not v:
        return default
    return v


def get_base_density(material_class: str) -> float:
    if material_class not in BASE_DENSITY:
        # Fallback to "other"
        return BASE_DENSITY["other"]
    return BASE_DENSITY[material_class]


def get_density(material_class: str, solidity: str) -> float:
    base_rho = get_base_density(material_class)
    factor = SOLIDITY_DENSITY_FACTOR.get(solidity, 1.0)
    return base_rho * factor


def get_base_young_modulus(material_class: str, hardness_level: str) -> float:
    mat_table = BASE_YOUNG_MODULUS.get(material_class)

    if mat_table is None:
        mat_table = BASE_YOUNG_MODULUS["other"]

    # If hardness level missing, default to "medium"
    if hardness_level not in mat_table:
        hardness_level = "medium"

    return mat_table[hardness_level]


def apply_size_thickness_to_E(E_base: float, size_thickness_hint: str) -> float:
    modifier = SIZE_THICKNESS_MODIFIER.get(size_thickness_hint, 1.0)
    return E_base * modifier


def get_coeff_restitution(bounce_category: str) -> float:
    return COEFF_RESTITUTION_FROM_BOUNCE.get(bounce_category, 0.3)


def get_friction_coeffs(
    friction_tendency: str, surface_roughness_level: Optional[int]
) -> (float, float):
    mu_s, mu_d = BASE_FRICTION_FROM_TENDENCY.get(
        friction_tendency, BASE_FRICTION_FROM_TENDENCY["medium"]
    )

    if isinstance(surface_roughness_level, int):
        factor = ROUGHNESS_FRICTION_FACTOR.get(surface_roughness_level, 1.0)
    else:
        factor = 1.0

    return mu_s * factor, mu_d * factor


# ---------------------------------------------------------------------------
# Main mapping function
# ---------------------------------------------------------------------------


def map_physical_params(semantic_desc: Dict[str, Any] = {}) -> PhysicalParams:
    """
    Map LVLM semantic attributes to numerical physical parameters.

    semantic_desc: dict with keys:
        "material_class"
        "solidity"
        "hardness_level"
        "bounce_category"
        "surface_roughness_level"
        "friction_tendency"
        "size_thickness_hint"
        "justification" (ignored here)

    Returns:
        PhysicalParams instance.
    """

    material_class = safe_get(semantic_desc, "material_class", "other")
    solidity = safe_get(semantic_desc, "solidity", "solid")
    hardness_level = safe_get(semantic_desc, "hardness_level", "very_soft")
    bounce_category = safe_get(semantic_desc, "bounce_category", "medium_bounce")
    friction_tendency = safe_get(semantic_desc, "friction_tendency", "medium")
    size_thickness_hint = safe_get(
        semantic_desc, "size_thickness_hint", "thick_and_small"
    )

    # roughness may come as int or string
    surface_roughness_level = semantic_desc.get("surface_roughness_level", 3)
    if isinstance(surface_roughness_level, str):
        try:
            surface_roughness_level = int(surface_roughness_level)
        except ValueError:
            surface_roughness_level = 3

    # 1. Density
    density = get_density(material_class, solidity)

    # 2. Young's modulus
    E_base = get_base_young_modulus(material_class, hardness_level)
    young_modulus = apply_size_thickness_to_E(E_base, size_thickness_hint)

    # 3. Coefficient of restitution
    coeff_restitution = get_coeff_restitution(bounce_category)

    # 4. Friction coefficients
    mu_static, mu_dynamic = get_friction_coeffs(
        friction_tendency, surface_roughness_level
    )

    return PhysicalParams(
        density=density,
        young_modulus=young_modulus,
        coeff_restitution=coeff_restitution,
        mu_static=mu_static,
        mu_dynamic=mu_dynamic,
    )


if __name__ == "__main__":
    # Example LVLM output
    example_semantic = {
        "material_class": "soft_plastic_or_rubber",
        "solidity": "hollow_thin_shell",
        "hardness_level": "medium",
        "bounce_category": "high_bounce",
        "surface_roughness_level": 3,
        "friction_tendency": "grippy",
        "size_thickness_hint": "thick_and_large",
        "justification": "The basketball is typically made of rubber with a slightly textured surface, is hollow inside, and is known for its high bounce. It appears large and holds its shape well.",
    }
    example_semantic = {}

    params = map_physical_params(example_semantic)
    print("Semantic input:")
    for k, v in example_semantic.items():
        print(f"  {k}: {v}")

    print("\nMapped physical parameters:")
    for k, v in asdict(params).items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
