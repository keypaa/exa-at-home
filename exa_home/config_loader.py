"""Unified configuration loader + hardware presets (Lane B)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class HardwareProfile:
    name: str = "rtx-pro-6000-classic"
    usable_vram_gb: float = 96.0
    usable_ram_gb: float = 160.0
    n_cpu: int = 4

    @property
    def tier(self) -> str:
        if self.usable_vram_gb <= 14:
            return "tiny"
        if self.usable_vram_gb <= 26:
            return "rtx-3090-classic"
        return "rtx-pro-6000-classic"

    @property
    def presets(self) -> dict[str, int]:
        if self.usable_vram_gb <= 14:
            return {
                "batch_size": 16,
                "embed_batch": 64,
                "max_seq_length": 128,
                "chunk_size": 2048,
                "top_coarse": 25,
            }
        if self.usable_vram_gb <= 26:
            return {
                "batch_size": 32,
                "embed_batch": 256,
                "max_seq_length": 256,
                "chunk_size": 4096,
                "top_coarse": 50,
            }
        return {
            "batch_size": 32,
            "embed_batch": 1024,
            "max_seq_length": 256,
            "chunk_size": 8192,
            "top_coarse": 50,
        }


# Strict schema: a typo'd key used to be copied through silently and the
# knob it was meant to set kept its default (finding P2-06). Unknown keys
# now fail loudly at load time instead of shipping a differently-configured
# server than the operator wrote down.
_TOP_LEVEL_KEYS = {
    "hardware",
    "models",
    "ingest",
    "serving",
    "embedding",
    "resilience",
    "operating_point",
    "hardware_profile",
}
_SECTION_KEYS = {
    "hardware": {"name", "usable_vram_gb", "usable_ram_gb", "n_cpu"},
    "models": {"embedder", "reranker", "dimensions"},
    "ingest": {"wet_files", "wet_target_count"},
    "serving": {
        "nprobe",
        "top_coarse",
        "max_pair_tokens",
        "batch_size",
        "max_query_tokens",
        "budget_ms",
        "k",
    },
    "embedding": {"max_seq_length"},
    "resilience": {"resume", "chunk_size", "embed_batch"},
}


def _reject_unknown(where: str, got, allowed: set[str]) -> None:
    if got is None:
        return
    if not isinstance(got, dict):
        raise ValueError(f"{where} must be an object")  # noqa: TRY004
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(
            f"unknown config key(s) in {where}: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(allowed))})"
        )


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(where: str, key: str, value, minimum: int = 1) -> None:
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{where}.{key} must be an integer >= {minimum}")


def _require_number(where: str, key: str, value, minimum: float = 0.0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= minimum
    ):
        raise ValueError(f"{where}.{key} must be a number > {minimum:g}")


def _require_str(where: str, key: str, value) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}.{key} must be a non-empty string")


def _require_bool(where: str, key: str, value) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key} must be a boolean")  # noqa: TRY004


_SERVING_INT_KEYS = (
    "nprobe",
    "top_coarse",
    "max_pair_tokens",
    "batch_size",
    "max_query_tokens",
    "k",
)


def _validate_serving(where: str, section) -> None:
    if not isinstance(section, dict):
        return
    for key in _SERVING_INT_KEYS:
        if key in section:
            _require_int(where, key, section[key])
    if "budget_ms" in section:
        _require_number(where, "budget_ms", section["budget_ms"])


def _validate(raw: dict) -> None:
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"unknown config key(s): {', '.join(unknown)} "
            f"(known: {', '.join(sorted(_TOP_LEVEL_KEYS))})"
        )
    for section, allowed in _SECTION_KEYS.items():
        _reject_unknown(section, raw.get(section), allowed)
    # `operating_point` is the serving override dict consumed by the MCP
    # server; it must use the same keys as `serving`.
    _reject_unknown(
        "operating_point", raw.get("operating_point"), _SECTION_KEYS["serving"]
    )
    # Type/range checks for every known key. Unknown keys already fail;
    # wrong TYPES used to sail through and surface later as an opaque
    # "internal error" from the server (a config typo looked like a bug).
    _validate_serving("serving", raw.get("serving"))
    _validate_serving("operating_point", raw.get("operating_point"))
    models = raw.get("models")
    if isinstance(models, dict):
        for key in ("embedder", "reranker"):
            if key in models:
                _require_str("models", key, models[key])
        if "dimensions" in models:
            _require_int("models", "dimensions", models["dimensions"])
    ingest = raw.get("ingest")
    if isinstance(ingest, dict):
        if "wet_files" in ingest:
            _require_int("ingest", "wet_files", ingest["wet_files"])
        if ingest.get("wet_target_count") is not None:
            _require_int("ingest", "wet_target_count", ingest["wet_target_count"])
    embedding = raw.get("embedding")
    if isinstance(embedding, dict) and "max_seq_length" in embedding:
        _require_int("embedding", "max_seq_length", embedding["max_seq_length"])
    resilience = raw.get("resilience")
    if isinstance(resilience, dict):
        if "resume" in resilience:
            _require_bool("resilience", "resume", resilience["resume"])
        for key in ("chunk_size", "embed_batch"):
            if key in resilience:
                _require_int("resilience", key, resilience[key])
    hp = raw.get("hardware_profile")
    if hp is not None and not isinstance(hp, str):
        raise ValueError("hardware_profile must be a string")
    hw = raw.get("hardware")
    if isinstance(hw, dict):
        # dataclasses do not type-check at construction; without this a
        # string vram value only explodes later inside HardwareProfile.tier.
        if "name" in hw and not isinstance(hw["name"], str):
            raise ValueError("hardware.name must be a string")
        for key in ("usable_vram_gb", "usable_ram_gb"):
            if key in hw and (
                isinstance(hw[key], bool) or not isinstance(hw[key], (int, float))
            ):
                raise ValueError(f"hardware.{key} must be a number")
        if "n_cpu" in hw and (
            isinstance(hw["n_cpu"], bool) or not isinstance(hw["n_cpu"], int)
        ):
            raise ValueError("hardware.n_cpu must be an integer")


def normalize(raw, fallback=None):
    from . import config as C

    if fallback is None:
        fallback = HardwareProfile()
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")  # noqa: TRY004
    _validate(raw)
    out = {"hardware": asdict(fallback)}
    out.update({k: v for k, v in raw.items() if k != "hardware"})
    if "hardware" in raw and isinstance(raw["hardware"], dict):
        hw = dict(asdict(fallback))
        hw.update(raw["hardware"])
        out["hardware"] = hw
    try:
        prof = HardwareProfile(**out["hardware"])
    except TypeError as e:
        # Previously swallowed into the default profile: the operator's
        # hardware block silently did nothing (finding P2-06).
        raise ValueError(f"invalid hardware config: {e}") from e

    def merged(defaults: dict, section: str) -> dict:
        """Defaults + the user's keys, as a fresh dict (never alias input)."""
        got = raw.get(section)
        return {**defaults, **(dict(got) if isinstance(got, dict) else {})}

    out["models"] = merged(
        {
            "embedder": C.EMBEDDER_MODEL,
            "reranker": C.RERANKER_MODEL,
            "dimensions": C.DIM,
        },
        "models",
    )
    out["ingest"] = merged({"wet_files": 4, "wet_target_count": None}, "ingest")
    # A `serving` block is always the COMPLETE seven-key operating point,
    # not only the keys the user wrote: a partial block used to rely on
    # configure_operating_point to merge defaults, the opposite of the
    # "one config" promise.
    serving_defaults = {
        "nprobe": C.OPERATING_POINT["nprobe"],
        "top_coarse": prof.presets["top_coarse"],
        "max_pair_tokens": C.OPERATING_POINT["max_pair_tokens"],
        "batch_size": prof.presets["batch_size"],
        "max_query_tokens": C.OPERATING_POINT["max_query_tokens"],
        "budget_ms": C.OPERATING_POINT["budget_ms"],
        "k": C.OPERATING_POINT["k"],
    }
    user_serving = raw.get("serving")
    user_serving = dict(user_serving) if isinstance(user_serving, dict) else {}
    out["serving"] = {**serving_defaults, **user_serving}
    out["embedding"] = merged(
        {"max_seq_length": prof.presets["max_seq_length"]}, "embedding"
    )
    out["resilience"] = merged(
        {
            "resume": True,
            "chunk_size": prof.presets["chunk_size"],
            "embed_batch": prof.presets["embed_batch"],
        },
        "resilience",
    )
    op = dict(out["serving"])
    raw_op = raw.get("operating_point")
    if raw_op is not None:
        # A key the operator set in BOTH places must agree; otherwise the
        # server and the pipeline would run different operating points.
        # Keys supplied only in operating_point are overrides of defaults,
        # not conflicts (the old check compared them to preset values and
        # rejected a perfectly valid operating_point-only config).
        conflicts = sorted(
            k for k, v in raw_op.items() if k in user_serving and user_serving[k] != v
        )
        if conflicts:
            raise ValueError(
                "operating_point conflicts with serving for: " + ", ".join(conflicts)
            )
        op.update(raw_op)
    out["operating_point"] = op
    out.setdefault(
        "hardware_profile", prof.tier if prof.name not in ("tiny",) else prof.name
    )
    return out


def load(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML not installed; use JSON configs") from e
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("config must be JSON/YAML object")  # noqa: TRY004
    fallback = HardwareProfile()
    if isinstance(raw.get("hardware"), dict) and raw["hardware"].get("name"):
        # No silent fallback: a bad hardware block used to be discarded
        # while the rest of the config was accepted (finding P2-06).
        try:
            fallback = HardwareProfile(**raw["hardware"])
        except TypeError as e:
            raise ValueError(f"invalid hardware config: {e}") from e
    return normalize(raw, fallback=fallback)


def hardware_profile_for(cfg):
    if isinstance(cfg.get("hardware"), dict):
        return HardwareProfile(**cfg["hardware"])
    return HardwareProfile()


def cli_target_count(cfg: dict, default_wet_files: int = 4) -> int | None:
    wet = cfg.get("ingest", {}).get("wet_target_count")
    if wet is None:
        return None
    if not _is_int(wet):
        raise ValueError("ingest.wet_target_count must be an integer")
    return wet
