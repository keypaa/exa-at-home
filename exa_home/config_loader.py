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


def _load_raw(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed")
        return yaml.safe_load(text) or {}
    return json.loads(text)


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

    out.setdefault(
        "models",
        {
            "embedder": C.EMBEDDER_MODEL,
            "reranker": C.RERANKER_MODEL,
            "dimensions": C.DIM,
        },
    )
    out.setdefault("ingest", {"wet_files": 4, "wet_target_count": None})
    out.setdefault(
        "serving",
        {
            "nprobe": C.OPERATING_POINT["nprobe"],
            "top_coarse": C.OPERATING_POINT["top_coarse"],
            "max_pair_tokens": C.OPERATING_POINT["max_pair_tokens"],
            "batch_size": C.OPERATING_POINT["batch_size"],
            "max_query_tokens": C.OPERATING_POINT["max_query_tokens"],
            "budget_ms": C.OPERATING_POINT["budget_ms"],
            "k": C.OPERATING_POINT["k"],
        },
    )
    out.setdefault("embedding", {"max_seq_length": prof.presets["max_seq_length"]})
    out.setdefault(
        "resilience",
        {
            "resume": True,
            "chunk_size": prof.presets["chunk_size"],
            "embed_batch": prof.presets["embed_batch"],
        },
    )
    user = raw.get("serving", {})
    for key in (
        "nprobe",
        "top_coarse",
        "max_pair_tokens",
        "batch_size",
        "max_query_tokens",
        "budget_ms",
        "k",
    ):
        if key in user:
            out["serving"][key] = user[key]
        elif key in ("top_coarse", "batch_size") and key not in user:
            out["serving"][key] = prof.presets[
                key if key != "batch_size" else "batch_size"
            ]
    ue = raw.get("embedding", {})
    if "max_seq_length" in ue:
        out["embedding"]["max_seq_length"] = ue["max_seq_length"]
    ur = raw.get("resilience", {})
    for key in ("resume", "chunk_size", "embed_batch"):
        if key in ur:
            out["resilience"][key] = ur[key]
    op = dict(out["serving"])
    raw_op = raw.get("operating_point")
    if raw_op is not None:
        # A key given in both places must agree; otherwise the server and
        # the pipeline would run different operating points.
        conflicts = sorted(k for k, v in raw_op.items() if k in op and op[k] != v)
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
    try:
        return int(wet)
    except (TypeError, ValueError):
        return None
