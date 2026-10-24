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
            return {"batch_size": 16, "embed_batch": 64, "max_seq_length": 128,
                    "chunk_size": 2048, "top_coarse": 25}
        if self.usable_vram_gb <= 26:
            return {"batch_size": 32, "embed_batch": 256, "max_seq_length": 256,
                    "chunk_size": 4096, "top_coarse": 50}
        return {"batch_size": 32, "embed_batch": 1024, "max_seq_length": 256,
                "chunk_size": 8192, "top_coarse": 50}


def _load_raw(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def normalize(raw, fallback=None):
    from . import config as C

    if fallback is None:
        fallback = HardwareProfile()
    if not isinstance(raw, dict):
        raise ValueError("config must be an object")  # noqa: TRY004
    out = {"hardware": asdict(fallback)}
    out.update({k: v for k, v in raw.items() if k != "hardware"})
    if "hardware" in raw and isinstance(raw["hardware"], dict):
        hw = dict(asdict(fallback))
        hw.update(raw["hardware"])
        out["hardware"] = hw
    prof = HardwareProfile(**out["hardware"])

    out.setdefault("models", {"embedder": C.EMBEDDER_MODEL,
                                 "reranker": C.RERANKER_MODEL,
                                 "dimensions": C.DIM})
    out.setdefault("ingest", {"wet_files": 4, "wet_target_count": None})
    out.setdefault("serving", {
        "nprobe": C.OPERATING_POINT["nprobe"],
        "top_coarse": C.OPERATING_POINT["top_coarse"],
        "max_pair_tokens": C.OPERATING_POINT["max_pair_tokens"],
        "batch_size": C.OPERATING_POINT["batch_size"],
        "max_query_tokens": C.OPERATING_POINT["max_query_tokens"],
        "budget_ms": C.OPERATING_POINT["budget_ms"],
        "k": C.OPERATING_POINT["k"],
    })
    out.setdefault("embedding", {"max_seq_length": prof.presets["max_seq_length"]})
    out.setdefault("resilience", {"resume": True,
                                 "chunk_size": prof.presets["chunk_size"],
                                 "embed_batch": prof.presets["embed_batch"]})
    user = raw.get("serving", {})
    for key in ("nprobe", "top_coarse", "max_pair_tokens", "batch_size",
                "max_query_tokens", "budget_ms", "k"):
        if key in user:
            out["serving"][key] = user[key]
        elif key in ("top_coarse", "batch_size") and key not in user:
            out["serving"][key] = prof.presets[key if key != "batch_size" else "batch_size"]
    ue = raw.get("embedding", {})
    if "max_seq_length" in ue:
        out["embedding"]["max_seq_length"] = ue["max_seq_length"]
    ur = raw.get("resilience", {})
    for key in ("resume", "chunk_size", "embed_batch"):
        if key in ur:
            out["resilience"][key] = ur[key]
    op = dict(out["serving"])
    out.setdefault("operating_point", op)
    out.setdefault("hardware_profile", prof.tier if prof.name not in ("tiny",) else prof.name)
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
        try:
            fallback = HardwareProfile(**raw["hardware"])
        except TypeError:
            fallback = HardwareProfile()
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