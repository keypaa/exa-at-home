import json

import pytest

from exa_home.config_loader import HardwareProfile, load, normalize


def test_inline_json_overrides_with_presets(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps(
            {
                "hardware": {"usable_vram_gb": 24, "usable_ram_gb": 64, "n_cpu": 8},
                "serving": {"batch_size": 7, "nprobe": 9},
            }
        )
    )
    cfg = load(p)
    assert cfg["hardware_profile"] == "rtx-3090-classic"
    assert cfg["operating_point"]["batch_size"] == 7
    assert cfg["operating_point"]["top_coarse"] == 50


def test_tiny_profile_values(tmp_path):
    p = tmp_path / "tiny.json"
    p.write_text("{}")
    prof = HardwareProfile(name="tiny", usable_vram_gb=8, usable_ram_gb=16, n_cpu=2)
    cfg = normalize({}, fallback=prof)
    assert cfg["hardware_profile"] == "tiny"
    assert cfg["operating_point"]["top_coarse"] == 25
    assert cfg["resilience"]["chunk_size"] == 2048


def test_invalid_config_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="object"):
        load(p)


def test_unknown_top_level_key_raises():
    """A typo'd key must fail instead of silently keeping the default."""
    with pytest.raises(ValueError, match="unknown config key.*top_coars"):
        normalize({"top_coars": 10})


def test_unknown_section_key_raises():
    with pytest.raises(ValueError, match="serving.*nprob"):
        normalize({"serving": {"nprob": 4}})
    with pytest.raises(ValueError, match="ingest.*wet_file"):
        normalize({"ingest": {"wet_file": 4}})
    with pytest.raises(ValueError, match="operating_point.*bogus"):
        normalize({"operating_point": {"bogus": 1}})


def test_invalid_hardware_key_raises(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"hardware": {"vram_gb": 24}}))
    with pytest.raises(ValueError, match="hardware.*vram_gb"):
        load(p)
    # Values with the wrong type fail too (dataclasses do not check).
    p.write_text(json.dumps({"hardware": {"name": "x", "usable_vram_gb": "lots"}}))
    with pytest.raises(ValueError, match="usable_vram_gb must be a number"):
        load(p)


def test_operating_point_and_serving_must_agree():
    cfg = normalize(
        {"serving": {"nprobe": 8}, "operating_point": {"nprobe": 8, "k": 3}}
    )
    assert cfg["operating_point"]["nprobe"] == 8
    assert cfg["operating_point"]["k"] == 3
    with pytest.raises(ValueError, match="conflicts with serving.*nprobe"):
        normalize({"serving": {"nprobe": 8}, "operating_point": {"nprobe": 9}})


def test_load_result_drives_mcp_operating_point(tmp_path):
    """The server consumes exactly the loaded config (finding P2-06)."""
    from exa_home import mcp_server
    from exa_home.config import OPERATING_POINT

    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"serving": {"k": 3, "top_coarse": 4}}))
    cfg = load(p)
    try:
        op = mcp_server.configure_operating_point(cfg["operating_point"])
        assert op["k"] == 3 and op["top_coarse"] == 4
        assert mcp_server.home_health()["operating_point"]["k"] == 3
    finally:
        mcp_server.configure_operating_point(dict(OPERATING_POINT))
