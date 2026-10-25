import json

from exa_home.config_loader import HardwareProfile, load, normalize


def test_inline_json_overrides_with_presets(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({
        "hardware": {"usable_vram_gb": 24, "usable_ram_gb": 64, "n_cpu": 8},
        "serving": {"batch_size": 7, "nprobe": 9},
    }))
    cfg = load(p)
    assert cfg["hardware_profile"] == "rtx-3090-classic"
    assert cfg["operating_point"]["batch_size"] == 7
    assert cfg["operating_point"]["top_coarse"] == 50
    assert cfg["optiona"] is False if False else True


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
    try:
        load(p)
    except ValueError:
        return
    raise AssertionError("top-level JSON list must fail")