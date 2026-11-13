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


def test_operating_point_only_config_is_not_a_false_conflict():
    """A valid operating_point-only override must not compare against presets."""
    cfg = normalize({"operating_point": {"nprobe": 16, "k": 5}})
    assert cfg["operating_point"]["nprobe"] == 16
    assert cfg["operating_point"]["k"] == 5


def test_partial_serving_block_is_filled_to_all_keys():
    cfg = normalize({"serving": {"nprobe": 9}})
    assert set(cfg["serving"]) == {
        "nprobe",
        "top_coarse",
        "max_pair_tokens",
        "batch_size",
        "max_query_tokens",
        "budget_ms",
        "k",
    }
    assert cfg["operating_point"] == cfg["serving"]


def test_wrong_known_key_types_fail_loudly():
    with pytest.raises(ValueError, match="serving.k must be an integer"):
        normalize({"serving": {"k": "3"}})
    with pytest.raises(ValueError, match="serving.nprobe must be an integer"):
        normalize({"serving": {"nprobe": None}})
    with pytest.raises(ValueError, match="ingest.wet_files must be an integer"):
        normalize({"ingest": {"wet_files": "abc"}})
    with pytest.raises(ValueError, match="resilience.resume must be a boolean"):
        normalize({"resilience": {"resume": "yes"}})
    with pytest.raises(ValueError, match="operating_point.budget_ms"):
        normalize({"operating_point": {"budget_ms": -1}})


def test_normalize_does_not_mutate_the_callers_serving_dict():
    raw = {"serving": {"nprobe": 9}}
    normalize(raw)
    assert raw == {"serving": {"nprobe": 9}}


def test_cli_target_count_rejects_non_int():
    from exa_home.config_loader import cli_target_count

    assert cli_target_count({"ingest": {"wet_target_count": 3}}) == 3
    assert cli_target_count({"ingest": {"wet_target_count": None}}) is None
    with pytest.raises(ValueError, match="must be an integer"):
        cli_target_count({"ingest": {"wet_target_count": "3"}})
