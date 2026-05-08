import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_from_config_returns_copy():
    """Mutating returned profile must not affect the _PRESETS singleton."""
    from engine.profiles import TradingProfile, _PRESETS
    config = {"strategy": {"trading_profile": "BALANCED"}}
    original_mode = _PRESETS["BALANCED"].htf_filter_mode
    p1 = TradingProfile.from_config(config)
    p1.htf_filter_mode = "strict"           # mutate the copy
    p2 = TradingProfile.from_config(config)  # get a fresh copy
    assert p2.htf_filter_mode == original_mode   # preset unchanged
    assert _PRESETS["BALANCED"].htf_filter_mode == original_mode  # original unchanged


def test_load_returns_canonical_preset():
    """load() and from_config() return different objects with equal values."""
    from engine.profiles import TradingProfile
    preset = TradingProfile.load("BALANCED")
    config = {"strategy": {"trading_profile": "BALANCED"}}
    copy = TradingProfile.from_config(config)
    assert preset is not copy
    assert preset.htf_filter_mode == copy.htf_filter_mode
