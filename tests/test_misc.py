"""Rate limiter, settings, charts and the command line."""

import json

import pytest

from plotify.__main__ import main
from plotify.charts import linear_ticks, locality_rate_bars, oof_scatter, rate_strip
from plotify.config import Settings
from plotify.security import RateLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_rate_limiter_sliding_window():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60, clock=clock)
    assert limiter.hit("a") is None and limiter.hit("a") is None
    assert limiter.hit("a") == pytest.approx(60)
    assert limiter.hit("b") is None  # keys are independent
    clock.now += 61
    assert limiter.hit("a") is None


def test_rate_limiter_bounds_memory():
    clock = FakeClock()
    limiter = RateLimiter(limit=5, window=60, clock=clock, max_keys=10)
    for i in range(50):
        limiter.hit(f"ip{i}")
    assert len(limiter._hits) <= 10


def test_rate_limiter_disabled():
    limiter = RateLimiter(limit=0)
    assert all(limiter.hit("a") is None for _ in range(100))


def test_settings_from_env(tmp_path):
    settings = Settings.from_env({"PLOTIFY_MODEL_PATH": str(tmp_path / "m.joblib"), "PLOTIFY_RATE_LIMIT": "5",
                                  "PLOTIFY_TRUSTED_PROXIES": "1", "PLOTIFY_HSTS": "true"})
    assert settings.model_path == tmp_path / "m.joblib"
    assert (settings.rate_limit_per_minute, settings.trusted_proxies, settings.hsts) == (5, 1, True)
    assert Settings.from_env({}).rate_limit_per_minute == 60
    with pytest.raises(ValueError, match="PLOTIFY_RATE_LIMIT"):
        Settings.from_env({"PLOTIFY_RATE_LIMIT": "lots"})
    with pytest.raises(ValueError, match="PLOTIFY_HSTS"):
        Settings.from_env({"PLOTIFY_HSTS": "maybe"})


def test_linear_ticks_cover_range():
    ticks = linear_ticks(0, 47_000, 4)
    assert ticks[0] == 0 and ticks[-1] >= 47_000
    assert len(ticks) <= 7


def test_charts_escape_labels_and_draw_every_mark():
    rows = [{"locality": "<b>Evil</b>", "rate_median": 40_000, "rate_p25": 30_000, "rate_p75": 50_000,
             "listings": 4},
            {"locality": "Isnapur", "rate_median": 35_000, "rate_p25": 30_000, "rate_p75": 40_000, "listings": 9}]
    svg = str(locality_rate_bars(rows))
    assert "<b>Evil" not in svg and "&lt;b&gt;Evil" in svg
    assert svg.count('class="viz-bar"') == 2
    points = [{"source_row": i, "locality": "X", "area_sqyd": 200.0, "actual": 1e6 * (i + 1),
               "predicted": 1.1e6 * (i + 1)} for i in range(5)]
    scatter = str(oof_scatter(points, {"level": 0.8, "log_low": -0.5, "log_high": 0.5}))
    assert scatter.count('class="viz-dot"') == 5
    comps = [{"locality": "X", "area_sqyd": 200.0, "price_per_sqyd": r, "price_inr": r * 200}
             for r in (30_000, 30_000, 31_000)]
    strip = str(rate_strip(32_000, 20_000, 50_000, comps, 0.8))
    assert strip.count('class="viz-dot-muted"') == 4  # 3 listings + the legend swatch


def test_cli_predict_json(trained, capsys):
    code = main(["predict", "--area", "1800", "--unit", "sqft", "--locality", "Isnapur",
                 "--model", str(trained["model_path"]), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input"]["area_sqyd"] == pytest.approx(200)
    assert payload["estimate"]["price_inr"] > 0


def test_cli_predict_text_and_errors(trained, capsys):
    assert main(["predict", "--area", "200", "--model", str(trained["model_path"])]) == 0
    assert "Estimated asking price" in capsys.readouterr().out
    assert main(["predict", "--area", "-5", "--model", str(trained["model_path"])]) == 2
    assert "area" in capsys.readouterr().err
