"""Shared fixtures: a synthetic raw listing table and a quickly trained bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plotify import create_app
from plotify.bundle import load_bundle
from plotify.config import Settings
from plotify.service import PlotPriceService
from plotify.training import MODEL_FILENAME, TrainConfig, train

LOCALITY_RATES = {"Isnapur": 40_000, "Muthangi": 32_000, "Patighanpur": 45_000, "Patancheru": 30_000}


def make_raw(n: int = 90, seed: int = 7, region: str = "Patancheru") -> pd.DataFrame:
    """Raw rows in the 99acres export format, with known locality rates.

    ``region`` is the wider area. It stands in for "Patancheru" in locations,
    so the same data can pose as another area's listings.
    """
    rng = np.random.default_rng(seed)
    rows = []
    localities = list(LOCALITY_RATES)
    for i in range(n):
        locality = localities[i % len(localities)]
        area = float(rng.choice([121, 150, 167, 200, 240, 267, 300, 400, 500, 1000]))
        rate = LOCALITY_RATES[locality] * float(np.exp(rng.normal(0, 0.15)))
        price = area * rate
        price_text = f"{price / 1e7:.2f} Crore" if price >= 1e7 else f"{price / 1e5:.1f} Lac"
        location = region if locality == "Patancheru" else f"{locality}, {region}"
        if i % 5 == 0:
            location = f"Venture {i}\n{location}"
        rows.append({
            "Title": f"Residential Land/Plot for Sale in {location.splitlines()[-1]}",
            "Price": price_text,
            "Location": location,
            "Plot_Area": area,
            "facing": ["East", "West", "North-East", None][i % 4],
            "open_sides": [1, 2, "Immediate", None][i % 4],
            "Width_road": [f"{w:.2f} ft" for w in (30, 40, 60, 100)][i % 4] if i % 6 else None,
        })
    # A rare locality, a unit error and an exact duplicate, as in the real data.
    rows.append({**rows[1], "Location": "Chitkul", "Title": "Residential Land/Plot for Sale in Chitkul"})
    rows.append({**rows[2], "Plot_Area": 22.11})
    rows.append(dict(rows[3]))
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("data") / "listings.csv"
    make_raw().to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw_path) -> dict:
    root = tmp_path_factory.mktemp("trained")
    config = TrainConfig(data_path=raw_path, model_dir=root / "models", report_dir=root / "reports",
                         processed_path=root / "plots.csv", n_repeats=2, log=lambda *_: None,
                         candidates=("median_rate", "locality_median_rate", "ridge_log_rate"))
    result = train(config)
    return {"root": root, "config": config, "result": result, "model_path": root / "models" / MODEL_FILENAME}


@pytest.fixture(scope="session")
def service(trained) -> PlotPriceService:
    return PlotPriceService(load_bundle(trained["model_path"]))


@pytest.fixture()
def app(trained, service):
    settings = Settings(model_path=trained["model_path"], rate_limit_per_minute=1000)
    application = create_app(settings, service=service)
    application.config.update(TESTING=True)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()
