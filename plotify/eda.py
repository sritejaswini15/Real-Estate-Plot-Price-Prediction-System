"""Exploratory analysis written to files: ``python -m plotify eda``.

This replaces the original ``EDA.py``, which read a live Google Sheet, showed
windows with ``plt.show()`` and wrote to a Colab path. Here every figure is
saved to ``reports/figures`` and the key tables go to
``reports/eda_summary.md``, so the analysis can be re-run and its output
attached to a report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cleaning import build_dataset, load_raw
from .domain import DEFAULT_REGION
from .market import market_summary

SERIES = "#257de0"
MUTED = "#8a8a8a"
INK = "#191919"
INK_2 = "#666666"
GRID = "#e7e7e7"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c2c2c2")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _rupees_k(value: float, _pos=None) -> str:
    return f"₹{value / 1000:g}k"


def run_eda(data_path: Path, out_dir: Path, region: str = DEFAULT_REGION) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    data = build_dataset(load_raw(data_path), region=region)
    df, report = data.data, data.report
    figures = Path(out_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK_2,
                         "axes.titlesize": 11, "axes.titleweight": "bold"})
    written = []

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(df["price_per_sqyd"], bins=20, color=SERIES, edgecolor="white", linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(_rupees_k))
    ax.set(title="Asking price per sq yd", xlabel="₹ per sq yd", ylabel="Listings")
    _style(ax)
    ax.grid(axis="x", visible=False)
    written.append(_save(fig, figures / "rate_distribution.png"))

    market = market_summary(df)
    rows = market["localities"][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1))
    ax.barh([f"{r['locality']} · {r['listings']}" for r in rows], [r["rate_median"] for r in rows],
            height=0.5, color=SERIES)
    ax.xaxis.set_major_formatter(FuncFormatter(_rupees_k))
    ax.set(title="Median asking price per sq yd by locality (· listings)", xlabel="₹ per sq yd")
    _style(ax)
    ax.grid(axis="y", visible=False)
    written.append(_save(fig, figures / "rate_by_locality.png"))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(df["area_sqyd"], df["price_inr"] / 1e5, s=22, color=SERIES, edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="Price against plot area (log scales)", xlabel="Plot area (sq yd)", ylabel="Asking price (₹ lakh)")
    _style(ax)
    written.append(_save(fig, figures / "price_vs_area.png"))

    known = df.dropna(subset=["road_width_ft"])
    widths = sorted(known["road_width_ft"].astype(float).unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    rng = np.random.default_rng(0)
    for i, width in enumerate(widths):
        rates = known.loc[known["road_width_ft"].astype(float) == width, "price_per_sqyd"]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(rates)), rates, s=18, color=SERIES, alpha=0.85,
                   edgecolor="white", linewidth=0.6)
        ax.plot([i - 0.25, i + 0.25], [rates.median()] * 2, color=INK, linewidth=2)
    ax.set_xticks(range(len(widths)), [f"{w:g} ft" for w in widths])
    ax.yaxis.set_major_formatter(FuncFormatter(_rupees_k))
    ax.set(title="Price per sq yd by road width (bar = median)", ylabel="₹ per sq yd")
    _style(ax)
    ax.grid(axis="x", visible=False)
    written.append(_save(fig, figures / "rate_by_road_width.png"))

    missing = report["missing_after_cleaning"]
    fig, ax = plt.subplots(figsize=(7, 2.4))
    labels = {"facing": "Facing", "open_sides": "Open sides", "road_width_ft": "Road width"}
    ax.barh([labels[k] for k in missing], [v / len(df) for v in missing.values()], height=0.5, color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(title="Share of listings missing each optional field", xlim=(0, 1))
    _style(ax)
    ax.grid(axis="y", visible=False)
    written.append(_save(fig, figures / "missing_values.png"))

    log_rate = np.log(df["price_per_sqyd"])
    numeric = pd.DataFrame({
        "log plot area": np.log(df["area_sqyd"]),
        "road width (ft)": pd.to_numeric(df["road_width_ft"], errors="coerce").astype(float),
        "open sides": pd.to_numeric(df["open_sides"], errors="coerce").astype(float),
        "in named layout": df["in_project"].astype(float),
    })
    corr = numeric.apply(lambda col: col.corr(log_rate, method="spearman"))
    lines = [
        "# Exploratory summary",
        "",
        f"Source: `{Path(data_path).name}`. {report['raw_rows']} raw rows, {report['clean_rows']} after cleaning, "
        f"{report['listing_groups']} distinct plots ({report['cross_posted_groups']} cross-posted).",
        "",
        "## Price per sq yd by locality",
        "",
        "| Locality | Listings | Median ₹/sq yd | Middle half | Median size (sq yd) |",
        "|---|---:|---:|---:|---:|",
        *[f"| {r['locality']} | {r['listings']} | {r['rate_median']:,.0f} | {r['rate_p25']:,.0f}–{r['rate_p75']:,.0f} "
          f"| {r['median_area_sqyd']:,.0f} |" for r in market["localities"]],
        "",
        "## Spearman correlation with log price per sq yd",
        "",
        "| Feature | ρ | Listings with value |",
        "|---|---:|---:|",
        *[f"| {name} | {value:+.2f} | {int(numeric[name].notna().sum())} |" for name, value in corr.items()],
        "",
        "## Dropped rows",
        "",
        *[f"- source row {d['source_row']}: {d['reason']}" for d in report["dropped_invalid"]],
        f"- {len(report['dropped_duplicate_source_rows'])} exact duplicates: source rows "
        f"{', '.join(map(str, report['dropped_duplicate_source_rows']))}",
        "",
        "## Figures",
        "",
        *[f"- `figures/{p.name}`" for p in written],
        "",
    ]
    summary = Path(out_dir) / "eda_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(written)} figures to {figures} and {summary}")
    return summary


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
