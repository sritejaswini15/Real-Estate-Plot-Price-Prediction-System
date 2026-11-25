"""Server-rendered SVG charts.

Charts are plain inline SVG, styled by CSS classes (``viz-*`` in app.css), so
they work under the strict Content-Security-Policy with no JavaScript. Every
mark has a native ``<title>`` tooltip. Each chart on the site has a table
beside it with the same numbers, so no value is reachable by hover alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from markupsafe import Markup, escape

from .formatting import format_inr, format_inr_short, format_number

WIDTH = 640


def _nice_step(span: float, target_ticks: int = 5) -> float:
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for factor in (1, 2, 2.5, 5, 10):
        if factor * magnitude >= raw:
            return factor * magnitude
    return 10 * magnitude


def linear_ticks(low: float, high: float, target_ticks: int = 5) -> list[float]:
    step = _nice_step(high - low, target_ticks)
    start = math.floor(low / step) * step
    ticks, value = [], start
    while value < high + step * 0.999:
        ticks.append(round(value, 6))
        value += step
    return ticks


def _rate_tick(value: float) -> str:
    return "₹0" if value == 0 else f"₹{value / 1000:g}k"


def _svg_open(height: float, chart_id: str, title: str, desc: str) -> str:
    return (f'<svg class="viz" viewBox="0 0 {WIDTH} {height:.0f}" role="img" '
            f'aria-labelledby="{chart_id}-t {chart_id}-d" preserveAspectRatio="xMinYMin meet">'
            f'<title id="{chart_id}-t">{escape(title)}</title><desc id="{chart_id}-d">{escape(desc)}</desc>')


def _bar_path(x0: float, y: float, width: float, height: float, radius: float = 4.0) -> str:
    """A bar square at the baseline with a rounded data end."""
    r = min(radius, width, height / 2)
    x1 = x0 + width
    return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} A{r},{r} 0 0 1 {x1:.1f},{y + r:.1f} "
            f"V{y + height - r:.1f} A{r},{r} 0 0 1 {x1 - r:.1f},{y + height:.1f} H{x0:.1f} Z")


def locality_rate_bars(rows: Sequence[dict[str, Any]], chart_id: str = "rates") -> Markup:
    """Ranked horizontal bars: median asking ₹/sq yd per locality."""
    label_w, right_pad, top, row_h, bar_h, axis_h = 190, 92, 6, 34, 18, 30
    plot_w = WIDTH - label_w - right_pad
    height = top + len(rows) * row_h + axis_h
    ticks = linear_ticks(0, max(r["rate_median"] for r in rows) if rows else 1, 4)
    scale = plot_w / ticks[-1]
    plot_bottom = top + len(rows) * row_h
    parts = [_svg_open(height, chart_id, "Median asking price per sq yd by locality",
                       "Horizontal bar chart. One bar per locality, longest first; the table below lists "
                       "the same values with listing counts and quartiles.")]
    for tick in ticks:
        x = label_w + tick * scale
        parts.append(f'<line class="viz-grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{plot_bottom}"/>')
        parts.append(f'<text class="viz-tick" x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle">'
                     f'{_rate_tick(tick)}</text>')
    for i, row in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        width = max(row["rate_median"] * scale, 1.0)
        name = escape(row["locality"])
        tip = (f"{row['locality']}: median {format_inr(row['rate_median'])} per sq yd "
               f"({row['listings']} listings, middle half {format_inr(row['rate_p25'])}–"
               f"{format_inr(row['rate_p75'])})")
        parts.append(f'<g class="viz-mark"><title>{escape(tip)}</title>'
                     f'<rect class="viz-hit" x="0" y="{top + i * row_h}" width="{WIDTH}" height="{row_h}"/>'
                     f'<path class="viz-bar" d="{_bar_path(label_w, y, width, bar_h)}"/></g>')
        parts.append(f'<text class="viz-label" x="{label_w - 10}" y="{y + bar_h / 2 + 4:.1f}" '
                     f'text-anchor="end">{name} <tspan class="viz-sub">· {row["listings"]}</tspan></text>')
        parts.append(f'<text class="viz-value" x="{label_w + width + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}">'
                     f'{escape(format_inr(row["rate_median"]))}</text>')
    parts.append(f'<line class="viz-axis" x1="{label_w}" x2="{label_w}" y1="{top}" y2="{plot_bottom}"/>')
    parts.append("</svg>")
    return Markup("".join(parts))


_LOG_TICK_BASES = (1, 3)


def _log_ticks(low: float, high: float) -> list[float]:
    ticks = []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for base in _LOG_TICK_BASES:
            value = base * 10 ** exponent
            if low <= value <= high:
                ticks.append(float(value))
    return ticks


def oof_scatter(points: Sequence[dict[str, Any]], interval: dict[str, float], chart_id: str = "oof") -> Markup:
    """Asking price vs out-of-fold estimate on log axes, with the 80% band."""
    height, left, right, top, bottom = 430, 78, 16, 12, 52
    plot_w, plot_h = WIDTH - left - right, height - top - bottom
    values = [p["actual"] for p in points] + [p["predicted"] for p in points]
    low, high = min(values) / 1.25, max(values) * 1.25
    lo_log, hi_log = math.log10(low), math.log10(high)

    def sx(v: float) -> float:
        return left + (math.log10(v) - lo_log) / (hi_log - lo_log) * plot_w

    def sy(v: float) -> float:
        return top + plot_h - (math.log10(v) - lo_log) / (hi_log - lo_log) * plot_h

    level = round(interval["level"] * 100)
    parts = [_svg_open(height, chart_id, f"Asking price against out-of-fold estimate, with the {level}% range",
                       "Scatter plot on logarithmic axes. Each dot is a listing; the diagonal marks a perfect "
                       f"estimate and the shaded band is the {level}% range. The table below lists every point.")]
    parts.append(f'<clipPath id="{chart_id}-clip"><rect x="{left}" y="{top}" width="{plot_w}" '
                 f'height="{plot_h}"/></clipPath>')
    for tick in _log_ticks(low, high):
        x, y = sx(tick), sy(tick)
        label = escape(format_inr_short(tick))
        parts.append(f'<line class="viz-grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<line class="viz-grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="viz-tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{label}</text>')
        parts.append(f'<text class="viz-tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{label}</text>')
    # Band: estimate × exp(log_low) .. estimate × exp(log_high), i.e. actual within the range.
    a, b = low, high
    band = [(sx(a), sy(a * math.exp(-interval["log_low"]))), (sx(b), sy(b * math.exp(-interval["log_low"]))),
            (sx(b), sy(b * math.exp(-interval["log_high"]))), (sx(a), sy(a * math.exp(-interval["log_high"])))]
    parts.append(f'<g clip-path="url(#{chart_id}-clip)"><polygon class="viz-band" points="'
                 + " ".join(f"{x:.1f},{y:.1f}" for x, y in band) + '"/>'
                 f'<line class="viz-ref" x1="{sx(a):.1f}" y1="{sy(a):.1f}" x2="{sx(b):.1f}" y2="{sy(b):.1f}"/></g>')
    parts.append(f'<line class="viz-axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="viz-axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    for p in points:
        x, y = sx(p["actual"]), sy(p["predicted"])
        error = p["predicted"] / p["actual"] - 1
        tip = (f"Row {p['source_row']} · {p['locality']} · {format_number(p['area_sqyd'])} sq yd — asking "
               f"{format_inr_short(p['actual'])}, estimated {format_inr_short(p['predicted'])} ({error:+.0%})")
        parts.append(f'<g class="viz-mark"><title>{escape(tip)}</title>'
                     f'<circle class="viz-hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
                     f'<circle class="viz-dot" cx="{x:.1f}" cy="{y:.1f}" r="4"/></g>')
    parts.append(f'<text class="viz-axis-title" x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">'
                 'Asking price (log scale)</text>')
    parts.append(f'<text class="viz-axis-title" transform="translate(14,{top + plot_h / 2:.1f}) rotate(-90)" '
                 'text-anchor="middle">Out-of-fold estimate (log scale)</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def rate_strip(estimate: float, low: float, high: float, comparables: Sequence[dict[str, Any]],
               level: float, chart_id: str = "strip") -> Markup:
    """Where the estimate's ₹/sq yd and range sit among comparable listings."""
    height, left, right, mid, axis_y = 150, 16, 16, 58, 108
    plot_w = WIDTH - left - right
    rates = [c["price_per_sqyd"] for c in comparables]
    lo_v, hi_v = min([low, *rates]) * 0.9, max([high, *rates]) * 1.05
    ticks = linear_ticks(lo_v, hi_v, 5)
    lo_v, hi_v = min(lo_v, ticks[0]), max(hi_v, ticks[-1])

    def sx(v: float) -> float:
        return left + (v - lo_v) / (hi_v - lo_v) * plot_w

    pct = round(level * 100)
    parts = [_svg_open(height, chart_id, "Estimated price per sq yd compared with similar listings",
                       f"Strip chart. The large dot is the estimate with its {pct}% range shaded; small dots "
                       "are comparable listings. The table below lists them.")]
    for tick in ticks:
        x = sx(tick)
        parts.append(f'<line class="viz-grid" x1="{x:.1f}" x2="{x:.1f}" y1="18" y2="{axis_y - 10}"/>')
        parts.append(f'<text class="viz-tick" x="{x:.1f}" y="{axis_y + 6}" text-anchor="middle">{_rate_tick(tick)}</text>')
    parts.append(f'<g class="viz-mark"><title>{escape(f"{pct}% range: {format_inr(low)}–{format_inr(high)} per sq yd")}</title>'
                 f'<rect class="viz-range" x="{sx(low):.1f}" y="{mid - 9}" width="{max(sx(high) - sx(low), 2):.1f}" '
                 f'height="18" rx="9"/></g>')
    lanes: list[float] = []
    for c in sorted(comparables, key=lambda c: c["price_per_sqyd"]):
        x = sx(c["price_per_sqyd"])
        lane = next((i for i, last in enumerate(lanes) if x - last >= 12), len(lanes))
        if lane == len(lanes):
            lanes.append(x)
        else:
            lanes[lane] = x
        y = mid + 22 + (lane % 3) * 10
        tip = (f"{c['locality']} · {format_number(c['area_sqyd'])} sq yd — {format_inr(c['price_per_sqyd'])} "
               f"per sq yd ({format_inr_short(c['price_inr'])})")
        parts.append(f'<g class="viz-mark"><title>{escape(tip)}</title>'
                     f'<circle class="viz-hit" cx="{x:.1f}" cy="{y}" r="12"/>'
                     f'<circle class="viz-dot-muted" cx="{x:.1f}" cy="{y}" r="4.5"/></g>')
    ex = sx(estimate)
    parts.append(f'<g class="viz-mark"><title>{escape(f"Estimate: {format_inr(estimate)} per sq yd")}</title>'
                 f'<circle class="viz-hit" cx="{ex:.1f}" cy="{mid}" r="14"/>'
                 f'<circle class="viz-dot viz-dot-lg" cx="{ex:.1f}" cy="{mid}" r="7"/></g>')
    anchor = "start" if ex < left + 90 else "end" if ex > WIDTH - right - 90 else "middle"
    parts.append(f'<text class="viz-value" x="{ex:.1f}" y="{mid - 16}" text-anchor="{anchor}">'
                 f'Estimate {escape(format_inr(estimate))}/sq yd</text>')
    legend_y = height - 12
    parts.append(f'<circle class="viz-dot" cx="{left + 6}" cy="{legend_y - 4}" r="5"/>'
                 f'<text class="viz-legend" x="{left + 16}" y="{legend_y}">Estimate (shaded: {pct}% range)</text>'
                 f'<circle class="viz-dot-muted" cx="{left + 236}" cy="{legend_y - 4}" r="4.5"/>'
                 f'<text class="viz-legend" x="{left + 246}" y="{legend_y}">Comparable listings</text>')
    parts.append("</svg>")
    return Markup("".join(parts))
