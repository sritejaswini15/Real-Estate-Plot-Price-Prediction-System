"""Human-friendly formatting for rupee amounts, areas and percentages."""

from __future__ import annotations

import math


def _group_indian(digits: str) -> str:
    """``"5840000"`` -> ``"58,40,000"`` (lakh/crore digit grouping)."""
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join([*groups, tail])


def format_inr(amount: float | None) -> str:
    """Full rupee amount with Indian grouping: ``₹58,40,000``."""
    if amount is None or not math.isfinite(amount):
        return "—"
    rounded = int(round(amount))
    sign = "−" if rounded < 0 else ""
    return f"{sign}₹{_group_indian(str(abs(rounded)))}"


def _three_significant(value: float) -> str:
    if value >= 100:
        text = f"{value:.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def format_inr_short(amount: float | None) -> str:
    """Compact rupee amount: ``₹58.4 lakh``, ``₹1.05 crore``, ``₹38,000``."""
    if amount is None or not math.isfinite(amount):
        return "—"
    sign = "−" if amount < 0 else ""
    value = abs(amount)
    if round(value / 1e7, 2) >= 1:
        return f"{sign}₹{_three_significant(value / 1e7)} crore"
    if round(value / 1e5, 2) >= 1:
        return f"{sign}₹{_three_significant(value / 1e5)} lakh"
    return f"{sign}{format_inr(value)}"


def format_number(value: float | None, decimals: int = 0) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if decimals == 0:
        return _group_indian(str(int(round(value))))
    whole, _, frac = f"{value:.{decimals}f}".partition(".")
    frac = frac.rstrip("0")
    return _group_indian(whole) + (f".{frac}" if frac else "")


def format_area(sqyd: float | None) -> str:
    return "—" if sqyd is None else f"{format_number(sqyd, 2)} sq yd"


def format_pct(value: float | None, decimals: int = 0) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value * 100:.{decimals}f}%"


INPUT_LABELS = {
    "area_sqyd": "plot area",
    "locality": "locality",
    "facing": "facing",
    "open_sides": "open sides",
    "road_width_ft": "road width",
    "in_project": "layout membership",
}


def describe_inputs(names) -> str:
    """``["area_sqyd", "locality"]`` -> ``"plot area and locality"``."""
    labels = [INPUT_LABELS.get(name, name) for name in names]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def format_multiplier(log_value: float) -> str:
    """``-0.4`` -> ``"−33%"``: a log residual as a percentage change."""
    change = math.exp(log_value) - 1.0
    return f"{'+' if change >= 0 else '−'}{abs(change) * 100:.0f}%"
