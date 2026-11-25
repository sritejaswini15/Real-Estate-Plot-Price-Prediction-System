"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plotify_model.joblib"


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    model_path: Path = DEFAULT_MODEL_PATH
    # Estimates per client IP per minute, across the web form and the API; 0 disables.
    rate_limit_per_minute: int = 60
    # Number of reverse proxies in front of the app whose X-Forwarded-* headers to trust.
    trusted_proxies: int = 0
    # Send Strict-Transport-Security; enable only when served over HTTPS.
    hsts: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            model_path=Path(env.get("PLOTIFY_MODEL_PATH") or DEFAULT_MODEL_PATH),
            rate_limit_per_minute=_int(env, "PLOTIFY_RATE_LIMIT", 60),
            trusted_proxies=_int(env, "PLOTIFY_TRUSTED_PROXIES", 0),
            hsts=_bool(env, "PLOTIFY_HSTS", False),
            log_level=(env.get("PLOTIFY_LOG_LEVEL") or "INFO").upper(),
        )
