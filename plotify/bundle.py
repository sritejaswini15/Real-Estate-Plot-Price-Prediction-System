"""Save and load the trained model bundle with an integrity check.

The bundle is a joblib (pickle) file, and loading a pickle can run code. The
loader therefore unpickles only a file whose SHA-256 matches the manifest
written beside it at training time. That catches a corrupted or swapped file.
It is no defence against someone who can write to both files. Keep ``models/``
read-only in deployment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import joblib
import sklearn

BUNDLE_FORMAT = 1


class BundleError(RuntimeError):
    """The model bundle is missing, altered or incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(bundle_path: Path) -> Path:
    return bundle_path.with_name(bundle_path.stem + ".manifest.json")


def save_bundle(bundle: dict[str, Any], path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {**bundle, "format": BUNDLE_FORMAT}
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)
    manifest = {
        "file": path.name,
        "sha256": sha256_file(path),
        "format": BUNDLE_FORMAT,
        "model_version": bundle["metadata"]["model_version"],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    manifest_path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_bundle(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise BundleError(f"Model bundle not found at {path}. Train one with: python -m plotify train")
    mpath = manifest_path(path)
    if not mpath.is_file():
        raise BundleError(f"Manifest {mpath.name} missing next to {path.name}; refusing to load an unverified pickle.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    actual = sha256_file(path)
    if manifest.get("sha256") != actual:
        raise BundleError(f"{path.name} does not match its manifest checksum; refusing to load it.")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"Bundle format {manifest.get('format')} is not supported (expected {BUNDLE_FORMAT}); retrain.")
    if manifest.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {manifest.get('sklearn_version')} but "
            f"{sklearn.__version__} is installed; retrain if predictions look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise BundleError(f"{path.name} is not a plotify model bundle.")
    bundle["manifest"] = manifest
    return bundle
