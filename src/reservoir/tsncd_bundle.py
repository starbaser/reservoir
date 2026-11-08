"""Where the tsncd bundle is, for PyNCD's page writer and headless capture.

PyNCD reads the bundle through the `TSNCD_DIST` environment variable. The Nix shell
exports it. Without Nix, `tools/setup_tsncd.py` builds the bundle under
`third_party/tsncd/dist` and writes the variable to `.env`, which `uv run --env-file
.env` would load. A plain `uv run` does not, so the entry points of this package name
the vendored bundle themselves when the variable is unset and the bundle is there.
"""

from __future__ import annotations

import os
import pathlib

VARIABLE = "TSNCD_DIST"
VENDORED_DIST = (
    pathlib.Path(__file__).resolve().parents[2] / "third_party" / "tsncd" / "dist"
)


def name_vendored_bundle_when_unset() -> pathlib.Path | None:
    """Set `TSNCD_DIST` to the vendored bundle when the variable is unset and the
    bundle exists, and return the path PyNCD will read, or None when nothing names
    one."""
    named = os.environ.get(VARIABLE)
    if named:
        return pathlib.Path(named)
    if (VENDORED_DIST / "index.html").is_file():
        os.environ[VARIABLE] = str(VENDORED_DIST)
        return VENDORED_DIST
    return None
