"""Build the tsncd bundle at the revision `flake.lock` pins, for an installation without
Nix.

    python3 tools/build_tsncd_bundle.py

tsncd is cloned under `third_party/tsncd/` and its bundle is built with `npm ci` and
`npm run build`. PyNCD's page writer and headless capture read the bundle through
`TSNCD_DIST`, and the script writes that variable into `.env` when the file does not
name it. It needs `git`, `node` 24 and `npm` on the path and uses nothing outside the
standard library, because it runs before the Python environment exists.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCK = ROOT / "flake.lock"
TSNCD_DIRECTORY = ROOT / "third_party" / "tsncd"
ENV_FILE = ROOT / ".env"
TSNCD_DIST_LINE = "TSNCD_DIST=third_party/tsncd/dist"


@dataclass(frozen=True)
class Pin:
    owner: str
    """The GitHub account holding the repository."""

    repo: str
    """The repository name."""

    rev: str
    """The commit the flake lock pins."""

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}.git"


def pin_from_lock(*, name: str) -> Pin:
    """The pin of the flake input `name`, read from `flake.lock`."""
    nodes = json.loads(LOCK.read_text())["nodes"]
    locked = nodes[name]["locked"]
    if locked["type"] != "github":
        raise ValueError(f"the input {name} is not a GitHub input: {locked}")
    return Pin(owner=locked["owner"], repo=locked["repo"], rev=locked["rev"])


def run(*, command: Sequence[str], cwd: pathlib.Path) -> None:
    print("$", " ".join(command), f"(in {cwd.relative_to(ROOT)})")
    subprocess.run(command, cwd=cwd, check=True)


def clone_at(*, pin: Pin, into: pathlib.Path) -> None:
    """A shallow checkout of `pin.rev` into `into`, fetched by commit."""
    into.mkdir(parents=True, exist_ok=True)
    run(command=["git", "init", "--quiet"], cwd=into)
    run(command=["git", "remote", "add", "origin", pin.url], cwd=into)
    run(
        command=["git", "fetch", "--quiet", "--depth", "1", "origin", pin.rev], cwd=into
    )
    run(command=["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=into)


def build_tsncd(*, pin: Pin) -> pathlib.Path:
    """tsncd at `pin.rev` under `third_party/tsncd/`, with its bundle built."""
    if TSNCD_DIRECTORY.exists():
        shutil.rmtree(TSNCD_DIRECTORY)
    clone_at(pin=pin, into=TSNCD_DIRECTORY)
    run(command=["npm", "ci", "--no-audit", "--no-fund"], cwd=TSNCD_DIRECTORY)
    run(command=["npm", "run", "build"], cwd=TSNCD_DIRECTORY)
    dist = TSNCD_DIRECTORY / "dist"
    if not (dist / "index.html").is_file():
        raise FileNotFoundError(f"{dist} holds no index.html after the build")
    return dist


def record_bundle_in_env_file() -> bool:
    """Whether `.env` was given the `TSNCD_DIST` line. A file already naming the
    variable is left as it is."""
    existing = ENV_FILE.read_text() if ENV_FILE.exists() else ""
    if any(line.startswith("TSNCD_DIST=") for line in existing.splitlines()):
        return False
    separator = "" if not existing or existing.endswith("\n") else "\n"
    ENV_FILE.write_text(f"{existing}{separator}{TSNCD_DIST_LINE}\n")
    return True


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    for tool in ("git", "node", "npm"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} is not on the path")
    pin = pin_from_lock(name="tsncd")
    dist = build_tsncd(pin=pin)
    print(f"tsncd {pin.rev[:7]} bundle is in {dist.relative_to(ROOT)}")
    if record_bundle_in_env_file():
        print(f"wrote {TSNCD_DIST_LINE} to .env")


if __name__ == "__main__":
    main()
