"""Each Dockerfile copies an explicit file list, so .env and caches can
never leak into an image. The cost is that a new import can silently miss
the list and crash the pod at startup (it did: swarm/features.py started
importing tier0). This rebuilds each image's file list in a temp directory
and imports the app from there, with the repo off the path.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

IMAGES = {
    "Dockerfile.router": ["risk_router.app"],
    "Dockerfile.swarm": ["swarm.ingestion", "swarm.quant"],
    "Dockerfile.inference": ["swarm.inference_agent"],
}


def _copied_files(dockerfile: Path) -> list[tuple[Path, str]]:
    """(source, destination) for every COPY except the requirements files."""
    out: list[tuple[Path, str]] = []
    joined = dockerfile.read_text().replace("\\\n", " ")
    for line in joined.splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY" or any(p.startswith("--") for p in parts):
            continue
        *sources, dest = parts[1:]
        out += [(ROOT / s, dest) for s in sources if "requirements" not in s]
    return out


@pytest.mark.parametrize(("dockerfile", "modules"), IMAGES.items())
def test_image_file_list_covers_every_import(dockerfile: str, modules: list[str], tmp_path: Path) -> None:
    for source, dest in _copied_files(ROOT / dockerfile):
        target = tmp_path / dest.removeprefix("./")
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target = target / source.name if dest.endswith("/") else target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
    code = "; ".join(f"import {m}" for m in modules)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=False,
                            env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode == 0, f"{dockerfile}: {result.stderr.strip().splitlines()[-1]}"
