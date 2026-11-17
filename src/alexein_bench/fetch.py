"""Datasets are downloaded, never bundled: each from its authors, at a pinned version, checked
against the digest it had when the published number was produced."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "alexein-bench"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, digest: str | None) -> Path:
    """`url` to `target`, once. With `digest`, a file that does not match is refused."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            with httpx.stream("GET", url, follow_redirects=True, timeout=300) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    tmp.write(chunk)
            part = Path(tmp.name)
        part.replace(target)
    if digest and (got := sha256(target)) != digest:
        target.unlink()
        raise SystemExit(f"{target.name}: sha256 {got} is not the pinned {digest}")
    return target


def clone(repo: str, commit: str, target: Path, keep: tuple[str, ...]) -> Path:
    """A git repository at `commit`, keeping only the paths in `keep`."""
    if target.exists():
        return target
    if not shutil.which("git"):
        raise SystemExit(f"git is needed to fetch {repo}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        work = Path(tmp) / "repo"
        print(f"Fetching {repo} at {commit[:7]}")
        subprocess.run(["git", "init", "-q", str(work)], check=True)
        git = ["git", "-C", str(work)]
        subprocess.run([*git, "remote", "add", "origin", repo], check=True)
        subprocess.run([*git, "fetch", "-q", "--depth", "1", "origin", commit], check=True)
        subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], check=True)
        staged = Path(tmp) / "kept"
        staged.mkdir()
        for path in keep:
            source = work / path
            dest = staged / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
        staged.rename(target)
    return target
