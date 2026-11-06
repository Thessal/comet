from __future__ import annotations
import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from comet.dmgr import AlignedFrame, DataManager
from comet.rl.optimizer import PortfolioOptimizer

try:
    from comet.comet_rs import StackEnv
except ImportError:
    StackEnv = None  # type: ignore

try:
    from comet.models.model import AgentModel
except ImportError:
    AgentModel = None  # type: ignore


def reload_rust(build: bool = True, release: bool = False):
    """
    Rebuilds `crates/runtime` (if `build=True`) and hot-reloads `comet_rs.StackEnv`
    into the current Python/Jupyter process without restarting the kernel.
    """
    global StackEnv
    repo_root = Path(__file__).resolve().parent.parent.parent
    profile = "release" if release else "debug"

    if build:
        cmd = ["cargo", "build", "-p", "runtime"]
        if release:
            cmd.append("--release")
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    so_src = repo_root / "target" / profile / "libcomet_rs.so"
    if not so_src.exists():
        raise FileNotFoundError(f"Compiled library not found at {so_src}")

    so_link = Path(__file__).resolve().parent / "comet_rs.so"
    if so_link.exists() or so_link.is_symlink():
        so_link.unlink()
    so_link.symlink_to(so_src)

    # Copy to a unique path with a fresh inode so Linux dlopen() loads the new .so
    mtime_ns = so_src.stat().st_mtime_ns
    hot_dir = repo_root / "target" / profile / ".hot"
    hot_dir.mkdir(parents=True, exist_ok=True)
    hot_so = hot_dir / f"comet_rs_{mtime_ns}_{os.getpid()}.so"
    shutil.copy2(so_src, hot_so)

    loader = importlib.machinery.ExtensionFileLoader("comet_rs", str(hot_so))
    spec = importlib.util.spec_from_loader("comet_rs", loader)
    if spec is None:
        raise ImportError(f"Could not create spec from {hot_so}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)

    sys.modules["comet.comet_rs"] = mod
    StackEnv = mod.StackEnv
    return StackEnv


__all__ = [
    "StackEnv",
    "DataManager",
    "AlignedFrame",
    "PortfolioOptimizer",
    "AgentModel",
    "reload_rust",
]
