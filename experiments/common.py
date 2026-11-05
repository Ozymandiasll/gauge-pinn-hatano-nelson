# Author: Ramsés C. S. Almeida (UFAL)

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def _jsonable(obj):
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj

def result_paths(name: str, results_dir=None):
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    return d / f"{name}.npz", d / f"{name}.json"

def already_done(name: str, results_dir=None) -> bool:
    npz, js = result_paths(name, results_dir)
    return npz.exists() and js.exists()

def save_result(name: str, arrays: dict, meta: dict, results_dir=None) -> None:
    npz, js = result_paths(name, results_dir)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, **arrays)
    meta = dict(_jsonable(meta))
    meta.setdefault("git_hash", git_hash())
    meta.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    with open(js, "w") as f:
        json.dump(meta, f, indent=1)

def load_result(name: str, results_dir=None):
    npz, js = result_paths(name, results_dir)
    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    with open(js) as f:
        meta = json.load(f)
    return arrays, meta

def save_json(name: str, obj: dict, results_dir=None) -> None:
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.json", "w") as f:
        json.dump(_jsonable(obj), f, indent=1)

def load_json(name: str, results_dir=None) -> dict:
    d = Path(results_dir) if results_dir is not None else RESULTS_DIR
    with open(d / f"{name}.json") as f:
        return json.load(f)
