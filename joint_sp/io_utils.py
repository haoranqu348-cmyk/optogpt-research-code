"""Crash-safe file publication helpers used by the joint s+p pipeline."""

import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        writer(Path(tmp_name))
        # Windows requires a writable descriptor for fsync on regular files.
        with open(tmp_name, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_pickle_dump(value, path):
    _atomic_write(path, lambda tmp: _pickle_dump(value, tmp))


def _pickle_dump(value, path):
    with open(path, "wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def atomic_json_dump(value, path, **kwargs):
    options = {"indent": 2, "ensure_ascii": True, **kwargs}

    def writer(tmp):
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, **options)

    _atomic_write(path, writer)


def atomic_numpy_save(value, path):
    def writer(tmp):
        with open(tmp, "wb") as handle:
            np.save(handle, value)

    _atomic_write(path, writer)


def atomic_torch_save(value, path):
    _atomic_write(path, lambda tmp: torch.save(value, str(tmp)))
