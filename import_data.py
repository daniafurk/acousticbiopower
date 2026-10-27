"""
Load standardized published exercise files.

Expected columns (one header line, whitespace-separated):
Nseq LOADCELL ACCX ACCY ACCZ EMGA EMGL ECG

Rows with a NaN in *any*
column are removed (now also rows with +/- infinity). This shortens
the time series: see warnings when rows are removed.
"""
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)
EXPECTED_COLUMNS = (
    "NSEQ", "LOADCELL", "ACCX", "ACCY", "ACCZ", "EMGA", "EMGL", "ECG"
)


def _infer_exercise_from_filename(filename: str) -> str:
    tokens = Path(filename).stem.lower().split("_")
    if "bp" in tokens:
        return "bp"
    if "squat" in tokens:
        return "squat"
    raise ValueError(f"Cannot determine exercise ('bp' or 'squat'): {filename}")


def load_raw_data(file_path: str) -> np.ndarray:
    """Return a finite N x 8 matrix, dropping invalid rows as before."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as f:
        columns = tuple(f.readline().strip().upper().split())
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected header in {path.name}: {columns!r}; expected {EXPECTED_COLUMNS!r}")
    data = np.loadtxt(path, skiprows=1, dtype=float, ndmin=2, encoding="utf-8-sig")
    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 numeric columns in {path.name}, got {data.shape[1]}")
    if data.shape[0] == 0:
        raise ValueError(f"No numeric rows in {path.name}")
    good_rows = np.isfinite(data).all(axis=1)
    removed = int((~good_rows).sum())
    if removed:
        logger.warning("Removed %d of %d rows with NaN/Inf from %s", removed, len(data), path)
        data = data[good_rows]
    if len(data) == 0:
        raise ValueError(f"No valid rows after filtering: {path}")
    return np.ascontiguousarray(data, dtype=float)


def load_exercise_data(file_path: str):
    """Return ACCZ, LOADCELL and the exercise-specific EMG channel."""
    data = load_raw_data(file_path)
    emg_col = 5 if _infer_exercise_from_filename(Path(file_path).name) == "bp" else 6
    return tuple(np.ascontiguousarray(data[:, idx]) for idx in (4, 1, emg_col))


def load_acc_axes(file_path: str):
    """Return ACCX, ACCY, ACCZ; available for other analysis scripts."""
    data = load_raw_data(file_path)
    return tuple(np.ascontiguousarray(data[:, idx]) for idx in (2, 3, 4))
