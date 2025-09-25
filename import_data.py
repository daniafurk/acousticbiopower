"""
Data-loading utilities for biomechanical experiments (.txt files).

This module provides robust helpers to load synchronized signals from whitespace-
delimited text files exported by acquisition tools (e.g., OpenSignals).
It automatically:

1) Skips any header block (comments, blank lines, non-numeric labels) until the
   first fully-numeric row.
2) Infers whether a DI (digital input) column is present by counting numeric
   columns in the first numeric row.
3) Infers the exercise type from the filename ("bp" for bench press, "squat")
   to select the appropriate EMG channel (arm vs. leg).
4) Maps columns to (ACCZ, LOADCELL, EMG) consistently across file layouts.

Column schemas (0-based indices):

WITHOUT DI (8 numeric columns):
    Nseq, LOADCELL, ACCX, ACCY, ACCZ, EMGA, EMGL, ECG
    ACCZ  -> 4
    LOAD  -> 1
    EMG   -> 5 (bp → EMGA) or 6 (squat → EMGL)

WITH DI (≥9 numeric columns):
    Nseq, DI, LOADCELL, ACCX, ACCY, ACCZ, EMGA, EMGL, ECG, ...
    ACCZ  -> 5
    LOAD  -> 2
    EMG   -> 6 (bp → EMGA) or 7 (squat → EMGL)

Returns:

acc_signal : np.ndarray
    ACCZ channel (as float).
loadcell_signal : np.ndarray
    Load cell (mass/force raw counts or already converted upstream).
emg_signal : np.ndarray
    EMG (arm for bench press, leg for squat), as float.

Notes:
- Lines starting with '#' and any non-numeric header lines are skipped.
- Rows containing NaNs in any column are dropped before slicing channels.
"""

import os
from typing import Tuple
import numpy as np


def _scan_header_and_first_numeric(file_path: str) -> Tuple[int, int]:
    """
    Scan the file to determine how many header lines to skip and how many
    numeric columns the first data row contains.

    The scanner:
      • Skips comment lines starting with '#', blank lines, and any initial
        non-numeric label lines.
      • Detects the first fully-numeric line and counts its columns.

    Parameters
    file_path : str
        Path to the .txt file.

    Returns
    skip_header_lines : int
        Number of lines to skip before the first numeric row.
    num_numeric_columns : int
        Number of numeric columns in the first numeric row.

    Raises
    ValueError
        If no numeric row is found.
    """
    def is_numeric_line(tokens):
        if not tokens:
            return False
        try:
            # Be tolerant to decimal commas just in case
            _ = [float(t.replace(",", ".")) for t in tokens]
            return True
        except ValueError:
            return False

    skip = 0
    num_cols = None

    #Open in text mode with tolerant decoding, then use np.genfromtxt later.
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                skip += 1
                continue
            tokens = s.split()  # whitespace-delimited
            if is_numeric_line(tokens):
                num_cols = len(tokens)
                break
            # Non-numeric label/header line (e.g., "Nseq LOADCELL ...")
            skip += 1

    if num_cols is None:
        raise ValueError(f"No numeric data lines found in: {file_path}")

    return skip, num_cols


def _infer_exercise_from_filename(filename: str) -> str:
    """
    Infer exercise type from the filename.

    Returns
    "bp" for bench press, "squat" for squat.

    Raises
    ValueError if neither token is present.
    """
    name = filename.lower()
    if "bp" in name:
        return "bp"
    if "squat" in name:
        return "squat"
    raise ValueError(
        f"Cannot infer exercise type from filename (expected 'bp' or 'squat'): {filename}"
    )


def load_exercise_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load accelerometer (ACCZ), load cell, and EMG from a formatted .txt file.

    This function auto-detects:
      • how many header lines to skip (comments / non-numeric),
      • whether a DI column exists (by counting numeric columns),
      • the exercise type from filename ("bp" vs "squat") to pick EMG channel.

    Parameters
    file_path : str
        Path to the .txt file.

    Returns
    acc_signal : np.ndarray
        ACCZ channel (float).
    loadcell_signal : np.ndarray
        Load cell channel (float).
    emg_signal : np.ndarray
        EMG channel (float): arm for bench press, leg for squat.

    Raises
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the numeric matrix has fewer columns than expected for the mapping.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    skip_header, num_cols = _scan_header_and_first_numeric(file_path)
    exercise = _infer_exercise_from_filename(os.path.basename(file_path))

    #Decide mapping based on whether DI column is present
    has_di = num_cols >= 9  # OpenSignals numeric block typically has 9+ columns
    if not has_di:
        acc_idx = 4
        load_idx = 1
        emg_idx = 5 if exercise == "bp" else 6
    else:
        acc_idx = 5
        load_idx = 2
        emg_idx = 6 if exercise == "bp" else 7

    # Load numeric data (whitespace-delimited). Let genfromtxt handle comments.
    data = np.genfromtxt(
        file_path,
        comments="#",
        delimiter=None,      # whitespace
        dtype=float,
        invalid_raise=False, # tolerate ragged lines
        skip_header=skip_header,
    )

    # Ensure 2D
    if data.ndim == 1:
        data = np.atleast_2d(data)

    # Drop any row with NaN in any column
    data = data[~np.isnan(data).any(axis=1)]

    #Check: enough columns for the chosen mapping
    max_needed = max(acc_idx, load_idx, emg_idx)
    if data.shape[1] <= max_needed:
        raise ValueError(
            "Unexpected column count in "
            f"'{file_path}': got {data.shape[1]}, need ≥ {max_needed + 1} "
            f"(first-row cols={num_cols}, has_di={has_di}, exercise={exercise})."
        )

    #Slice channels and enforce contiguous float arrays
    acc_signal = np.ascontiguousarray(data[:, acc_idx].astype(float))
    loadcell_signal = np.ascontiguousarray(data[:, load_idx].astype(float))
    emg_signal = np.ascontiguousarray(data[:, emg_idx].astype(float))

    return acc_signal, loadcell_signal, emg_signal




