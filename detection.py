"""
Repetition and EMG onset detection algorithms.

This module implements automatic and (optional) interactive/manual algorithms to:
- detect concentric phases from velocity peaks,
- infer repetition starts preceding each concentric start,
- package index pairs for biomechanical phases,
- detect EMG onsets using EstOpt on a TKEO-based envelope.

Notes
-----
• Units used downstream:
    - Velocity: m/s
    - Force: N
    - Power: W
    - EMG (raw/filtered): mV
    - EMG envelope (TKEO-based): ~mV² (nonlinear transform of mV)

• Not all functions here are invoked by `main_pipeline.run_pipeline`. Some are kept for
  interactive QA/annotation and exploratory analysis (e.g., the “manual” variants).

Dependencies
------------
NumPy, SciPy (signal + ndimage), scikit-learn (LinearRegression).
"""

from typing import List, Tuple, Sequence, Dict
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression


#Concentric phases and start repetitions

def detect_concentric_phases(
    velocity: np.ndarray,
    power: np.ndarray,
    time: np.ndarray,
    fs: float,
    min_peak_height: float = 0.09,
    num_peaks: int = 3,
    min_distance_s: float = 1.30,
) -> Tuple[List[int], List[int], List[float], List[int]]:
    """
    Detect concentric phases centered on the top-N positive velocity peaks.

    For each selected velocity peak:
      1) Finds the zero-crossing *before* the peak (negative → positive) → concentric start.
      2) Finds the zero-crossing *after* the peak (positive → negative) → concentric end.
      3) Extracts the peak power value within [start, end).

    Args:
        velocity: Velocity signal (m/s).
        power: Power signal (W), same length as `velocity`.
        time: Time vector aligned with `velocity` (s).
        fs: Sampling frequency (Hz).
        min_peak_height: Minimum velocity (m/s) for a point to qualify as a peak.
        num_peaks: Maximum number of concentric phases to return (top-N by peak height).
        min_distance_s: Minimum temporal distance (s) enforced between chosen peaks.

    Returns:
        concentric_starts: List of concentric start indices.
        concentric_ends: List of concentric end indices.
        peak_powers: List of peak power values (W) per concentric phase.
        peak_indices: List of velocity peak indices (samples) used to define phases.

    Notes:
        • Indices are returned in chronological order.
        • If no peaks are found above threshold, returns empty lists.
    """
    concentric_starts: List[int] = []
    concentric_ends: List[int] = []
    peak_powers: List[float] = []
    peak_indices: List[int] = []

    # 1) Find candidate peaks
    peaks, _ = find_peaks(velocity, height=min_peak_height)
    if peaks.size == 0:
        return [], [], [], []

    # 2) Sort by peak height (desc)
    sorted_peaks = sorted(peaks, key=lambda i: velocity[i], reverse=True)
    print("\nAll detected peaks above threshold:")
    for idx in sorted_peaks:
        print(f" → Index: {idx} (t = {idx / fs:.2f} s, vel = {velocity[idx]:.3f})")

    # 3) Keep top-N peaks with a min distance constraint
    selected_peaks: List[int] = []
    min_distance_samples = int(min_distance_s * fs)

    for peak_idx in sorted_peaks:
        if all(abs(peak_idx - sel) > min_distance_samples for sel in selected_peaks):
            selected_peaks.append(peak_idx)
        if len(selected_peaks) >= num_peaks:
            break

    selected_peaks.sort()  # chronological

    # 4) Derive phase limits and peak power per phase
    n = len(velocity)
    for peak_idx in selected_peaks:
        # before: negative → positive
        start_idx = 0
        for i in range(peak_idx - 1, 0, -1):
            if velocity[i - 1] <= 0 and velocity[i] > 0:
                start_idx = i
                break

        # after: positive → negative
        end_idx = n - 1
        for i in range(peak_idx + 1, n):
            if velocity[i - 1] > 0 and velocity[i] <= 0:
                end_idx = i
                break

        seg = power[start_idx:end_idx]  # [start, end)
        peak_pwr = float(np.max(seg)) if seg.size > 0 else 0.0

        concentric_starts.append(start_idx)
        concentric_ends.append(end_idx)
        peak_powers.append(peak_pwr)
        peak_indices.append(peak_idx)

    return concentric_starts, concentric_ends, peak_powers, peak_indices


def detect_and_correct_concentric_phases(
    velocity: np.ndarray,
    power: np.ndarray,
    time: np.ndarray,
    fs: float,
    min_peak_height: float = 0.15,
    num_peaks: int = 3,
    min_distance_s: float = 1.30,
) -> Tuple[List[int], List[int], List[float], List[int]]:
    """
    Same as `detect_concentric_phases`, but allows interactive/manual correction via `input()`.

    This is intended for QA/annotation sessions. In batch runs, prefer the automatic function.

    Args:
        velocity, power, time, fs: See `detect_concentric_phases`.
        min_peak_height, num_peaks, min_distance_s: See `detect_concentric_phases`.

    Returns:
        corrected_starts, corrected_ends, peak_powers, peak_indices
    """
    concentric_starts, concentric_ends, peak_powers, peak_indices = detect_concentric_phases(
        velocity=velocity,
        power=power,
        time=time,
        fs=fs,
        min_peak_height=min_peak_height,
        num_peaks=num_peaks,
        min_distance_s=min_distance_s,
    )

    if len(concentric_starts) == 0:
        print(" No peaks found above threshold.")
        return [], [], [], []

    print("\n Detected concentric phases:")
    for i, (start, end, peak) in enumerate(zip(concentric_starts, concentric_ends, peak_indices)):
        print(f"  [{i}] Start: {start} ({start/fs:.2f}s) — End: {end} ({end/fs:.2f}s) — Peak: {peak} ({peak/fs:.2f}s)")

    corrected_starts = concentric_starts.copy()
    corrected_ends = concentric_ends.copy()

    while True:
        edit_choice = input("\n Do you want to edit a [start], [end], [both], or [none]? ").strip().lower()
        if edit_choice == "none":
            print(" No further edits.")
            break

        if edit_choice in ["start", "end", "both"]:
            try:
                idx = int(input("Enter the index of the repetition to edit (e.g. 0, 1, 2): "))
                if edit_choice in ["start", "both"]:
                    new_start = int(input(f"New START index for repetition {idx}: "))
                    corrected_starts[idx] = new_start
                if edit_choice in ["end", "both"]:
                    new_end = int(input(f"New END index for repetition {idx}: "))
                    corrected_ends[idx] = new_end
            except Exception as e:
                print(f" Invalid input: {e}")
                continue
        else:
            print(" Invalid option. Please type 'start', 'end', 'both' or 'none'.")

    return corrected_starts, corrected_ends, peak_powers, peak_indices


def detect_rep_starts_from_concentric(
    con_starts: Sequence[int],
    velocity: np.ndarray,
    fs: float,
    max_inflection_velocity: float = 0.03,
    min_eccentric_duration: float = 0.60,
) -> List[int]:
    """
    Infer repetition starts preceding each concentric start.

    Strategy:
      (1) Backtrack from each `con_start` to the first valid zero-crossing (vel > 0 → vel ≤ 0),
          enforcing at least `min_eccentric_duration` seconds of eccentric motion.
      (2) Prefer the last inflection (local maximum) immediately before that zero-crossing if:
          - its velocity is small but > 0 (≤ `max_inflection_velocity`)
          - its time distance to the zero-crossing is ≤ 10 ms
          - the eccentric duration from that inflection to `con_start` is ≥ `min_eccentric_duration`
      (3) Otherwise, fall back to the zero-crossing.

    Args:
        con_starts: Indices of concentric-phase starts.
        velocity: Velocity signal (m/s).
        fs: Sampling frequency (Hz).
        max_inflection_velocity: Upper bound (m/s) for the local max to be considered “slow”.
        min_eccentric_duration: Minimum eccentric duration (s) before the concentric start.

    Returns:
        rep_starts: List of detected repetition-start indices.

    Notes:
        • This function prints diagnostic messages useful during development/QA.
    """
    velocity = np.asarray(velocity, dtype=float)
    rep_starts: List[int] = []
    min_samples_eccentric = int(min_eccentric_duration * fs)

    for cs in con_starts:
        print(f"\n Analyzing con_start at index {cs} (t = {cs / fs:.2f} s)")

        # Step 1: search backwards for a valid zero-crossing
        search_window_start = max(0, cs - int(1.7 * fs))
        zero_cross_idx = None
        for i in range(cs - 1, search_window_start, -1):
            if np.sign(velocity[i - 1]) != np.sign(velocity[i]):
                eccentric_duration = cs - i
                if eccentric_duration >= min_samples_eccentric:
                    zero_cross_idx = i
                    print(f"  → Zero-crossing at idx = {zero_cross_idx} (t = {zero_cross_idx / fs:.2f} s, vel = {velocity[zero_cross_idx]:.3f})")
                    break

        if zero_cross_idx is None:
            print("No valid zero-crossing found → skipping repetition")
            continue  # skip this rep

        # Step 2: try to find the last local max before that zero-crossing
        window_start = max(0, zero_cross_idx - int(1.7 * fs))
        segment = velocity[window_start:zero_cross_idx]
        segment_indices = np.arange(window_start, zero_cross_idx)
        local_maxima = argrelextrema(segment, np.greater)[0]

        if local_maxima.size > 0:
            inflection_idx = int(segment_indices[local_maxima[-1]])
            inflection_velocity = float(velocity[inflection_idx])
            eccentric_duration_inflection = cs - inflection_idx
            distance_to_zero_cross = zero_cross_idx - inflection_idx  # samples
            distance_to_zero_cross_sec = distance_to_zero_cross / fs

            print(f"  → Inflection at idx = {inflection_idx} (t = {inflection_idx / fs:.2f} s, vel = {inflection_velocity:.3f})")
            print(f"     Eccentric duration = {eccentric_duration_inflection / fs:.3f} s")
            print(f"     Distance to zero-crossing = {distance_to_zero_cross_sec:.3f} s")

            if (
                inflection_velocity > 0
                and inflection_velocity <= max_inflection_velocity
                and (eccentric_duration_inflection >= min_samples_eccentric)
                and distance_to_zero_cross_sec <= 0.01
            ):
                rep_starts.append(inflection_idx)
                print("Using inflection point as rep_start")
                continue

        # Fallback to zero-crossing
        rep_starts.append(zero_cross_idx)
        print("Using zero-crossing as fallback rep_start")

    return rep_starts


def detect_and_correct_rep_starts_from_concentric(
    con_starts: Sequence[int],
    velocity: np.ndarray,
    fs: float,
    max_inflection_velocity: float = 0.30,
    min_eccentric_duration: float = 0.60,
) -> List[int]:
    """
    Interactive/manual version of `detect_rep_starts_from_concentric`.

    Returns:
        corrected_starts: Sorted list of edited/added repetition starts (indices).
    """
    rep_starts = detect_rep_starts_from_concentric(
        con_starts=con_starts,
        velocity=velocity,
        fs=fs,
        max_inflection_velocity=max_inflection_velocity,
        min_eccentric_duration=min_eccentric_duration,
    )

    corrected_starts = rep_starts.copy()
    print("\nDetected rep_starts:")
    for i, idx in enumerate(corrected_starts):
        print(f"  [{i}] Index: {idx} (t = {idx / fs:.2f} s)")

    while True:
        edit_choice = input("\nDo you want to [edit], [add], or [done]? ").strip().lower()
        if edit_choice == "done":
            print("Finished editing.")
            break
        elif edit_choice == "edit":
            try:
                idx = int(input("Enter the index of the repetition to edit (e.g. 0, 1, 2): "))
                new_val = int(input(f"New START index for repetition {idx}: "))
                corrected_starts[idx] = new_val
                print(f"  → Updated rep_start[{idx}] to {new_val} (t = {new_val / fs:.2f} s)")
            except Exception as e:
                print(f"  Invalid input: {e}")
        elif edit_choice == "add":
            try:
                new_val = int(input("Enter index of NEW rep_start to add: "))
                corrected_starts.append(new_val)
                corrected_starts.sort()
                print(f"  → Added new rep_start: {new_val} (t = {new_val / fs:.2f} s)")
            except Exception as e:
                print(f"  Invalid input: {e}")
        else:
            print("  Invalid option. Type 'edit', 'add', or 'done'.")

    return corrected_starts


def get_phase_index_pairs(
    rep_starts: Sequence[int],
    con_starts: Sequence[int],
    con_ends: Sequence[int],
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Package index pairs for phase-level analysis.

    Args:
        rep_starts: Repetition start indices.
        con_starts: Concentric phase start indices.
        con_ends: Concentric phase end indices.

    Returns:
        Dict with:
            - 'rep_con_pairs': list of (rep_start, con_end)
            - 'con_phase_pairs': list of (con_start, con_end)
    """
    rep_con_pairs = list(zip(rep_starts, con_ends))
    con_phase_pairs = list(zip(con_starts, con_ends))
    return {"rep_con_pairs": rep_con_pairs, "con_phase_pairs": con_phase_pairs}


# EMG / EstOpt

def detect_emg_onsets_estopt(
    emg_signal: np.ndarray,
    fs: float,
    concentric_starts: Sequence[int],
    time_vector: np.ndarray,
    sigma_ms: float = 50.0,
    window_ms: float = 300.0,
) -> Tuple[List[int], List[float], np.ndarray]:
    """
    Detect EMG onsets using EstOpt applied to a TKEO-based, Gaussian-smoothed envelope.

    Pipeline per repetition:
      1) Build the TKEO signal: ψ[x](n) = x(n)^2 − x(n−1)·x(n+1)
      2) Envelope = GaussianSmooth(|TKEO|, sigma = `sigma_ms`)
      3) In a window of `window_ms` before the concentric start, sweep a candidate onset index i
         that minimizes: SSE(pre) + SSE(post), where pre is fit by a constant level and post by
         a linear trend (simple two-regime approximation of the activation transition).

    Args:
        emg_signal: EMG signal (cut), in mV, same length as `time_vector`.
        fs: Sampling frequency (Hz).
        concentric_starts: Indices of concentric starts in the cut signal.
        time_vector: Time (s) aligned with `emg_signal`.
        sigma_ms: Gaussian sigma for envelope smoothing (ms).
        window_ms: Look-back window before each concentric start (ms).

    Returns:
        onset_indices: Detected onset indices (samples, in the same reference as inputs).
        onset_times: Detected onset times (s).
        envelope: Smoothed absolute TKEO envelope (~mV²).

    Raises:
        ValueError: If 'emg_signal' and 'time_vector' have different lengths.
    """
    if len(emg_signal) != len(time_vector):
        raise ValueError("emg_signal and time_vector must have the same length.")

    # --- TKEO (nonlinear, unit ~ mV^2) ---
    tkeo = np.copy(emg_signal)
    tkeo[1:-1] = emg_signal[1:-1] ** 2 - emg_signal[:-2] * emg_signal[2:]

    # --- Smoothed envelope ---
    sigma_samples = max(1, int((sigma_ms / 1000.0) * fs))
    envelope = gaussian_filter1d(np.abs(tkeo), sigma=sigma_samples)

    # --- EstOpt windowing params ---
    window_samples = int((window_ms / 1000.0) * fs)
    guard = max(1, int(0.01 * fs))   # 10 ms guard to avoid edges
    min_window = int(0.10 * fs)      # 100 ms minimum window

    onset_indices: List[int] = []
    onset_times: List[float] = []

    # --- Loop over reps ---
    for cs in concentric_starts:
        start_idx = cs - window_samples
        if start_idx < 0 or (cs - start_idx) < min_window:
            continue

        env_win = envelope[start_idx:cs]
        time_win = time_vector[start_idx:cs]

        best_idx, best_error = None, np.inf

        # Sweep candidate onset within the window (excluding guards)
        for i in range(guard, len(env_win) - guard):
            # Pre: constant model (mean)
            error_pre = np.sum((env_win[:i] - np.mean(env_win[:i])) ** 2)

            # Post: simple linear trend
            model = LinearRegression().fit(time_win[i:].reshape(-1, 1), env_win[i:])
            error_post = np.sum((env_win[i:] - model.predict(time_win[i:].reshape(-1, 1))) ** 2)

            total_error = error_pre + error_post
            if total_error < best_error:
                best_error = total_error
                best_idx = i

        if best_idx is not None:
            global_idx = start_idx + best_idx
            onset_indices.append(global_idx)
            onset_times.append(float(time_vector[global_idx]))

    return onset_indices, onset_times, envelope