"""
Computes biomechanical and EMG-derived metrics per repetition.

This module includes:

Kinematics & mechanics
  - Velocity by numerical integration (m/s)
  - Instantaneous power (W = N × m/s)
  - Peak power and velocity per repetition (using filtered indices, reading raw values)
  - Rate of Power Development (RPD) from smoothed power (W/s)
  - Local (windowed) and overall power slopes to peak (W/s)

EMG features
  - Delay from EMG onset (EstOpt) to power peak (s)
  - Delay from EMG onset (EstOpt) to EMG envelope peak (s) (EMG Rise Time(s)) with robustness checks
  - EMG rise slope between onset and envelope peak (envelope units per second)
  - EMG median frequency between onset and concentric end (Hz)

Notes:
Power and velocity follow standard SI units (W, m/s).

All functions are vectorized per repetition via index pairs (start/end).
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import integrate
from scipy.signal import find_peaks, welch


#Power and Velocity Metrics

def compute_velocity(acc: np.ndarray, fs: int) -> np.ndarray:
    """
    Integrate acceleration to estimate velocity (trapezoidal rule).

    Parameters
    acc : np.ndarray
        Acceleration (m/s²), 1-D.
    fs : int
        Sampling frequency (Hz).

    Returns
    np.ndarray
        Velocity (m/s), same length as 'acc'.
    """
    dt = 1 / fs
    time = np.arange(len(acc)) * dt
    velocity = integrate.cumulative_trapezoid(acc, time, initial=0.0)
    return velocity


def compute_power(force: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous mechanical power.

    Parameters
    force : np.ndarray
        Force (N), 1-D.
    velocity : np.ndarray
        Velocity (m/s), 1-D.

    Returns
    np.ndarray
        Power (W), element-wise product.
    """
    return force * velocity


def compute_max_power_per_rep_from_filtered_index(
    power_filtered: np.ndarray,
    power_raw: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int]
) -> Tuple[List[float], List[int]]:
    """
    For each repetition, locate the peak index on the *filtered* power segment
    and read the value at that index from the *raw* power. This preserves the
    peak position robustness while reporting the raw amplitude.

    Parameters
    power_filtered : np.ndarray
        Filtered power (used to locate the peak).
    power_raw : np.ndarray
        Raw power (used to read the peak amplitude).
    starts, ends : Sequence[int]
        Start/end indices per repetition (inclusive on end when slicing +1).

    Returns
    max_powers : List[float]
        Max raw power per repetition. (Multiplied by 2)
    peak_indices : List[int]
        Global indices of the detected peaks (from 'power_filtered').
    """
    max_powers: List[float] = []
    peak_indices: List[int] = []

    n = len(power_filtered)
    for start, end in zip(starts, ends):
        s = max(0, int(start))
        e = min(n - 1, int(end))
        if e < s:
            max_powers.append(np.nan)
            peak_indices.append(-1)
            continue

        seg = power_filtered[s:e + 1]
        if seg.size == 0:
            max_powers.append(np.nan)
            peak_indices.append(-1)
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        peak_indices.append(peak_idx)

        # Read raw amplitude at the same location
        true_raw = float(power_raw[peak_idx])
        max_powers.append(2.0 * true_raw)

    return max_powers, peak_indices


def compute_max_power_raw_at_filtered_peaks_interactive(
    power_filtered: np.ndarray,
    power_raw: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int]
) -> Tuple[List[float], List[int]]:
    """
    Same as compute_max_power_per_rep_from_filtered_index, but allows manual
    override of peak indices.

    Returns the updated raw peak values (×2) and their indices.
    """
    max_powers_raw: List[float] = []
    peak_indices: List[int] = []

    print("\n--- Automatic Peak Detection (Power) ---")
    n = len(power_filtered)
    for i, (start, end) in enumerate(zip(starts, ends)):
        s = max(0, int(start)); e = min(n - 1, int(end))
        seg = power_filtered[s:e + 1]
        if seg.size == 0:
            max_powers_raw.append(np.nan)
            peak_indices.append(-1)
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        peak_val_raw = float(power_raw[peak_idx])
        max_powers_raw.append(2.0 * peak_val_raw)
        peak_indices.append(peak_idx)

        print(f"Rep {i+1}: Peak @ {peak_idx} → Raw power = {2.0 * peak_val_raw:.4f} W")

    print("\n--- Manual Override ---")
    print("Type the repetition number to edit (1-based), or 'q' to finish.")
    while True:
        user_input = input("Repetition to edit (or 'q' to finish): ").strip().lower()
        if user_input == "q":
            break
        if not user_input.isdigit():
            print("Please enter a valid number or 'q'.")
            continue

        rep = int(user_input)
        if 1 <= rep <= len(peak_indices):
            try:
                new_idx = int(input(f"New global index for repetition {rep}: "))
                new_val = float(power_raw[new_idx])
                peak_indices[rep - 1] = new_idx
                max_powers_raw[rep - 1] = 2.0 * new_val
                print(f"Rep {rep} updated → idx {new_idx}, Raw Power = {2.0 * new_val:.4f} W")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid repetition number.")

    return max_powers_raw, peak_indices


def compute_max_velocity_per_rep(
    velocity_signal: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int]
) -> List[float]:
    """
    Maximum velocity per repetition.

    Returns
    List[float]
        Max velocity (same units as `velocity_signal`, typically m/s).
    """
    out: List[float] = []
    n = len(velocity_signal)
    for start, end in zip(starts, ends):
        s = max(0, int(start)); e = min(n - 1, int(end))
        seg = velocity_signal[s:e + 1]
        out.append(float(np.max(seg)) if seg.size else np.nan)
    return out


def compute_max_velocity_raw_at_filtered_peaks(
    velocity_filtered: np.ndarray,
    velocity_raw: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int]
) -> Tuple[List[float], List[int]]:
    """
    Find the peak index on filtered velocity, then read the amplitude from raw velocity.

    Returns
    max_velocities_raw : List[float]
        Peak raw velocity per rep (m/s).
    peak_indices : List[int]
        Global indices of those peaks.
    """
    max_velocities_raw: List[float] = []
    peak_indices: List[int] = []
    n = len(velocity_filtered)

    for start, end in zip(starts, ends):
        s = max(0, int(start)); e = min(n - 1, int(end))
        seg = velocity_filtered[s:e + 1]
        if seg.size == 0:
            max_velocities_raw.append(np.nan)
            peak_indices.append(-1)
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        max_velocities_raw.append(float(velocity_raw[peak_idx]))
        peak_indices.append(peak_idx)

    return max_velocities_raw, peak_indices


def compute_max_velocity_raw_at_filtered_peaks_interactive(
    velocity_filtered: np.ndarray,
    velocity_raw: np.ndarray,
    starts: Sequence[int],
    ends: Sequence[int]
) -> Tuple[List[float], List[int]]:
    """
    Same as compute_max_velocity_raw_at_filtered_peaks, but allows manual
    override of peak indices.
    """
    max_velocities_raw: List[float] = []
    peak_indices: List[int] = []

    print("\n--- Automatic Peak Detection (Velocity) ---")
    n = len(velocity_filtered)
    for i, (start, end) in enumerate(zip(starts, ends)):
        s = max(0, int(start)); e = min(n - 1, int(end))
        seg = velocity_filtered[s:e + 1]
        if seg.size == 0:
            max_velocities_raw.append(np.nan)
            peak_indices.append(-1)
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        val_raw = float(velocity_raw[peak_idx])
        max_velocities_raw.append(val_raw)
        peak_indices.append(peak_idx)
        print(f"Rep {i+1}: Peak @ {peak_idx} → Raw vel = {val_raw:.4f}")

    print("\n--- Manual Override ---")
    print("Type the repetition number to edit (1-based), or 'q' to finish.")
    while True:
        user_input = input("Repetition to edit (or 'q' to finish): ").strip().lower()
        if user_input == "q":
            break
        if not user_input.isdigit():
            print("Please enter a valid number or 'q'.")
            continue
        rep = int(user_input)
        if 1 <= rep <= len(peak_indices):
            try:
                new_idx = int(input(f"New global index for repetition {rep}: "))
                new_val = float(velocity_raw[new_idx])
                peak_indices[rep - 1] = new_idx
                max_velocities_raw[rep - 1] = new_val
                print(f"Rep {rep} updated → idx {new_idx}, Raw Vel = {new_val:.4f}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid repetition number.")
    return max_velocities_raw, peak_indices


def compute_max_rpd_from_smoothed_power(
    power_smoothed: np.ndarray,
    rep_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    fs: float,
    return_indices: bool = False
) -> Union[List[float], Tuple[List[float], List[int]]]:
    """
    Maximum Rate of Power Development (RPD) per repetition from *smoothed* power.

    RPD = d(power_smoothed)/dt (W/s). For each rep, return the maximum between
    repetition start and concentric end (inclusive).

    Parameters
    power_smoothed : np.ndarray
        Smoothed power (e.g., trend or low-pass).
    rep_starts_cut, concentric_ends_cut : Sequence[int]
        Start / concentric-end indices for each repetition (within the cut).
    fs : float
        Sampling frequency (Hz).
    return_indices : bool
        If True, also return the global index where max RPD occurs.

    Returns
    max_rpd_values : List[float]
        Max RPD (W/s) per repetition (NaN if segment invalid).
    max_rpd_indices : List[int], optional
        Global indices of the max RPD (or -1 if invalid).
    """
    rpd_smoothed = np.gradient(power_smoothed) * fs
    max_vals: List[float] = []
    max_idxs: List[int] = []
    N = len(rpd_smoothed)

    for start, end in zip(rep_starts_cut, concentric_ends_cut):
        s = max(0, int(start))
        e = min(N - 1, int(end))
        if e <= s:
            max_vals.append(np.nan); max_idxs.append(-1); continue

        seg = rpd_smoothed[s:e + 1]
        if seg.size == 0 or not np.isfinite(seg).any():
            max_vals.append(np.nan); max_idxs.append(-1); continue

        loc = int(np.nanargmax(seg))
        max_vals.append(float(seg[loc]))
        max_idxs.append(s + loc)

    return (max_vals, max_idxs) if return_indices else max_vals


def compute_power_slopes_to_peak(
    power_signal_cut: np.ndarray,
    concentric_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    time_cut: np.ndarray,
    fs: float,
    window_ms: float = 50.0
) -> Tuple[List[List[float]], List[float]]:
    """
    For each repetition, compute:
      • windowed slopes from concentric start to power peak (non-overlapping windows)
      • the overall slope from concentric start to power peak.

    Slopes are Δpower/Δtime in W/s.

    Parameters
    power_signal_cut : np.ndarray
        Filtered power for the cut segment.
    concentric_starts_cut, concentric_ends_cut : Sequence[int]
        Concentric start/end indices.
    time_cut : np.ndarray
        Time vector (s), aligned to `power_signal_cut`.
    fs : float
        Sampling frequency (Hz).
    window_ms : float
        Window size for local slopes (ms).

    Returns
    all_window_slopes : List[List[float]]
        Per-window slopes (W/s) for each repetition.
    total_slopes : List[float]
        Overall slope (W/s) from concentric start to power peak per repetition.
    """
    window_size = max(2, int(round((window_ms / 1000.0) * fs)))
    all_window_slopes: List[List[float]] = []
    total_slopes: List[float] = []

    n = len(power_signal_cut)
    for cs, ce in zip(concentric_starts_cut, concentric_ends_cut):
        cs = max(0, min(int(cs), n - 1))
        ce = max(0, min(int(ce), n - 1))
        if ce <= cs:
            all_window_slopes.append([])
            total_slopes.append(np.nan)
            continue

        seg_con = power_signal_cut[cs:ce + 1]
        if seg_con.size == 0:
            all_window_slopes.append([])
            total_slopes.append(np.nan)
            continue

        peak_rel = int(np.argmax(seg_con))
        peak_idx = cs + peak_rel

        segment = power_signal_cut[cs:peak_idx + 1]
        if segment.size < 2:
            all_window_slopes.append([])
            total_slopes.append(np.nan)
            continue

        # Non-overlapping windows inside [0, len(segment)-1]
        slopes: List[float] = []
        for start_idx in range(0, len(segment) - window_size + 1, window_size):
            end_idx = start_idx + window_size  # exclusive
            if end_idx > len(segment):
                break
            p_start = float(segment[start_idx])
            p_end   = float(segment[end_idx - 1])
            dt_samples = window_size - 1
            if dt_samples <= 0:
                continue
            slopes.append((p_end - p_start) * fs / dt_samples)  # W/s

        # Overall slope from concentric start to peak
        dt_total_samples = len(segment) - 1
        slope_total = (float(segment[-1]) - float(segment[0])) * fs / dt_total_samples if dt_total_samples > 0 else np.nan

        all_window_slopes.append(slopes)
        total_slopes.append(slope_total)

    return all_window_slopes, total_slopes


#EMG Metrics

def compute_emg_onset_to_power_peak_delay(
    estopt_onset_indices: Sequence[int],
    concentric_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    power_signal_cut: np.ndarray,
    fs: float
) -> List[float]:
    """
    Delay (s) from EMG onset (EstOpt) to power peak within the concentric phase.

    Parameters
    estopt_onset_indices : Sequence[int]
        EMG onset indices (one per repetition).
    concentric_starts_cut, concentric_ends_cut : Sequence[int]
        Concentric start/end indices (cut signals).
    power_signal_cut : np.ndarray
        Filtered power (cut segment).
    fs : float
        Sampling frequency (Hz) — use the timebase of `power_signal_cut`.

    Returns
    List[float]
        Delays in seconds. Negative if onset occurs after the power peak.
        NaN when onset is missing or indices invalid.
    """
    delays: List[float] = []
    n = len(power_signal_cut)

    for i, (cs, ce) in enumerate(zip(concentric_starts_cut, concentric_ends_cut)):
        if i >= len(estopt_onset_indices):
            delays.append(np.nan)
            continue

        cs = max(0, min(int(cs), n - 1))
        ce = max(0, min(int(ce), n - 1))
        onset_idx = int(estopt_onset_indices[i])

        if ce < cs or not (0 <= onset_idx < n):
            delays.append(np.nan)
            continue

        seg = power_signal_cut[cs:ce + 1]
        if seg.size == 0:
            delays.append(np.nan)
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = cs + peak_rel

        delays.append((peak_idx - onset_idx) / float(fs))

    return delays


def compute_emg_onset_to_peak_delay(
    estopt_onset_times: Sequence[float],
    envelope: np.ndarray,
    fs: float,
    window_ms: float = 60.0,
    decline_tol: float = 0.0,
    max_after_s: Optional[float] = None,
    peak_min_prominence: Optional[float] = None
) -> Tuple[List[float], List[List[float]], List[int]]:
    """
    Delay (s) from EMG onset (EstOpt) to EMG envelope peak with robustness checks.

    Peak selection:
      • Peaks are detected on the envelope (optionally with minimum prominence).
      • A candidate is accepted only if the average first difference in the
        60 ms window after the peak is ≤ `decline_tol` (i.e., the envelope
        shows a brief decay/plateau resolution).
      • Peaks must occur at/after the onset, and (optionally) within a limited
        window after it (e.g., ≤ 2 s).

    Parameters
    estopt_onset_times : Sequence[float]
        Onset times in seconds, **in the same time reference as `envelope`** (e.g., cut segment).
        Use NaN/None for missing onsets.
    envelope : np.ndarray
        Smoothed EMG envelope (same length as the cut EMG signal).
    fs : float
        Sampling frequency (Hz).
    window_ms : float
        Post-peak window size used to check decay (default 60 ms).
    decline_tol : float
        Tolerance on the mean of first differences in the post-peak window.
        Typical range: 0 to 1e-4.
    max_after_s : float, optional
        If provided, restricts peak search to onset ≤ t ≤ onset + max_after_s.
    peak_min_prominence : float, optional
        If provided, enforce a minimum prominence in `find_peaks`.

    Returns
    delays_s : List[float]
        Onset→peak delays (s), NaN if no valid peak found.
    peak_times_s : List[List[float]]
        Peak time(s) per onset (currently single-element lists or empty).
    emg_peak_indices : List[int]
        Peak indices per onset (or -1 if not found).
    """
    fs = float(fs)
    postN = max(1, int(round((window_ms / 1000.0) * fs)))

    # 1) Envelope peaks (optionally with prominence)
    if peak_min_prominence is not None:
        all_peaks, _ = find_peaks(envelope, prominence=peak_min_prominence)
    else:
        all_peaks, _ = find_peaks(envelope)

    # 2) Keep only peaks with small average rise in the post-window (i.e., decay)
    valid_peaks: List[int] = []
    for p in all_peaks:
        end = p + postN
        if end <= len(envelope):
            post = envelope[p:end]
            slope_mean = float(np.mean(np.diff(post)))
            if slope_mean <= float(decline_tol):
                valid_peaks.append(int(p))
    valid_peaks = np.asarray(valid_peaks, dtype=int)

    delays_s: List[float] = []
    peak_times_s: List[List[float]] = []
    emg_peak_indices: List[int] = []

    for onset_t in estopt_onset_times:
        if onset_t is None or (isinstance(onset_t, float) and np.isnan(onset_t)):
            delays_s.append(np.nan); peak_times_s.append([]); emg_peak_indices.append(-1)
            continue

        onset_t = float(onset_t)
        onset_idx = int(round(onset_t * fs))

        cand = valid_peaks[valid_peaks >= onset_idx]
        if max_after_s is not None:
            max_idx = onset_idx + int(round(max_after_s * fs))
            cand = cand[cand <= max_idx]

        if cand.size > 0:
            best_idx = int(cand[np.argmax(envelope[cand])])
            best_t = best_idx / fs
            delay_t = best_t - onset_t
            delays_s.append(delay_t if delay_t >= 0.0 else np.nan)
            peak_times_s.append([best_t])
            emg_peak_indices.append(best_idx)
        else:
            delays_s.append(np.nan); peak_times_s.append([]); emg_peak_indices.append(-1)

    return delays_s, peak_times_s, emg_peak_indices


def compute_emg_rise_slope(
    estopt_onset_times: Sequence[float],
    emg_peak_times: List[List[float]],
    envelope: np.ndarray,
    fs: float
) -> List[float]:
    """
    EMG rise slope between EstOpt onset and envelope peak.

    Slope = (Envelope[peak] − Envelope[onset]) / (t_peak − t_onset)

    Parameters
    estopt_onset_times : Sequence[float]
        Onset times (s) in the same time reference as `envelope`.
    emg_peak_times : List[List[float]]
        Peak times per repetition (s), as lists (empty if not found).
    envelope : np.ndarray
        Smoothed EMG envelope.
    fs : float
        Sampling frequency (Hz).

    Returns
    List[float]
        Slope per repetition in envelope-units per second (e.g., mV/s if
        envelope in mV). NaN when onset/peak invalid.
    """
    slopes: List[float] = []
    for onset_t, peak_list in zip(estopt_onset_times, emg_peak_times):
        if (onset_t is None) or (isinstance(onset_t, float) and np.isnan(onset_t)) or not peak_list:
            slopes.append(np.nan)
            continue

        peak_t = float(peak_list[0])
        onset_idx = int(round(float(onset_t) * fs))
        peak_idx  = int(round(peak_t * fs))

        if peak_idx <= onset_idx or peak_idx >= len(envelope) or onset_idx < 0:
            slopes.append(np.nan)
            continue

        delta_t = peak_t - float(onset_t)
        delta_amp = float(envelope[peak_idx]) - float(envelope[onset_idx])
        slopes.append(delta_amp / delta_t)

    return slopes


def compute_emg_median_frequency(
    emg_signal_cut: np.ndarray,
    estopt_onset_indices: Sequence[int],
    concentric_ends_cut: Sequence[int],
    fs: float
) -> List[float]:
    """
    EMG median frequency between EMG onset (EstOpt) and concentric end.

    The median frequency is where the cumulative PSD reaches 50% of total power.

    Parameters
    emg_signal_cut : np.ndarray
        Cut EMG signal (same length as the cut time vector).
    estopt_onset_indices : Sequence[int]
        Onset indices per repetition.
    concentric_ends_cut : Sequence[int]
        Concentric end indices per repetition.
    fs : float
        Sampling frequency (Hz).

    Returns
    List[float]
        Median frequency (Hz) per repetition; NaN if invalid segment.
    """
    median_freqs: List[float] = []
    n = len(emg_signal_cut)

    for onset_idx, end_idx in zip(estopt_onset_indices, concentric_ends_cut):
        onset_idx = int(onset_idx); end_idx = int(end_idx)
        if not (0 <= onset_idx < n) or not (0 <= end_idx <= n) or end_idx <= onset_idx:
            median_freqs.append(np.nan)
            continue

        segment = emg_signal_cut[onset_idx:end_idx]
        if segment.size < 4:
            median_freqs.append(np.nan)
            continue

        nperseg = min(int(fs) // 2, len(segment))
        if nperseg < 4:
            median_freqs.append(np.nan)
            continue

        freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
        if psd.size == 0:
            median_freqs.append(np.nan)
            continue

        cumulative_power = np.cumsum(psd)
        total_power = cumulative_power[-1]
        if total_power <= 0:
            median_freqs.append(np.nan)
            continue

        median_idx = int(np.searchsorted(cumulative_power, total_power / 2.0))
        median_idx = min(median_idx, len(freqs) - 1)
        median_freqs.append(float(freqs[median_idx]))

    return median_freqs