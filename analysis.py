"""
Performs analysis of repetitions and segments.

Includes signal cropping, normalization of repetition segments, detection of movement inversion points,
correlation analysis between repetitions, and computation of average and variability across normalized curves.
"""

import numpy as np
import pandas as pd


def filter_and_crop_repetitions(acc_signal, velocity, loadcell, power, power_filtered, velocity_raw,time,
                                rep_starts, ends, concentric_starts, concentric_ends,
                                fs, min_peak_velocity, extra_before_s, extra_after_s, emg_filtered, emg_original):
    """
    Filters valid repetitions and crops signals with margins before/after.

    Args:
        acc_signal (np.ndarray): Filtered acceleration signal.
        velocity (np.ndarray): Velocity signal.
        loadcell (np.ndarray): Load cell signal (smoothed).
        power (np.ndarray): Power signal (filtered).
        time (np.ndarray): Time vector.
        rep_starts (List[int]): Indices of detected repetition starts.
        ends (List[int]): Indices of detected repetition ends.
        concentric_starts (List[int]): Indices of concentric phase starts.
        concentric_ends (List[int]): Indices of concentric phase ends.
        fs (int): Sampling rate (Hz).
        min_peak_velocity (float): Minimum velocity to accept repetition.
        extra_before_s (float): Time before first rep to include (sec).
        extra_after_s (float): Time after last rep to include (sec).

    Returns:
        dict: Dictionary containing cropped signals and updated indices.
    """
    extra_samples_before = int(extra_before_s * fs)
    extra_samples_after = int(extra_after_s * fs)

    valid_starts = []
    valid_ends = []

    for start, end in zip(rep_starts, ends):
        if end <= start:
            continue
        segment = velocity_raw[start:end]
        if len(segment) > 0 and np.max(np.abs(segment)) > min_peak_velocity:
            valid_starts.append(start)
            valid_ends.append(end)

    if not valid_starts:
        raise ValueError("No valid repetitions detected with current parameters.")

    first_valid_start = valid_starts[0]
    last_valid_end = valid_ends[-1]

    cut_start_index = max(0, first_valid_start - extra_samples_before)
    cut_end_index = min(last_valid_end + extra_samples_after, len(acc_signal) - 1)

    print(f"Valid starts: {valid_starts}")
    print(f"Valid ends: {valid_ends}")
    print(f"Cut start index: {cut_start_index}")
    print(f"Cut end index: {cut_end_index}")

    return {
        "acc_cut": acc_signal[cut_start_index:cut_end_index + 1],
        "velocity_cut": velocity[cut_start_index:cut_end_index + 1],
        "velocity_raw_cut": velocity_raw[cut_start_index:cut_end_index + 1],
        "filtered_loadcell_gaussian_cut": loadcell[cut_start_index:cut_end_index + 1],
        "power_cut": power[cut_start_index:cut_end_index + 1],
        "power_filtered_cut": power_filtered[cut_start_index:cut_end_index + 1],
        "time_cut": time[cut_start_index:cut_end_index + 1],
        "starts_cut": [s - cut_start_index for s in valid_starts if cut_start_index <= s <= cut_end_index],
        "ends_cut": [e - cut_start_index for e in valid_ends if cut_start_index <= e <= cut_end_index],
        "concentric_starts_cut": [s - cut_start_index for s in concentric_starts if
                                  cut_start_index <= s <= cut_end_index],
        "concentric_ends_cut": [e - cut_start_index for e in concentric_ends if
                                cut_start_index <= e <= cut_end_index],
        "emg_filtered_cut": emg_filtered[cut_start_index:cut_end_index + 1],
        "emg_cut": emg_original[cut_start_index:cut_end_index + 1]
    }


def normalize_segments(signal, concentric_starts_cut, concentric_ends_cut, num_points=100):
    """
    Normalizes segments to same length via interpolation.

    Args:
        signal (np.ndarray): Input signal (e.g., power, velocity).
        concentric_starts_cut(List[int]): Start indices.
        concentric_ends_cut(List[int]): End indices.
        num_points (int): Length to interpolate to.

    Returns:
        np.ndarray: Matrix (reps x num_points)
    """
    normalized = []

    for start, end in zip(concentric_starts_cut, concentric_ends_cut):
        segment = signal[start:end+1]
        interp = np.interp(np.linspace(0, len(segment)-1, num_points), np.arange(len(segment)), segment)
        normalized.append(interp)

    return np.array(normalized)


def compute_peak_positions_in_percent(normalized_curves: np.ndarray) -> list[float]:
    """
    Computes the time (in percent of cycle) at which the peak occurs in each normalized segment.

    Args:
        normalized_curves (np.ndarray): Matrix of normalized signals (reps x time).

    Returns:
        List[float]: Peak position in % for each repetition.
    """
    peak_positions = []

    for curve in normalized_curves:
        peak_idx = np.argmax(curve)
        peak_percent = (peak_idx / (len(curve) - 1)) * 100
        peak_positions.append(peak_percent)

    return peak_positions



def compute_correlation_matrix(normalized_data: np.ndarray) -> pd.DataFrame:
    """
    Computes correlation matrix between normalized curves.

    Args:
        normalized_data (np.ndarray): Matrix (reps x time).

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    df = pd.DataFrame(normalized_data.T, columns=[f"Rep {i+1}" for i in range(normalized_data.shape[0])])
    return df.corr()

def extract_correlation_pairs(corr_matrix: pd.DataFrame) -> list:
    """
    Extracts the unique correlation values between repetitions from a correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix between repetitions.

    Returns:
        List[Tuple[str, str, float]]: List of tuples with (Rep A, Rep B, correlation value).
    """
    correlation_pairs = []
    reps = corr_matrix.columns

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):  # skip diagonal and symmetric duplicates
            rep_i = reps[i]
            rep_j = reps[j]
            corr_value = corr_matrix.loc[rep_i, rep_j]
            correlation_pairs.append((rep_i, rep_j, corr_value))

    return correlation_pairs


def compute_mean_std_curve(normalized_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes average and standard deviation curves.

    Args:
        normalized_data (np.ndarray): Matrix (reps x time).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and std across reps.
    """
    return np.mean(normalized_data, axis=0), np.std(normalized_data, axis=0)






