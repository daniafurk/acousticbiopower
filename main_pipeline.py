"""
Single-exercise processing pipeline for standardized published dataset.

Input: one header row, followed by eight numeric columns:
Nseq LOADCELL ACCX ACCY ACCZ EMGA EMGL ECG

The scientific signal processing, detection and metric calls are retained.
NaN/Inf rows are removed consistently across all channels by import_data.
"""
import logging
import os
import re

import numpy as np
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.seasonal import STL

from import_data import load_raw_data, _infer_exercise_from_filename
from plots import *
from processing import transfer_weight, transfer_acc, bandpass_filter, transfer_EMG
from config import *
from metrics import *
from detection import *
from analysis import *

logger = logging.getLogger(__name__)


def run_pipeline(file_path: str, output_dir: str,
                 save_plots: bool = True, show_plots: bool = False):
    """Return (per-repetition metrics dict, numeric relative load in %1RM)."""
    filename = os.path.basename(file_path)
    stem = os.path.splitext(filename)[0]
    # Published pattern: id3_30_mp_bp_abf.txt, id3_40_mp_squat_gm.txt
    load_match = re.search(r"_(\d+)_mp_(?:bp|squat)_(?:abf|gm)$", stem, re.IGNORECASE)
    if load_match is None:
        raise ValueError(f"Cannot read numeric relative load from filename: {filename}")
    percentage = int(load_match.group(1))
    final_output_dir = os.path.join(output_dir, str(percentage))
    if save_plots:
        os.makedirs(final_output_dir, exist_ok=True)
    logger.info("Processing file: %s (relative load %d%%)", filename, percentage)

    # Load once. Every channel shares the same post-NaN-removal sample indices.
    raw_data = load_raw_data(file_path)
    exercise = _infer_exercise_from_filename(filename)
    loadcell_signal = raw_data[:, 1]
    emg_signal = raw_data[:, 5 if exercise == "bp" else 6]
    x_data, y_data, z_data = raw_data[:, 2], raw_data[:, 3], raw_data[:, 4]

    # Calibration of ACC axes (→ m/s²) with a local calibration file
    calib_file = r"C:\Users\Dania\Documents\ASTROPOWER\Mestrados\Beatriz_Guerreiro\Dados_Beatriz\Acc_calib.txt"
    acc_x, acc_y, acc_z = transfer_acc(x_data, y_data, z_data, calib_file)
    force = transfer_weight(loadcell_signal)

    # Filtering accelerometer and force signals
    filtered_acc_bandpass = bandpass_filter(
        acc_z, ACC_BANDPASS_LOW, ACC_BANDPASS_HIGH, FS_ACC, order=BANDPASS_ORDER
    )
    filtered_acc_bandpass_gaussian = gaussian_filter1d(
        filtered_acc_bandpass, sigma=SIGMA_ACC_BAND_STRONG
    )
    filtered_loadcell_gaussian = gaussian_filter1d(force, sigma=SIGMA_LOADCELL)

    # Veloity Computation
    velocity_raw = compute_velocity(filtered_acc_bandpass_gaussian, FS_ACC)
    velocity_raw -= np.mean(velocity_raw)
    stl_v = STL(velocity_raw, period=300)
    velocity = stl_v.fit().trend
    # Time array
    time = np.arange(len(filtered_acc_bandpass_gaussian)) / FS_ACC

    # Computing Power
    power = compute_power(filtered_loadcell_gaussian, velocity_raw)
    stl_p = STL(power, period=300)
    power_filtered_2 = stl_p.fit().trend

    # EMG calibration to mV + detrend + bandpass
    original_signal = transfer_EMG(emg_signal)
    original_signal -= np.mean(original_signal)
    filtered_emg_bandpass = bandpass_filter(
        original_signal, low_cutoff=10, high_cutoff=499, fs=FS_EMG, order=4
    )

    # Concentric Phase Detection
    concentric_starts, concentric_ends, peak_powers, peak_idxs = detect_concentric_phases(
        velocity=velocity_raw, power=power_filtered_2, time=time, fs=FS_ACC,
    )
    rep_starts = detect_rep_starts_from_concentric(
        con_starts=concentric_starts, velocity=velocity_raw, fs=FS_ACC,
        max_inflection_velocity=0.03, min_eccentric_duration=0.60,
    )
    logger.info("Detected rep starts: %s", ", ".join(f"{time[i]:.2f}s" for i in rep_starts))
    logger.info("Detected conc. ends: %s", ", ".join(f"{time[i]:.2f}s" for i in concentric_ends))
    phase_pairs = get_phase_index_pairs(
        rep_starts=rep_starts, con_starts=concentric_starts, con_ends=concentric_ends
    )
    # Crop all signals per valid repetition (adds guard time before/after)
    crop_results = filter_and_crop_repetitions(
        acc_signal=filtered_acc_bandpass_gaussian,
        velocity=velocity,
        velocity_raw=velocity_raw,
        loadcell=filtered_loadcell_gaussian,
        power=power,
        power_filtered=power_filtered_2,
        time=time,
        rep_starts=rep_starts,
        ends=concentric_ends,
        concentric_starts=concentric_starts,
        concentric_ends=concentric_ends,
        fs=FS_ACC,
        min_peak_velocity=MIN_PEAK_VELOCITY,
        extra_before_s=EXTRA_TIME_BEFORE_START,
        extra_after_s=EXTRA_TIME_AFTER_END,
        emg_filtered=filtered_emg_bandpass,
        emg_original=original_signal,
    )
    # Unpack cropped arrays (all in the "cut" reference)
    acc_cut = crop_results["acc_cut"]
    velocity_cut = crop_results["velocity_cut"]
    velocity_raw_cut = crop_results["velocity_raw_cut"]
    filtered_loadcell_gaussian_cut = crop_results["filtered_loadcell_gaussian_cut"]
    power_filtered_cut = crop_results["power_filtered_cut"]
    power_cut = crop_results["power_cut"]
    time_cut = crop_results["time_cut"]
    starts_cut = crop_results["starts_cut"]
    ends_cut = crop_results["ends_cut"]
    concentric_starts_cut = crop_results["concentric_starts_cut"]
    concentric_ends_cut = crop_results["concentric_ends_cut"]
    emg_filtered_cut = crop_results["emg_filtered_cut"]
    emg_cut = crop_results["emg_cut"]
    logger.info("CUT concentric starts: %s", ", ".join(f"{time_cut[i]:.2f}s" for i in concentric_starts_cut))
    logger.info("CUT concentric ends: %s", ", ".join(f"{time_cut[i]:.2f}s" for i in concentric_ends_cut))
    logger.info("CUT rep starts: %s", ", ".join(f"{time_cut[i]:.2f}s" for i in starts_cut))
    logger.info("CUT rep ends: %s", ", ".join(f"{time_cut[i]:.2f}s" for i in ends_cut))

    # Plots: concentric-only and full phases (QA/diagnostics)
    plot_concentric_phase_signals_seasonal(
        time_cut=time_cut, acc_cut=acc_cut,
        filtered_loadcell_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_cut, velocity_raw_cut=velocity_raw_cut,
        peak_powers=peak_powers, power_cut=power_cut,
        power_filtered_cut=power_filtered_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "concentric_phases_signals.png"),
        save=save_plots, show=show_plots,
    )
    plot_full_phase_signals(
        time_cut=time_cut, acc_cut=acc_cut,
        filtered_loadcell_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_raw_cut, power_cut=power_filtered_cut,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "full_phases_signals.png"),
        save=save_plots, show=show_plots,
    )

    #  Power/velocity metrics (per-rep peaks, timing, correlations)
    signals = {
        "Velocity": {"data": velocity_cut, "color": "royalblue", "unit": "m/s"},
        "Power": {"data": power_filtered_cut, "color": "orchid", "unit": "W"},
    }
    peak_positions_dict = {}
    for signal_name, signal_info in signals.items():
        # Normalizing each repetition from o-100% of the Concentric Phase
        normalized_curves = normalize_segments(
            signal=signal_info["data"],
            concentric_starts_cut=concentric_starts_cut,
            concentric_ends_cut=concentric_ends_cut,
            num_points=100,
        )
        peak_percents = compute_peak_positions_in_percent(normalized_curves)
        peak_positions_dict[signal_name] = peak_percents
        # Plot normalized reps with peak markers
        time_percents = [
            np.linspace(0, 100, normalized_curves.shape[1])
            for _ in range(normalized_curves.shape[0])
        ]
        logger.info("Peak positions for %s: %s", signal_name,
                    ", ".join(f"rep{i+1}={float(p):.1f}%" for i, p in enumerate(peak_percents)))
        plot_normalized_signal_per_repetition(
            relative_time_percent=time_percents,
            normalized_curves=normalized_curves,
            peak_positions=peak_percents,
            signal_name=signal_name, unit=signal_info["unit"],
            color=signal_info["color"],
            save_path=os.path.join(final_output_dir, f"{signal_name.lower()}_normalized.png"),
            save=save_plots, show=show_plots,
        )

    # Per-rep max power and corresponding velocity-at-peak (raw)
    max_powers, peak_indices = compute_max_power_per_rep_from_filtered_index(
        power_filtered=power_filtered_cut, power_raw=power_cut,
        starts=concentric_starts_cut, ends=concentric_ends_cut,
    )
    max_velocities_raw, velocity_peak_indices = compute_max_velocity_raw_at_filtered_peaks(
        velocity_filtered=velocity_cut, velocity_raw=velocity_raw_cut,
        starts=concentric_starts_cut, ends=concentric_ends_cut,
    )

    # Velocity Power Difference in seconds
    dif_time_vel_power = [
        (velocity_peak_indices[i] - peak_indices[i]) / FS_ACC
        for i in range(len(velocity_peak_indices))
    ]
    for i, v in enumerate(max_velocities_raw):
        logger.info("Max velocity rep %d: %.2f m/s", i + 1, float(v))
    for i, p in enumerate(max_powers):
        logger.info("Max power rep %d: %.2f W", i + 1, float(p))
    if max_powers:
        logger.info("Highest max power across reps: %.2f W", float(max(max_powers)))
    plot_bar_max_power(
        max_powers=max_powers,
        save_path=os.path.join(final_output_dir, "bar_max_power.png"),
        save=save_plots, show=show_plots,
    )

    # Similarity across reps (correlation in 0–100% normalized domain)
    velocity_segments = normalize_segments(
        signal=velocity_cut, concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut, num_points=100,
    )
    velocity_corr_matrix = compute_correlation_matrix(velocity_segments)
    correlation_pairs = extract_correlation_pairs(velocity_corr_matrix)
    for a, b, c in correlation_pairs:
        logger.info("Velocity corr rep %s vs %s: %.3f", a, b, float(c))
    acc_segments = normalize_segments(
        signal=acc_cut, concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut, num_points=100,
    )
    acc_corr_matrix = compute_correlation_matrix(acc_segments)
    acc_correlation_pairs = extract_correlation_pairs(acc_corr_matrix)
    for a, b, c in acc_correlation_pairs:
        logger.info("ACC corr rep %s vs %s: %.3f", a, b, float(c))

    # Mean ± SD of power over full phases
    normalized_power_segments = normalize_segments(
        signal=power_filtered_cut, concentric_starts_cut=starts_cut,
        concentric_ends_cut=ends_cut, num_points=100,
    )
    mean_power_curve, std_power_curve = compute_mean_std_curve(normalized_power_segments)
    plot_mean_std_curve(
        mean=mean_power_curve, std=std_power_curve,
        title="Average Power Curve with Standard Deviation Band", ylabel="Power (W)",
        save_path=os.path.join(final_output_dir, "mean_std_power_curve.png"),
        save=save_plots, show=show_plots,
    )
    plot_each_normalized_segment(
        normalized_data=normalized_power_segments, ylabel="Power (W)",
        save_path=os.path.join(final_output_dir, "individual_normalized_power_curves.png"),
        save=save_plots, show=show_plots,
    )

    # Rate of power development  (indices local to `time_cut`)
    max_rpd_values = compute_max_rpd_from_smoothed_power(
        power_smoothed=power_filtered_cut, rep_starts_cut=starts_cut,
        concentric_ends_cut=concentric_ends_cut, fs=FS_ACC,
        return_indices=False,
    )
    # Total Power Slope and Window Power Slope
    all_window_slopes, total_slopes = compute_power_slopes_to_peak(
        power_signal_cut=power_filtered_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        time_cut=time_cut, fs=FS_ACC, window_ms=50.0,
    )
    plot_rpd_power(
        time_cut=time_cut, power_filtered_cut=power_filtered_cut,
        starts_cut=starts_cut, concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "rpd_power.png"),
        save=save_plots, show=show_plots,
    )
    plot_power_slopes_all_reps(
        all_window_slopes=all_window_slopes, total_slopes=total_slopes,
        window_ms=50.0, save_dir=final_output_dir,
        save=save_plots, show=show_plots,
    )

    # EMG metrics (EstOpt onsets → envelope peak time & slope, median freq)
    estopt_onset_indices, estopt_onset_times, envelope = detect_emg_onsets_estopt(
        emg_signal=emg_filtered_cut, fs=FS_EMG,
        concentric_starts=concentric_starts_cut, time_vector=time_cut,
    )
    logger.info("Detected %d EMG onsets.", len(estopt_onset_indices))

    # Onset → peak POWER delay (s), measured against power_filtered_cut
    emg_onset_to_power_peak_delays = compute_emg_onset_to_power_peak_delay(
        estopt_onset_indices=estopt_onset_indices,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        power_signal_cut=power_filtered_cut,
        fs=FS_EMG,
    )

    # Align time references: convert absolute → cut-local seconds
    t0_cut_s = float(time_cut[0])
    onsets_cut_s = [
        (float(t) - t0_cut_s)
        if (t is not None and not (isinstance(t, float) and np.isnan(t)))
        else np.nan
        for t in estopt_onset_times
    ]
    onsets_plot_s = [(oc + t0_cut_s) if not np.isnan(oc) else np.nan
                     for oc in onsets_cut_s]

    # Onset → peak ENVELOPE delay (s) and chosen peak index (samples, cut reference)
    emg_onset_to_peak_delays, emg_peak_times, emg_peak_indices = compute_emg_onset_to_peak_delay(
        estopt_onset_times=onsets_cut_s,
        envelope=envelope, fs=FS_EMG,
        window_ms=60.0, decline_tol=1e-4, max_after_s=2.0,
    )

    # Mean envelope rise slope (mV²/s)
    emg_rise_slopes = compute_emg_rise_slope(
        estopt_onset_times=onsets_cut_s,
        emg_peak_times=emg_peak_times, envelope=envelope, fs=FS_EMG,
    )

    # Median frequency (per onset → concentric end)
    median_freqs = compute_emg_median_frequency(
        emg_signal_cut=emg_filtered_cut,
        estopt_onset_indices=estopt_onset_indices,
        concentric_ends_cut=concentric_ends_cut, fs=FS_EMG,
    )
    # EMG visualization (absolute time reference)
    emg_peak_indices_wrapped = [[idx] if idx >= 0 else [] for idx in emg_peak_indices]
    plot_full_phase_signals_with_emg(
        time_cut=time_cut, acc_cut=acc_cut,
        filtered_loadcell_gaussian_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_raw_cut, power_filtered_cut=power_filtered_cut,
        emg_cut=emg_filtered_cut, starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "full_phases_with_emg.png"),
        save=save_plots, show=show_plots,
    )
    plot_emg_with_onsets_and_valid_peaks(
        time_cut=time_cut, emg_cut=emg_filtered_cut,
        envelope=envelope, estopt_onset_times=onsets_plot_s,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        emg_peak_indices=emg_peak_indices_wrapped,
        save_path=os.path.join(final_output_dir, "emg_with_onsets_and_peaks_new_.png"),
        save=save_plots, show=show_plots,
    )

    # Pack results and return
    results = {
        "max_powers": max_powers,
        "max_velocities": max_velocities_raw,
        "peak_positions": peak_positions_dict,
        "power_indices": peak_indices,
        "velocity_indices": velocity_peak_indices,
        "dif_vel_power": dif_time_vel_power,
        "correlation_pairs": {"Velocity": correlation_pairs,
                              "Acceleration": acc_correlation_pairs},
        "phase_indices": phase_pairs,
        "max_rpd": max_rpd_values,
        "wind_slope_power": all_window_slopes,
        "total_slope_power": total_slopes,
        "emg_onset_power_peak": emg_onset_to_power_peak_delays,
        "emg_rise_time": emg_onset_to_peak_delays,
        "emg_rise_slope": emg_rise_slopes,
        "median_freq": median_freqs,
    }
    return results, percentage
