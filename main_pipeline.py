"""
Main processing pipeline for a single exercise file.

This script loads synchronized ACC, load-cell and EMG signals from a text file,
applies calibration and filtering, detects repetitions and concentric phases,
crops signals per valid reps, computes kinematic (velocity), mechanical (power),
and neuromuscular (EMG) metrics, and saves diagnostic plots. It returns a
'results' dictionary (for CSV export) and the %1RM extracted from the filename.

Assumptions and units:
- Sampling rates come from 'config.py': FS_ACC (Hz) and FS_EMG (Hz).
- Accelerometer (Z-axis) is calibrated to m/s² (transfer_acc).
- Load-cell is converted via `transfer_weight` (project-specific): treated as force (N)
  or mass (kg) depending on your calibration; `compute_power` expects compatible inputs.
- Velocity is derived from acceleration and optionally detrended via STL.
- Power is computed from load-cell and velocity; an STL trend is used as smoothed power.
- EMG is converted to mV (transfer_EMG), detrended and bandpass filtered (10–499 Hz);
  the EMG envelope used for rise time/slope is TKEO-based (units mV²).

Side effects:
- Creates {output_dir}/{percentage}/ with multiple PNG plots for QA.
- Prints progress and detected events to the console.

Returns:
- results: dict with per-repetition metrics ready for CSV assembly.
- percentage: int or None, parsed from the input filename (e.g., “…_30_…” → 30).
"""

from import_data import *
from plots import *
from processing import transfer_weight, transfer_acc, bandpass_filter, transfer_EMG
from config import *
from metrics import *
from detection import *
from analysis import *
from scipy.ndimage import gaussian_filter1d
import os
import re
import numpy as np
from statsmodels.tsa.seasonal import STL


def run_pipeline(
    file_path: str,
    output_dir: str,
    save_plots: bool = True,
    show_plots: bool = False,
):
    """Run the full processing and analysis pipeline for one file."""
    # 1) Resolve filename → % folder and create output directory

    filename = os.path.basename(file_path)
    filename_prefix = os.path.splitext(filename)[0]

    # Extract the percentage number (e.g., "_30_") → "30"
    percentage_match = re.search(r"_(\d+)_", filename_prefix)
    percentage_folder = percentage_match.group(1) if percentage_match else "unknown"

    # Final output dir for this file's plots (grouped by %)
    final_output_dir = os.path.join(output_dir, percentage_folder)
    os.makedirs(final_output_dir, exist_ok=True)

    print(f"Processing file: {filename} (% folder: {percentage_folder})")

    # 2) Load raw signals (ACC Z, load-cell, EMG) + full raw for XYZ ACC

    acc_signal, loadcell_signal, emg_signal = load_exercise_data(file_path)

    # Load the whole row set (to get all ACC axes)
    raw_data = np.loadtxt(file_path, skiprows=1)  # skip header
    x_data = raw_data[:, 2]  # ACCx (raw units)
    y_data = raw_data[:, 3]  # ACCy
    z_data = raw_data[:, 4]  # ACCz

    # Calibration of ACC axes (→ m/s²) with a local calibration file
    calib_file = r"C:\Users\beatr\Documents\UNIVERSIDADE\5ºANO MIEB\TESE\Recolhas\Acc_calib_new.txt"
    acc_x, acc_y, acc_z = transfer_acc(x_data, y_data, z_data, calib_file)

    # Convert load-cell to calibrated quantity (project-specific).
    force = transfer_weight(loadcell_signal)

    # 3) Filtering and smoothing (ACC Z, load-cell)

    filtered_acc_bandpass = bandpass_filter(
        acc_z, ACC_BANDPASS_LOW, ACC_BANDPASS_HIGH, FS_ACC, order=BANDPASS_ORDER
    )

    # Lightly smooth signals for robust phase detection & power calc
    filtered_acc_bandpass_gaussian = gaussian_filter1d(
        filtered_acc_bandpass, sigma=SIGMA_ACC_BAND_STRONG
    )
    filtered_loadcell_gaussian = gaussian_filter1d(force, sigma=SIGMA_LOADCELL)

    # 4) Velocity derivation (from ACC Z) and STL trend extraction
    velocity_raw = compute_velocity(filtered_acc_bandpass_gaussian, FS_ACC)
    velocity_raw -= np.mean(velocity_raw)  # remove DC drift prior to STL
    stl_v = STL(velocity_raw, period=300)
    velocity = stl_v.fit().trend  # smoothed/trend velocity for plotting

    # Common time vector (s) for ACC/vel/power signals
    time = np.arange(len(filtered_acc_bandpass_gaussian)) / FS_ACC

    # 5) Power computation (from load-cell & velocity) + STL smoothing
    power = compute_power(filtered_loadcell_gaussian, velocity_raw)
    stl_p = STL(power, period=300)
    power_filtered_2 = stl_p.fit().trend  # smoothed/trend power

    # 6) EMG calibration to mV + detrend + bandpass
    original_signal = transfer_EMG(emg_signal)  # → mV
    original_signal -= np.mean(original_signal)  # detrend (remove DC)
    filtered_emg_bandpass = bandpass_filter(
        original_signal, low_cutoff=10, high_cutoff=499, fs=FS_EMG, order=4
    )

    # 7) Phase concentric detection
    concentric_starts, concentric_ends, peak_powers, peak_idxs = detect_concentric_phases(
        velocity=velocity_raw,
        power=power_filtered_2,
        time=time,
        fs=FS_ACC,
    )

    # Rep starts (eccentric-start → concentric-start) using velocity shape
    rep_starts = detect_rep_starts_from_concentric(
        con_starts=concentric_starts,
        velocity=velocity_raw,
        fs=FS_ACC,
        max_inflection_velocity=0.03,
        min_eccentric_duration=0.60,
    )

    print("Detected rep starts: " + ", ".join(f"{time[i]:.2f}s" for i in rep_starts))
    print("Detected conc. ends: " + ", ".join(f"{time[i]:.2f}s" for i in concentric_ends))

    phase_pairs = get_phase_index_pairs(
        rep_starts=rep_starts, con_starts=concentric_starts, con_ends=concentric_ends
    )

    # 8) Crop all signals per valid repetition (adds guard time before/after)
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

    print("CUT concentric starts: " + ", ".join(f"{time_cut[i]:.2f}s" for i in concentric_starts_cut))
    print("CUT concentric ends:   " + ", ".join(f"{time_cut[i]:.2f}s" for i in concentric_ends_cut))
    print("CUT rep starts:        " + ", ".join(f"{time_cut[i]:.2f}s" for i in starts_cut))
    print("CUT rep ends:          " + ", ".join(f"{time_cut[i]:.2f}s" for i in ends_cut))

    # 9) Plots: concentric-only and full phases (QA/diagnostics)
    plot_concentric_phase_signals_seasonal(
        time_cut=time_cut,
        acc_cut=acc_cut,
        filtered_loadcell_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_cut,
        velocity_raw_cut=velocity_raw_cut,
        peak_powers=peak_powers,
        power_cut=power_cut,
        power_filtered_cut=power_filtered_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "concentric_phases_signals.png"),
        save=save_plots,
        show=show_plots,
    )

    plot_full_phase_signals(
        time_cut=time_cut,
        acc_cut=acc_cut,
        filtered_loadcell_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_raw_cut,
        power_cut=power_filtered_cut,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "full_phases_signals.png"),
        save=save_plots,
        show=show_plots,
    )

    # 10) Power/velocity metrics (per-rep peaks, timing, correlations)
    signals = {
        "Velocity": {"data": velocity_cut, "color": "royalblue", "unit": "m/s"},
        "Power": {"data": power_filtered_cut, "color": "orchid", "unit": "W"},
    }
    peak_positions_dict = {}

    for signal_name, signal_info in signals.items():
        # Normalize each rep to 0..100% of the concentric phase
        normalized_curves = normalize_segments(
            signal=signal_info["data"],
            concentric_starts_cut=concentric_starts_cut,
            concentric_ends_cut=concentric_ends_cut,
            num_points=100,
        )

        # Peak position as % of the normalized cycle
        peak_percents = compute_peak_positions_in_percent(normalized_curves)
        peak_positions_dict[signal_name] = peak_percents

        # Plot normalized reps with peak markers
        time_percents = [
            np.linspace(0, 100, normalized_curves.shape[1])
            for _ in range(normalized_curves.shape[0])
        ]
        print(
            "Peak positions for "
            + signal_name
            + ": "
            + ", ".join(f"rep{i+1}={p:.1f}%" for i, p in enumerate(peak_percents))
        )
        plot_normalized_signal_per_repetition(
            relative_time_percent=time_percents,
            normalized_curves=normalized_curves,
            peak_positions=peak_percents,
            signal_name=signal_name,
            unit=signal_info["unit"],
            color=signal_info["color"],
            save_path=os.path.join(final_output_dir, f"{signal_name.lower()}_normalized.png"),
            save=save_plots,
            show=show_plots,
        )

    # Per-rep max power and corresponding velocity-at-peak (raw)
    max_powers, peak_indices = compute_max_power_per_rep_from_filtered_index(
        power_filtered=power_filtered_cut,
        power_raw=power_cut,
        starts=concentric_starts_cut,
        ends=concentric_ends_cut,
    )
    max_velocities_raw, velocity_peak_indices = compute_max_velocity_raw_at_filtered_peaks(
        velocity_filtered=velocity_cut,
        velocity_raw=velocity_raw_cut,
        starts=concentric_starts_cut,
        ends=concentric_ends_cut,
    )

    # Velocity Power Difference in seconds
    dif_time_vel_power = [
        (velocity_peak_indices[i] - peak_indices[i]) / FS_ACC
        for i in range(len(velocity_peak_indices))
    ]

    for i, v in enumerate(max_velocities_raw):
        print(f"Max velocity rep {i + 1}: {v:.2f} m/s")
    for i, p in enumerate(max_powers):
        print(f"Max power    rep {i + 1}: {p:.2f} W")
    if max_powers:
        print(f"Highest max power across reps: {max(max_powers):.2f} W")

    plot_bar_max_power(
        max_powers=max_powers,
        save_path=os.path.join(final_output_dir, "bar_max_power.png"),
        save=save_plots,
        show=show_plots,
    )

    # Similarity across reps (correlation in 0–100% normalized domain)
    velocity_segments = normalize_segments(
        signal=velocity_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        num_points=100,
    )
    velocity_corr_matrix = compute_correlation_matrix(velocity_segments)
    correlation_pairs = extract_correlation_pairs(velocity_corr_matrix)
    for a, b, c in correlation_pairs:
        print(f"Velocity corr rep {a} vs {b}: {c:.3f}")

    acc_segments = normalize_segments(
        signal=acc_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        num_points=100,
    )
    acc_corr_matrix = compute_correlation_matrix(acc_segments)
    acc_correlation_pairs = extract_correlation_pairs(acc_corr_matrix)
    for a, b, c in acc_correlation_pairs:
        print(f"ACC corr      rep {a} vs {b}: {c:.3f}")

    # Mean ± SD of power over full phases (note: uses starts_cut/ends_cut intentionally)
    normalized_power_segments = normalize_segments(
        signal=power_filtered_cut,
        concentric_starts_cut=starts_cut,
        concentric_ends_cut=ends_cut,
        num_points=100,
    )
    mean_power_curve, std_power_curve = compute_mean_std_curve(normalized_power_segments)
    plot_mean_std_curve(
        mean=mean_power_curve,
        std=std_power_curve,
        title="Average Power Curve with Standard Deviation Band",
        ylabel="Power (W)",
        save_path=os.path.join(final_output_dir, "mean_std_power_curve.png"),
        save=save_plots,
        show=show_plots,
    )
    plot_each_normalized_segment(
        normalized_data=normalized_power_segments,
        ylabel="Power (W)",
        save_path=os.path.join(final_output_dir, "individual_normalized_power_curves.png"),
        save=save_plots,
        show=show_plots,
    )

    # Rate of power development  (indices local to `time_cut`)
    max_rpd_values = compute_max_rpd_from_smoothed_power(
        power_smoothed=power_filtered_cut,
        rep_starts_cut=starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        fs=FS_ACC,
        return_indices=False,
    )
    # Total Power Slope and Window Power Slope
    all_window_slopes, total_slopes = compute_power_slopes_to_peak(
        power_signal_cut=power_filtered_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        time_cut=time_cut,
        fs=FS_ACC,
        window_ms=50.0,
    )
    plot_rpd_power(
        time_cut=time_cut,
        power_filtered_cut=power_filtered_cut,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "rpd_power.png"),
        save=save_plots,
        show=show_plots,
    )
    plot_power_slopes_all_reps(
        all_window_slopes=all_window_slopes,
        total_slopes=total_slopes,
        window_ms=50.0,
        save_dir=final_output_dir,
        save=save_plots,
        show=show_plots,
    )

    # 11) EMG metrics (EstOpt onsets → envelope peak time & slope, median freq)
    estopt_onset_indices, estopt_onset_times, envelope = detect_emg_onsets_estopt(
        emg_signal=emg_filtered_cut,  # EMG in mV (cut segment)
        fs=FS_EMG,
        concentric_starts=concentric_starts_cut,
        time_vector=time_cut,
    )
    print(f"Detected {len(estopt_onset_indices)} EMG onsets.")

    # Onset → peak POWER delay (s), measured against power_filtered_cut
    emg_onset_to_power_peak_delays = compute_emg_onset_to_power_peak_delay(
        estopt_onset_indices=estopt_onset_indices,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        power_signal_cut=power_filtered_cut,
        fs=FS_EMG,  # FS_EMG or FS_ACC depending on how indices were estimated
    )

    # Align time references: convert absolute → cut-local seconds
    t0_cut_s = float(time_cut[0])
    onsets_cut_s = [
        (float(t) - t0_cut_s)
        if (t is not None and not (isinstance(t, float) and np.isnan(t)))
        else np.nan
        for t in estopt_onset_times
    ]
    onsets_plot_s = [(oc + t0_cut_s) if not np.isnan(oc) else np.nan for oc in onsets_cut_s]

    # Onset → peak ENVELOPE delay (s) and chosen peak index (samples, cut reference)
    emg_onset_to_peak_delays, emg_peak_times, emg_peak_indices = compute_emg_onset_to_peak_delay(
        estopt_onset_times=onsets_cut_s,  # seconds in CUT reference
        envelope=envelope,                # TKEO-based envelope (mV²)
        fs=FS_EMG,
        window_ms=60.0,
        decline_tol=1e-4,
        max_after_s=2.0,
    )

    # Mean envelope rise slope (mV²/s)
    emg_rise_slopes = compute_emg_rise_slope(
        estopt_onset_times=onsets_cut_s,
        emg_peak_times=emg_peak_times,
        envelope=envelope,
        fs=FS_EMG,
    )

    # Median frequency (per onset → concentric end)
    median_freqs = compute_emg_median_frequency(
        emg_signal_cut=emg_filtered_cut,
        estopt_onset_indices=estopt_onset_indices,
        concentric_ends_cut=concentric_ends_cut,
        fs=FS_EMG,
    )

    # EMG visualization (absolute time reference)
    emg_peak_indices_wrapped = [[idx] if idx >= 0 else [] for idx in emg_peak_indices]
    plot_full_phase_signals_with_emg(
        time_cut=time_cut,
        acc_cut=acc_cut,
        filtered_loadcell_gaussian_cut=filtered_loadcell_gaussian_cut,
        velocity_cut=velocity_raw_cut,
        power_filtered_cut=power_filtered_cut,
        emg_cut=emg_filtered_cut,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        save_path=os.path.join(final_output_dir, "full_phases_with_emg.png"),
        save=save_plots,
        show=show_plots,
    )
    plot_emg_with_onsets_and_valid_peaks(
        time_cut=time_cut,
        emg_cut=emg_filtered_cut,
        envelope=envelope,
        estopt_onset_times=onsets_plot_s,
        starts_cut=starts_cut,
        concentric_starts_cut=concentric_starts_cut,
        concentric_ends_cut=concentric_ends_cut,
        emg_peak_indices=emg_peak_indices_wrapped,
        save_path=os.path.join(final_output_dir, "emg_with_onsets_and_peaks_new_.png"),
        save=save_plots,
        show=show_plots,
    )

    # 12) Pack results and return
    try:
        percentage = int(percentage_folder)
    except ValueError:
        percentage = None

    results = {
        "max_powers": max_powers,
        "max_velocities": max_velocities_raw,
        "peak_positions": peak_positions_dict,
        "power_indices": peak_indices,
        "velocity_indices": velocity_peak_indices,
        "dif_vel_power": dif_time_vel_power,
        "correlation_pairs": {"Velocity": correlation_pairs, "Acceleration": acc_correlation_pairs},
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