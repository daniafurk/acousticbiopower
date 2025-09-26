"""
Batch runner and CLI for processing participant folders.

This module orchestrates:
  1) Batch runs over a Participants folder tree (participant → session → stimulus → Files/*.txt).
  2) Single-file processing for quick checks.
  3) CSV export of all aggregated per-repetition metrics (one CSV per participant/session/stimulus/exercise).

Notes:
- Uses 'main_pipeline.run_pipeline(...)' to compute all metrics and generate plots.
- Aggregates results into 'results_per_participant' (in-memory) and exports to CSV at the end.
- CSV separator is ';' to be Excel-friendly in locales where ',' is the decimal separator.
- Logging is used instead of print for better control of verbosity and destinations.

Folder structure expected:
  base_path/
    ID1/
      Session_1/
        ABF/
          Files/*.txt
          Plots/...
        GM/
          Files/*.txt
          Plots/...
    ID2/
      ...

Units follow what 'main_pipeline' emits:
- Velocity: m/s
- Power: W
- EMG envelope (TKEO-based): mV^2
- EMG rise slope: mV^2/s
- Times: s (seconds)
"""

import os
import logging
from typing import Dict, List
from collections import defaultdict
import numpy as np
import pandas as pd
import traceback
from main_pipeline import run_pipeline


# Configuration & global stores

# Base directory where participant/session/stimulus folders live
BASE_PATH = "C:/Users/beatr/Documents/UNIVERSIDADE/5ºANO MIEB/TESE/Recolhas/Participants"
STIMULI = ("ABF", "GM")

# Global stores (populated during runs)
participants_data: Dict[str, dict] = {}
results_per_participant: Dict[str, List[dict]] = {}

# Batch processing
def run_batch_processing(session_name: str = "Session_1",
                         base_path: str = BASE_PATH,
                         stimuli=STIMULI) -> None:
    """Process all *.txt files for all participants for a given session."""

    for participant_folder in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant_folder)
        if not os.path.isdir(participant_path):
            continue

        session_path = os.path.join(participant_path, session_name)
        if not os.path.exists(session_path):
            print(f"Skipping {participant_folder}: No {session_name} folder.")
            continue

        for stimulus in stimuli:
            files_dir = os.path.join(session_path, stimulus, "Files")
            plots_base_dir = os.path.join(session_path, stimulus, "Plots")

            if not os.path.exists(files_dir):
                print(f"Skipping {participant_folder} - {stimulus}: No Files directory.")
                continue

            txt_files = [f for f in os.listdir(files_dir) if f.endswith(".txt")]
            if not txt_files:
                print(f"No .txt files in {files_dir}")
                continue

            local_power_by_percentage = {
                "ABF": {"bench_press": defaultdict(list), "squat": defaultdict(list)},
                "GM": {"bench_press": defaultdict(list), "squat": defaultdict(list)},
            }

            for filename in txt_files:
                try:
                    exercise_type = "bench_press" if "bp" in filename.lower() else "squat"
                    file_path = os.path.join(files_dir, filename)
                    output_dir = os.path.join(plots_base_dir, exercise_type)

                    print(f"Processing {filename} ({participant_folder} | {session_name} | {stimulus} | {exercise_type})")

                    results, percentage = run_pipeline(file_path, output_dir)

                    if results is not None and percentage is not None and results.get("max_powers"):
                        max_power = max(results["max_powers"])
                        local_power_by_percentage[stimulus][exercise_type][percentage].append(max_power)

                    participants_data.setdefault(participant_folder, {}) \
                        .setdefault(session_name, {}) \
                        .setdefault(stimulus, {}) \
                        .setdefault(exercise_type, []).append(filename)

                    row = {
                        "participant_id": participant_folder,
                        "session": session_name,
                        "stimulus": stimulus,
                        "exercise": exercise_type,
                        "filename": filename,
                        "percentage_1RM": percentage,
                    }

                    for i, p in enumerate(results.get("max_powers", [])):
                        row[f"max_power_rep_{i + 1}"] = p
                    for i, v in enumerate(results.get("max_velocities", [])):
                        row[f"max_velocity_rep_{i + 1}"] = v
                    for i, idx in enumerate(results.get("power_indices", [])):
                        row[f"power_index_rep_{i + 1}"] = idx
                    for i, idx in enumerate(results.get("velocity_indices", [])):
                        row[f"velocity_index_rep_{i + 1}"] = idx
                    for i, dt in enumerate(results.get("dif_vel_power", [])):
                        row[f"dif_vel_power_rep_{i + 1}"] = dt

                    phase = results.get("phase_indices", {})
                    for i, (rep_start, con_end) in enumerate(phase.get("rep_con_pairs", [])):
                        row[f"rep_start_rep_{i + 1}"] = rep_start
                        row[f"con_end_rep_{i + 1}"] = con_end
                    for i, (con_start, con_end) in enumerate(phase.get("con_phase_pairs", [])):
                        row[f"con_start_rep_{i + 1}"] = con_start
                        row[f"con_end_rep_{i + 1}"] = con_end

                    for signal_name, peaks in results.get("peak_positions", {}).items():
                        for i, pos in enumerate(peaks):
                            row[f"peak_{signal_name.lower()}_rep_{i + 1}"] = pos

                    for signal_name, corr_pairs in results.get("correlation_pairs", {}).items():
                        for repA, repB, corr in corr_pairs:
                            row[f"corr_{signal_name.lower()}_{repA}_{repB}"] = corr

                    for i, r in enumerate(results.get("max_rpd", [])):
                        row[f"max_rpd_rep_{i + 1}"] = r

                    for i, s in enumerate(results.get("total_slope_power", [])):
                        row[f"power_total_slope_rep_{i + 1}"] = s
                    wind_slopes = results.get("wind_slope_power", [])
                    for i, slopes in enumerate(wind_slopes):
                        try:
                            mean_slope = float(np.nanmean(slopes)) if hasattr(slopes, "__len__") and len(slopes) > 0 else np.nan
                        except Exception:
                            mean_slope = np.nan
                        row[f"power_window_slope_mean_rep_{i + 1}"] = mean_slope

                    for i, val in enumerate(results.get("emg_onset_power_peak", [])):
                        row[f"emg_onset_to_power_peak_rep_{i + 1}"] = val
                    for i, val in enumerate(results.get("emg_rise_time", [])):
                        row[f"emg_rise_time_rep_{i + 1}"] = val
                    for i, val in enumerate(results.get("emg_rise_slope", [])):
                        row[f"emg_rise_slope_rep_{i + 1}"] = val
                    for i, val in enumerate(results.get("median_freq", [])):
                        row[f"emg_median_freq_rep_{i + 1}"] = val

                    results_per_participant.setdefault(participant_folder, []).append(row)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()


# Single-file processing
def process_single_file(participant_id: str,
                        session_name: str,
                        stimulus: str,
                        filename: str,
                        save_plots: bool = True,
                        show_plots: bool = False,
                        base_path: str = BASE_PATH) -> None:
    """Process a single file for a given participant/session/stimulus."""

    file_path = os.path.join(base_path, participant_id, session_name, stimulus, "Files", filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    exercise_type = "bench_press" if "bp" in filename.lower() else "squat"
    output_dir = os.path.join(base_path, participant_id, session_name, stimulus, "Plots", exercise_type)

    print(f"Processing single file: {filename}")
    try:
        results, percentage = run_pipeline(file_path, output_dir, save_plots=save_plots, show_plots=show_plots)
        if results is None or percentage is None:
            print(f"No valid results returned for {filename}")
            return

        participants_data.setdefault(participant_id, {}) \
            .setdefault(session_name, {}) \
            .setdefault(stimulus, {}) \
            .setdefault(exercise_type, []).append(filename)

        row = {
            "participant_id": participant_id,
            "session": session_name,
            "stimulus": stimulus,
            "exercise": exercise_type,
            "filename": filename,
            "percentage_1RM": percentage,
        }

        for i, p in enumerate(results.get("max_powers", [])):
            row[f"max_power_rep_{i + 1}"] = p
        for i, v in enumerate(results.get("max_velocities", [])):
            row[f"max_velocity_rep_{i + 1}"] = v
        for i, idx in enumerate(results.get("power_indices", [])):
            row[f"power_index_rep_{i + 1}"] = idx
        for i, idx in enumerate(results.get("velocity_indices", [])):
            row[f"velocity_index_rep_{i + 1}"] = idx
        for i, dt in enumerate(results.get("dif_vel_power", [])):
            row[f"dif_vel_power_rep_{i + 1}"] = dt

        phase = results.get("phase_indices", {})
        for i, (rep_start, con_end) in enumerate(phase.get("rep_con_pairs", [])):
            row[f"rep_start_rep_{i + 1}"] = rep_start
            row[f"con_end_rep_{i + 1}"] = con_end
        for i, (con_start, con_end) in enumerate(phase.get("con_phase_pairs", [])):
            row[f"con_start_rep_{i + 1}"] = con_start
            row[f"con_end_rep_{i + 1}"] = con_end

        for signal_name, peaks in results.get("peak_positions", {}).items():
            for i, pos in enumerate(peaks):
                row[f"peak_{signal_name.lower()}_rep_{i + 1}"] = pos

        for signal_name, corr_pairs in results.get("correlation_pairs", {}).items():
            for repA, repB, corr in corr_pairs:
                row[f"corr_{signal_name.lower()}_{repA}_{repB}"] = corr

        for i, r in enumerate(results.get("max_rpd", [])):
            row[f"max_rpd_rep_{i + 1}"] = r

        for i, s in enumerate(results.get("total_slope_power", [])):
            row[f"power_total_slope_rep_{i + 1}"] = s

        wind_slopes = results.get("wind_slope_power", [])
        for i, slopes in enumerate(wind_slopes):
            try:
                mean_slope = float(np.nanmean(slopes)) if hasattr(slopes, "__len__") and len(slopes) > 0 else np.nan
            except Exception:
                mean_slope = np.nan
            row[f"power_window_slope_mean_rep_{i + 1}"] = mean_slope

        for i, val in enumerate(results.get("emg_onset_power_peak", [])):
            row[f"emg_onset_to_power_peak_rep_{i + 1}"] = val
        for i, val in enumerate(results.get("emg_rise_time", [])):
            row[f"emg_rise_time_rep_{i + 1}"] = val
        for i, val in enumerate(results.get("emg_rise_slope", [])):
            row[f"emg_rise_slope_rep_{i + 1}"] = val
        for i, val in enumerate(results.get("median_freq", [])):
            row[f"emg_median_freq_rep_{i + 1}"] = val

        results_per_participant.setdefault(participant_id, []).append(row)
        print(f"Finished processing {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        traceback.print_exc()


# CSV export
def export_results_to_csv(base_path: str = BASE_PATH,
                          sep: str = ';') -> None:
    """Export one CSV per participant/session/stimulus/exercise, Excel-friendly."""

    for participant_id, rows in results_per_participant.items():
        df = pd.DataFrame(rows)
        grouped = df.groupby(["session", "stimulus", "exercise"], dropna=False)

        for (session, stimulus, exercise), group_df in grouped:
            save_dir = os.path.join(base_path, participant_id, session, stimulus)
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{participant_id}_{session}_{stimulus}_{exercise}_results.csv"
            save_path = os.path.join(save_dir, filename)

            group_df.to_csv(save_path,
                            mode="a",
                            header=not os.path.exists(save_path),
                            index=False,
                            sep=sep,
                            encoding="utf-8")
            print(f"Saved CSV: {save_path}")