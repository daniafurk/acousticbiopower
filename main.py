"""
Batch and single-file runner for the published ASTROPOWER dataset.
Expected tree: Dataset/ID{1,...,24}/Session_{1,2,3}}/{ABF,GM}/idx_relativeload_mp_{bp,squat}_{abf,gm}.txt
Notes:
- Uses 'main_pipeline.run_pipeline(...)' to compute all metrics and generate plots.
- Aggregates results into 'results_per_participant' (in-memory) and exports to CSV at the end.
- CSV separator is ';' to be Excel-friendly in locales where ',' is the decimal separator.
- Logging is used for progress/warnings/errors; prints are only used  for interactive CLI .
Analysis products are written to OUTPUT_ROOT, not into Dataset.



"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from main_pipeline import run_pipeline

# Adjust only these two paths for your machine.
BASE_PATH = Path(r"C:\Users\Dania\Documents\ASTROPOWER\Mestrados\Beatriz_Guerreiro\Dados_Beatriz")
OUTPUT_ROOT = BASE_PATH.parent / "Dataset_Results"
STIMULI = ("ABF", "GM")

logger = logging.getLogger(__name__)
participants_data: Dict[str, dict] = {}
results_per_participant: Dict[str, List[dict]] = {}

# Example: id3_30_mp_squat_gm.txt (participant identifiers may contain hyphens).
FILE_RE = re.compile(
    r"^(?P<id>.+)_(?P<load>\d+)_mp_(?P<exercise>bp|squat)_(?P<stimulus>abf|gm)\.txt$",
    flags=re.IGNORECASE,
)


def parse_exercise_filename(filename: str, participant_id: str, stimulus: str):
    """Return (exercise_name, relative_load) if filename belongs to folder; else None."""
    match = FILE_RE.fullmatch(filename)
    if not match:
        return None
    if match.group("id").casefold() != participant_id.casefold():
        logger.warning("Filename participant ID does not match folder: %s", filename)
        return None
    if match.group("stimulus").casefold() != stimulus.casefold():
        logger.warning("Filename condition does not match folder: %s", filename)
        return None
    exercise = "bench_press" if match.group("exercise").casefold() == "bp" else "squat"
    return exercise, int(match.group("load"))


def iter_exercise_files(condition_path: Path, participant_id: str, stimulus: str):
    """Yield only recognized exercise text files; skip other session measurements."""
    if not condition_path.is_dir():
        logger.warning("Condition directory missing: %s", condition_path)
        return
    for path in sorted(condition_path.iterdir()):
        if not path.is_file() or path.suffix.casefold() != ".txt":
            continue
        parsed = parse_exercise_filename(path.name, participant_id, stimulus)
        if parsed:
            yield path, *parsed
        else:
            logger.info("Skipping unrecognized exercise filename: %s", path.name)


def build_result_row(results: dict, participant_id: str, session_name: str,
                     stimulus: str, exercise_type: str, filename: str,
                     relative_load: int) -> dict:
    """Preserve the original CSV metrics and column names for compatibility."""
    row = dict(participant_id=participant_id, session=session_name,
               stimulus=stimulus, exercise=exercise_type, filename=filename,
               percentage_1RM=relative_load)

    metrics = {
        "max_powers": "max_power",
        "max_velocities": "max_velocity",
        "power_indices": "power_index",
        "velocity_indices": "velocity_index",
        "dif_vel_power": "dif_vel_power",
        "max_rpd": "max_rpd",
        "total_slope_power": "power_total_slope",
        "emg_onset_power_peak": "emg_onset_to_power_peak",
        "emg_rise_time": "emg_rise_time",
        "emg_rise_slope": "emg_rise_slope",
        "median_freq": "emg_median_freq",
    }
    for source_key, column_prefix in metrics.items():
        for i, value in enumerate(results.get(source_key, []) or [], start=1):
            row[f"{column_prefix}_rep_{i}"] = value

    phase = results.get("phase_indices", {}) or {}
    for i, (rep_start, con_end) in enumerate(phase.get("rep_con_pairs", []), start=1):
        row[f"rep_start_rep_{i}"] = rep_start
        row[f"con_end_rep_{i}"] = con_end
    for i, (con_start, con_end) in enumerate(phase.get("con_phase_pairs", []), start=1):
        row[f"con_start_rep_{i}"] = con_start
        row[f"con_end_rep_{i}"] = con_end

    for signal_name, peaks in (results.get("peak_positions", {}) or {}).items():
        for i, pos in enumerate(peaks, start=1):
            row[f"peak_{signal_name.lower()}_rep_{i}"] = pos

    for signal_name, pairs in (results.get("correlation_pairs", {}) or {}).items():
        for rep_a, rep_b, corr in pairs:
            row[f"corr_{signal_name.lower()}_{rep_a}_{rep_b}"] = corr

    for i, slopes in enumerate(results.get("wind_slope_power", []) or [], start=1):
        values = np.asarray(slopes, dtype=float) if slopes is not None else np.array([])
        row[f"power_window_slope_mean_rep_{i}"] = (
            float(np.nanmean(values)) if values.size and np.isfinite(values).any()
            else np.nan
        )
    return row


def process_exercise_file(file_path: Path, participant_id: str, session_name: str,
                          stimulus: str, exercise_type: str, expected_load: int,
                          save_plots: bool = True, show_plots: bool = False,
                          output_root: Path = OUTPUT_ROOT) -> bool:
    plots_dir = Path(output_root) / participant_id / session_name / stimulus / "Plots" / exercise_type
    try:
        logger.info("Processing %s", file_path)
        results, pipeline_load = run_pipeline(
            str(file_path), str(plots_dir), save_plots=save_plots,
            show_plots=show_plots,
        )
        if results is None:
            logger.warning("No results for %s", file_path)
            return False
        if pipeline_load != expected_load:
            raise ValueError(
                f"Pipeline load ({pipeline_load}) differs from filename load "
                f"({expected_load}) for {file_path.name}. Update main_pipeline.py "
                f"as instructed."
            )
        participants_data.setdefault(participant_id, {}).setdefault(session_name, {}) \
            .setdefault(stimulus, {}).setdefault(exercise_type, []).append(file_path.name)
        row = build_result_row(results, participant_id, session_name, stimulus,
                               exercise_type, file_path.name, expected_load)
        results_per_participant.setdefault(participant_id, []).append(row)
        return True
    except Exception:
        logger.exception("Error processing %s", file_path)
        return False


def run_batch_processing(session_name: str = "Session_1",
                         base_path: str | Path = BASE_PATH,
                         stimuli=STIMULI,
                         output_root: str | Path = OUTPUT_ROOT,
                         save_plots: bool = True,
                         show_plots: bool = False) -> None:
    root = Path(base_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for participant_dir in sorted(root.iterdir()):
        if not participant_dir.is_dir():
            continue
        session_dir = participant_dir / session_name
        if not session_dir.is_dir():
            logger.warning("Skipping %s: no %s", participant_dir.name, session_name)
            continue
        for stimulus in stimuli:
            condition_dir = session_dir / stimulus
            for file_path, exercise, load in iter_exercise_files(
                condition_dir, participant_dir.name, stimulus
            ):
                process_exercise_file(file_path, participant_dir.name, session_name,
                                      stimulus, exercise, load, save_plots,
                                      show_plots, Path(output_root))


def process_single_file(participant_id: str, session_name: str, stimulus: str,
                        filename: str, save_plots: bool = True,
                        show_plots: bool = False,
                        base_path: str | Path = BASE_PATH,
                        output_root: str | Path = OUTPUT_ROOT) -> None:
    stimulus = stimulus.upper()
    if stimulus not in STIMULI:
        raise ValueError(f"Unrecognized condition: {stimulus}")
    parsed = parse_exercise_filename(filename, participant_id, stimulus)
    if parsed is None:
        raise ValueError(f"Unrecognized filename: {filename}")
    file_path = Path(base_path) / participant_id / session_name / stimulus / filename
    if not file_path.is_file():
        logger.error("File not found: %s", file_path)
        return
    exercise, load = parsed
    process_exercise_file(file_path, participant_id, session_name, stimulus,
                          exercise, load, save_plots, show_plots, Path(output_root))


def export_results_to_csv(output_root: str | Path = OUTPUT_ROOT,
                          sep: str = ";") -> None:
    """Write one CSV per participant/session/condition/exercise; overwrite on export."""
    for participant_id, rows in results_per_participant.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for (session, stimulus, exercise), group in df.groupby(
            ["session", "stimulus", "exercise"], dropna=False
        ):
            save_dir = Path(output_root) / participant_id / session / stimulus
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / f"{participant_id}_{session}_{stimulus}_{exercise}_results.csv"
            # Prevent duplicated records when rerunning within the same process.
            group = group.drop_duplicates(subset=["filename"], keep="last")
            group.sort_values(["percentage_1RM", "filename"]).to_csv(
                file_path, sep=sep, index=False, encoding="utf-8-sig"
            )
            logger.info("Saved CSV: %s", file_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    print(f"Dataset: {BASE_PATH}\nResults: {OUTPUT_ROOT}")
    print("1 - Batch process a session (all participants)")
    print("2 - Process selected files")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        session = input("Session (e.g. Session_1, Session_2, Session_3): ").strip()
        run_batch_processing(session_name=session)
        export_results_to_csv()
    elif choice == "2":
        participant_id = input("Participant ID (e.g. ID3): ").strip()
        session_name = input("Session (e.g. Session_2): ").strip()
        stimulus = input("Condition (ABF or GM): ").strip().upper()
        save_plots = input("Save plots? (y/n): ").strip().lower() == "y"
        show_plots = input("Display plots? (y/n): ").strip().lower() == "y"
        condition_dir = BASE_PATH / participant_id / session_name / stimulus
        available = list(iter_exercise_files(condition_dir, participant_id, stimulus))
        if not available:
            print("No matching exercise .txt files found.")
        else:
            print("\nAvailable files:")
            for i, (path, exercise, load) in enumerate(available):
                print(f"[{i}] {path.name} ({exercise}, {load}% 1RM)")
            selected = input("Indices to process (e.g. 0,2,3): ").strip()
            try:
                indices = [int(i.strip()) for i in selected.split(",")]
            except ValueError:
                raise SystemExit("Invalid selection")
            for i in dict.fromkeys(indices):
                if 0 <= i < len(available):
                    path, _, _ = available[i]
                    process_single_file(participant_id, session_name, stimulus,
                                        path.name, save_plots, show_plots)
                else:
                    print(f"Invalid index: {i}")
            if input("Export CSV? (y/n): ").strip().lower() == "y":
                export_results_to_csv()
    else:
        print("Invalid choice")
