# Acoustic BioPower — Signal Processing and Analysis

## Project Overview

This repository contains the python scripts developed to process and analyse kinetic, kinematic, and electromyographic recordings collected during the ASTROPOWER project's pilot study. The [Acoustic BioPower Dataset](https://doi.org/10.6084/m9.figshare.32090722) is available on Figshare, and the processing pipeline was developed as part of [Beatriz Guerreiro's Master's thesis](http://hdl.handle.net/10362/202184).

The pilot study investigated whether acoustic biofeedback (ABF) acutely influences muscle power production compared with gym music (GM) in male and female participants. It also examined sex-related differences in neuromuscular coordination of the upper and lower limbs under both auditory conditions. The work aimed to establish a reference for performance in healthy young adults and to assess the feasibility of this approach for future studies involving older adults.

ASTROPOWER is an ESA-funded project under the PRODEX programme exploring parallels between the effects of ageing on Earth and microgravity in space. The project focuses on skeletal muscle power production and examines electromyography-based acoustic biofeedback as a potential aid to exercise interventions.

The pipeline processes bench press and squat recordings, including sensor calibration, filtering, movement-phase detection, repetition segmentation, velocity and power estimation, EMG analysis, and extraction of temporal and mechanical metrics. It produces diagnostic plots and CSV files for subsequent analysis.

## Dataset Structure

The complete experimental procedures and recording details are described in the [Figshare dataset repository](https://doi.org/10.6084/m9.figshare.32090722). The scripts expect the following organization:

```text
Acoustic BioPower Dataset/
├── ID1/
│   ├── Session_1/
│   │   ├── ABF/
│   │   │   ├── id1_30_mp_bp_abf.txt
│   │   │   └── ...
│   │   ├── GM/
│   │   │   ├── id1_30_mp_bp_gm.txt
│   │   │   └── ...
│   │   ├── id1_initial_measures.txt
│   │   └── id1_final_measures.txt
│   ├── Session_2/
│   └── Session_3/
├── ID2/
└── ...
```

Participant folders contain session-specific exercise recordings organized by auditory condition (ABF or GM). In an exercise filename such as `id1_30_mp_bp_abf.txt`, `id1` identifies the participant, `30` represents relative load (%1RM), `bp` denotes bench press (`squat` denotes squat), and `abf` identifies the auditory condition. The `initial_measures` and `final_measures` files are part of the dataset but are not analysed by this exercise pipeline.

## Data Format

Exercise recordings are whitespace-delimited `.txt` files with one header row followed by eight numeric columns:

| Column | Description |
|---|---|
| `Nseq` | Sequential sample number |
| `LOADCELL` | Raw load-cell sensor signal |
| `ACCX` | Raw accelerometer X-axis signal |
| `ACCY` | Raw accelerometer Y-axis signal |
| `ACCZ` | Raw accelerometer Z-axis signal |
| `EMGA` | Raw EMG from the triceps brachii (upper limb) |
| `EMGL` | Raw EMG from the vastus lateralis (lower limb) |
| `ECG` | Raw electrocardiographic signal |

The physiological and biomechanical channels contain unprocessed digital acquisition values and require sensor-specific transfer functions to obtain physical units. Signals were acquired at **1000 Hz**, as reflected in the supplied configuration. The pipeline selects `EMGA` for bench press and `EMGL` for squat.

## Repository Structure

| Module | Purpose |
|---|---|
| `main.py` | Batch and single-file command-line runner and CSV export |
| `main_pipeline.py` | Per-recording processing and metric extraction |
| `import_data.py` | Standardized eight-column data loading and validation |
| `config.py` | Sampling frequencies and processing parameters |
| `processing.py` | Unit conversion, calibration, filtering, integration, and smoothing |
| `detection.py` | Movement-phase and EMG onset detection |
| `analysis.py` | Cropping, normalization, waveform correlations, and other analyses |
| `metrics.py` | Mechanical, kinematic, and EMG-derived metrics |
| `plots.py` | Diagnostic visualizations |

## Installation

The project was developed using **python 3.10**. From the repository directory, install the required python packages:

```bash
python -m pip install -r requirements.txt
```

This command assumes `requirements.txt` is included in the repository.

## Configuration

1. **Dataset directory:** In `main.py`, set `BASE_PATH` to the downloaded dataset folder. For example:

   ```python
   BASE_PATH = Path(r"C:\path\to\Acoustic BioPower Dataset")
   ```

2. **Sampling frequencies and processing settings:** Review `FS_ACC`, `FS_EMG`, and the filtering, smoothing, and detection parameters in `config.py`. Keep the original values when seeking to reproduce the thesis analyses.

3. **Calibration:** Ensure that `Acc_calib.txt` is accessible at the location specified by `CALIB_FILE` in `main_pipeline.py`. Calibration or parameter changes may affect extracted metrics.

Results are written to `OUTPUT_ROOT`, which by default points to a `Dataset_Results` folder beside `BASE_PATH`, rather than inside the published dataset.

## Usage

Run the command-line program:

```bash
python main.py
```

**Batch mode (option 1).** Select a session (`Session_1`, `Session_2`, or `Session_3`). The program searches all participant folders for matching exercise files under `ABF` and `GM`, processes valid recordings, and exports summary CSVs. Progress and diagnostic information are logged to the console.

**Single-file mode (option 2).** Specify the participant, session, and condition; choose whether to save or display plots; select one or more files from the displayed list; and optionally export the extracted metrics.

The output structure is illustrated below:

```text
Dataset_Results/
├── ID1/
│   └── Session_1/
│       ├── ABF/
│       │   ├── ID1_Session_1_ABF_bench_press_results.csv
│       │   ├── ID1_Session_1_ABF_squat_results.csv
│       │   └── Plots/
│       │       ├── bench_press/
│       │       │   ├── 30/
│       │       │   │   ├── bar_max_power.png
│       │       │   │   └── ...
│       │       │   └── 40/
│       │       └── squat/
│       └── GM/
└── ID2/
```

Output files and folders are created as applicable, depending on the input recordings and processing results.

## Signal Processing and Analysis

The pipeline comprises the following stages:

1. **Signal calibration and preprocessing:** Convert raw measurements into physical units and filter or smooth acceleration, load-cell, and EMG signals.
2. **Kinematic and kinetic analysis:** Estimate movement velocity from acceleration and calculate mechanical power.
3. **Movement-phase detection:** Identify repetitions and their concentric phases.
4. **Mechanical metrics:** Extract peak power, peak velocity, rate of power development, peak timing, and repetition-to-repetition measures.
5. **EMG metrics:** Detect activation onset and compute timing, envelope rise measures, and median frequency.
6. **Reporting:** Generate diagnostic figures and repetition-level CSV metrics.

The methodological details, mathematical formulations, and parameter choices are documented in [*The Acute Effects of Acoustic Biofeedback on Muscle Power*](http://hdl.handle.net/10362/202184), Beatriz Guerreiro's 2025 Master's thesis. The implementation is distributed across `processing.py`, `detection.py`, `metrics.py`, `analysis.py`, and `main_pipeline.py`.

## Output Metrics

Each CSV row represents a processed exercise recording. Repetition-specific columns use the suffix `_rep_N`, where `N` is the repetition number.

### Recording identifiers

| Column | Description |
|---|---|
| `participant_id` | Participant identifier |
| `session` | Experimental session |
| `stimulus` | ABF or GM condition |
| `exercise` | `bench_press` or `squat` |
| `filename` | Original exercise recording filename |
| `percentage_1RM` | Relative load, expressed as %1RM |

### Mechanical and kinematic metrics

| Column prefix | Description | Unit |
|---|---|---|
| `max_power_rep_` | Maximum mechanical power per repetition | W |
| `max_velocity_rep_` | Movement velocity peak measure per repetition | m/s |
| `max_rpd_rep_` | Maximum rate of power development | W/s |
| `power_total_slope_rep_` | Overall power slope up to its peak | W/s |
| `power_window_slope_mean_rep_` | Mean of window-based power slopes | W/s |

### Timing and phase metrics

| Column prefix | Description | Unit |
|---|---|---|
| `power_index_rep_` | Index of the detected power peak | Sample index |
| `velocity_index_rep_` | Index of the detected velocity peak | Sample index |
| `dif_vel_power_rep_` | Velocity-peak time minus power-peak time | s |
| `rep_start_rep_` | Beginning of repetition in the cropped signal | Sample index |
| `con_start_rep_` | Beginning of concentric phase in the cropped signal | Sample index |
| `con_end_rep_` | End of concentric phase in the cropped signal | Sample index |
| `peak_velocity_rep_` | Velocity peak position within normalized concentric phase | % |
| `peak_power_rep_` | Power peak position within normalized concentric phase | % |

The exported indices are positions within processed/cropped signal arrays; they should **not** be interpreted directly as the original `Nseq` numbers. Normalized peak positions use 0% for the start and 100% for the end of the concentric phase.

### Electromyographic metrics

| Column prefix | Description | Unit |
|---|---|---|
| `emg_onset_to_power_peak_rep_` | Interval from detected EMG onset to the power peak | s |
| `emg_rise_time_rep_` | Interval from EMG onset to the envelope peak | s |
| `emg_rise_slope_rep_` | Rate of rise of the TKEO-based EMG envelope | mV²/s (as implemented) |
| `emg_median_freq_rep_` | EMG median spectral frequency | Hz |

EMG measures use the triceps brachii channel for bench press and the vastus lateralis channel for squat.

### Repetition-to-repetition correlations

| Column prefix | Description | Unit |
|---|---|---|
| `corr_velocity_` | Correlation between normalized velocity waveforms for a pair of repetitions | Dimensionless |
| `corr_acceleration_` | Correlation between normalized acceleration waveforms for a pair of repetitions | Dimensionless |

Columns identify the pair compared, for example `corr_velocity_Rep 1_Rep 2`. Correlation coefficients typically lie between −1 and +1.

### CSV organization

CSV files use a semicolon (`;`) separator and follow this pattern:

```text
{participant}_{session}_{stimulus}_{exercise}_results.csv
```

Diagnostic PNG plots are created separately for visual quality assessment.

## Data Quality and Missing Values

The dataset includes `files_with_missing_values.xlsx`, which identifies recordings with missing data and reports:

- Total number of missing (`NaN`) entries per file.
- Maximum length of a consecutive sequence of missing samples.
- Number of missing-data blocks, defined as sequences of at least two consecutive missing samples.

### Dataset Quality and Exclusion Criteria
The recordings included in the dataset are provided in their original, unprocessed format, without interpolation or correction of missing values. This allows users to independently select and apply missing-data handling strategies according to their analytical objectives.

### Missing-Value Handling in the Processing Pipeline
The provided Python pipeline automatically identifies and removes rows containing NaN or infinite values before signal processing. Consequently, only samples with valid values across all recorded channels are retained for subsequent analyses. If the concentric phases cannot be correctly detected, no metrics are extracted for that file.

Users should be aware that removing missing samples may introduce temporal discontinuities and affect time-dependent metrics, including movement velocity, repetition segmentation, EMG activation timing, and power development.


## Citation

If you use or adapt this processing pipeline, please acknowledge the ASTROPOWER project and cite the software and relevant scientific resources.

### Software repository

Contributors:

- Beatriz Guerreiro
- Dania Furk
- Luís Silva
- Hugo Gamboa

**Repository:** Acoustic BioPower — Signal Processing and Analysis  
**GitHub:** [Repository URL](https://github.com/daniafurk/acousticbiopower.git)

### Associated dataset

**Acoustic BioPower Dataset.** [Figshare](https://doi.org/10.6084/m9.figshare.32090722).

### Master's thesis

Guerreiro, B. V. (2025). *The Acute Effects of Acoustic Biofeedback on Muscle Power* [Master's thesis, NOVA University Lisbon]. [http://hdl.handle.net/10362/202184](http://hdl.handle.net/10362/202184)

```bibtex
@mastersthesis{guerreirobeatrizvicente2025,
    author = {Guerreiro, Beatriz Vicente},
    title  = {The Acute Effects of Acoustic Biofeedback on Muscle Power},
    school = {NOVA University Lisbon},
    year   = {2025},
    url    = {http://hdl.handle.net/10362/202184}
}
```
