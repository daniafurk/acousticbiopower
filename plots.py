"""
Visualization utilities for the project (Matplotlib + seaborn).

This module provides reusable plotting functions for time-series, phase annotations,
per-rep comparisons, mean±SD envelopes, correlation heatmaps, and slope/RPD views.
Not all functions here are invoked by `main_pipeline.run_pipeline; several are
kept for debugging, QA, and exploratory analysis. Use them ad hoc as needed.

Units (as typically produced by the pipeline):
- Acceleration: m/s²
- Velocity: m/s
- Force (load cell): N
- Power: W
- EMG (raw/filtered): mV
- EMG envelope (TKEO-based): mV²
"""

import os
from typing import Sequence, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

############################### ACC AND LOAD CELL SIGNALS ###############################

def plot_raw_signals(
    acc_signal: np.ndarray,
    loadcell_signal: np.ndarray,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
) -> None:
    """Plot raw accelerometer and load-cell signals (samples on x-axis)."""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(acc_signal, color="blue")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Raw Accelerometer Signal")

    plt.subplot(2, 1, 2)
    plt.plot(loadcell_signal, color="red", label="Load Cell")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.title("Raw Load Cell Signal")

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_raw_and_filtered_signals(
    acc_raw,
    acc_filtered,
    load_raw,
    load_filtered,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Compare raw vs. filtered accelerometer and load-cell signals."""
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(acc_raw, color="orange")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("Original ACC Signal")
    axs[0].grid()

    axs[1].plot(acc_filtered, color="blue")
    axs[1].set_ylabel("Acc (m/s²)")
    axs[1].set_title("Filtered ACC Signal (Bandpass + Gaussian)")
    axs[1].grid()

    axs[2].plot(load_raw, color="purple")
    axs[2].set_ylabel("Force (N)")
    axs[2].set_title("Original Load Cell Signal")
    axs[2].grid()

    axs[3].plot(load_filtered, color="red")
    axs[3].set_ylabel("Force (N)")
    axs[3].set_xlabel("Samples")
    axs[3].set_title("Filtered Load Cell Signal (Gaussian)")
    axs[3].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_acc_and_velocity(
    acc_signal,
    velocity,
    time,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot filtered acceleration (m/s²) and its integrated velocity (m/s) vs time."""
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(acc_signal, color="blue")
    plt.title("Filtered ACC Signal")
    plt.ylabel("Acceleration (m/s²)")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, velocity, color="red")
    plt.title("Velocity (Integrated from ACC)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_force_over_time(
    force,
    time,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot force (N) vs time (s)."""
    plt.figure(figsize=(12, 6))
    plt.plot(time, force, color="green")
    plt.title("Force Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.grid()
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_power_filters(
    time,
    power,
    filtered_list: list[np.ndarray],
    labels: list[str],
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot original power and multiple filtered variants for comparison."""
    num_plots = len(filtered_list) + 1
    colors = plt.get_cmap("tab10")
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time, power, color="purple")
    axs[0].set_ylabel("Power (W)")
    axs[0].set_title("Original Power Signal")
    axs[0].grid()

    for i, (filtered, label) in enumerate(zip(filtered_list, labels)):
        axs[i + 1].plot(time, filtered, color=colors(i % 10))
        axs[i + 1].set_ylabel("Power (W)")
        axs[i + 1].set_title(label)
        axs[i + 1].grid()

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_biomechanical_signals(
    time,
    acc,
    velocity,
    filtered_loadcell_gaussian,
    power,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot acceleration, velocity, force and power in a single figure."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time, acc, color="g")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("Acceleration Signal")
    axs[0].grid()

    axs[1].plot(time, velocity, color="r")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity")
    axs[1].grid()

    axs[2].plot(time, filtered_loadcell_gaussian, color="purple")
    axs[2].set_ylabel("Force (N)")
    axs[2].set_title("Force")
    axs[2].grid()

    axs[3].plot(time, power, color="orange")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("Power (Force × Velocity)")
    axs[3].set_xlabel("Time (s)")
    axs[3].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_repetition_ends(
    time,
    velocity,
    ends,
    show: bool = True,
    save: bool = False,
    save_path: str = None,
):
    """Plot velocity with markers at detected repetition ends."""
    plt.figure(figsize=(10, 4))
    plt.plot(time, velocity, "r", label="Velocity")
    plt.scatter(time[ends], velocity[ends], color="green", label="Repetition End", zorder=5)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title("Detected Repetition Ends (First zero-crossing after concentric peak)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_signals_with_repetition_markers(
    time,
    acc_signal,
    velocity_signal,
    power_signal,
    rep_starts,
    rep_ends,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot ACC, velocity and power with vertical lines at rep start/end."""
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(time, acc_signal, label="Acceleration", color="green")
    for i, s in enumerate(rep_starts):
        axs[0].axvline(time[s], color="green", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(rep_ends):
        axs[0].axvline(time[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("Acceleration")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time, velocity_signal, label="Velocity", color="blue")
    for i, s in enumerate(rep_starts):
        axs[1].axvline(time[s], color="green", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(rep_ends):
        axs[1].axvline(time[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].set_title("Velocity")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(time, power_signal, label="Power", color="orange")
    for i, s in enumerate(rep_starts):
        axs[2].axvline(time[s], color="green", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(rep_ends):
        axs[2].axvline(time[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Power (W)")
    axs[2].set_title("Power (Force × Velocity)")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_cut_signals_with_peak_power(
    time_cut,
    acc_cut,
    filtered_loadcell_cut,
    velocity_cut,
    power_cut,
    starts_cut,
    ends_cut,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot cut ACC, load-cell, velocity and power with rep markers and peak power."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(time_cut, acc_cut, color="r")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("ACC Signal")
    axs[0].grid()

    axs[1].plot(time_cut, filtered_loadcell_cut, color="b")
    for i, s in enumerate(starts_cut):
        axs[1].axvline(time_cut[s], color="purple", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(ends_cut):
        axs[1].axvline(time_cut[e], color="purple", linestyle="--", label="End" if i == 0 else "")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Load Cell Signal")
    axs[1].grid()

    axs[2].plot(time_cut, velocity_cut, color="g")
    for i, s in enumerate(starts_cut):
        axs[2].axvline(time_cut[s], color="purple", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(ends_cut):
        axs[2].axvline(time_cut[e], color="purple", linestyle="--", label="End" if i == 0 else "")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Velocity")
    axs[2].grid()

    axs[3].plot(time_cut, power_cut, label="Power", color="pink")
    for i, s in enumerate(starts_cut):
        axs[3].axvline(time_cut[s], color="purple", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(ends_cut):
        axs[3].axvline(time_cut[e], color="purple", linestyle="--", label="End" if i == 0 else "")

    # Peak power per repetition
    for i, (start, end) in enumerate(zip(starts_cut, ends_cut)):
        segment = power_cut[start : end + 1]
        if len(segment) == 0:
            continue
        max_power = np.max(segment)
        max_idx = np.argmax(segment) + start
        axs[3].plot(time_cut[max_idx], max_power, "o", color="purple")
        axs[3].text(
            time_cut[max_idx],
            max_power + 5,
            f"{max_power:.1f} W",
            color="purple",
            ha="center",
            fontsize=9,
        )

    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("Detection of Repetitions and Peak Power")
    axs[3].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_concentric_phase_signals(
    time_cut,
    acc_cut,
    filtered_loadcell_cut,
    velocity_cut,
    power_filtered_cut,
    power_cut,
    concentric_starts_cut,
    concentric_ends_cut,
    peak_powers,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot signals with concentric start/end and peak power markers (filtered & raw)."""
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # 1. ACC
    axs[0].plot(time_cut, acc_cut, color="orange")
    for i, s in enumerate(concentric_starts_cut):
        axs[0].axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(concentric_ends_cut):
        axs[0].axvline(time_cut[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("ACC Signal")
    axs[0].grid()

    # 2. Load Cell
    axs[1].plot(time_cut, filtered_loadcell_cut, color="b")
    for i, s in enumerate(concentric_starts_cut):
        axs[1].axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(concentric_ends_cut):
        axs[1].axvline(time_cut[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Load Cell Signal")
    axs[1].grid()

    # 3. Velocity (filtered for trend view)
    axs[2].plot(time_cut, velocity_cut, color="g")
    for i, s in enumerate(concentric_starts_cut):
        axs[2].axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(concentric_ends_cut):
        axs[2].axvline(time_cut[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Velocity")
    axs[2].grid()

    # 4. Power filtered + markers
    peak_indices_power = []
    axs[3].plot(time_cut, power_filtered_cut, label="Power", color="orchid", alpha=0.5)
    for i, (cs, ce, peak_power) in enumerate(zip(concentric_starts_cut, concentric_ends_cut, peak_powers)):
        axs[3].axvline(time_cut[cs], color="green", linestyle="--")
        axs[3].axvline(time_cut[ce], color="red", linestyle="--")
        peak_index_relative = np.argmax(power_filtered_cut[cs:ce])
        peak_index_global = cs + peak_index_relative
        peak_indices_power.append(peak_index_global)
        axs[3].plot(time_cut[peak_index_global], power_filtered_cut[peak_index_global], "o", color="purple")
        axs[3].annotate(
            f"{peak_power:.1f} W",
            (time_cut[peak_index_global], power_filtered_cut[peak_index_global]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="purple",
        )
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("Detection of Concentric Phase and Peak Power")
    axs[3].legend()
    axs[3].grid()

    # 5. Power raw with the same peak markers
    axs[4].plot(time_cut, power_cut, label="Power (raw)", color="darkorchid", alpha=0.6)
    for cs, ce in zip(concentric_starts_cut, concentric_ends_cut):
        axs[4].axvline(time_cut[cs], color="green", linestyle="--")
        axs[4].axvline(time_cut[ce], color="red", linestyle="--")
    for peak_idx_glob in peak_indices_power:
        peak_val = power_cut[peak_idx_glob]
        axs[4].plot(time_cut[peak_idx_glob], peak_val, "o", color="orange")
        axs[4].annotate(
            f"{peak_val:.1f} W",
            (time_cut[peak_idx_glob], peak_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="orange",
        )
    axs[4].set_ylabel("Power (W)")
    axs[4].set_title("Raw Power Signal with Peak Markers (from Filtered)")
    axs[4].legend()
    axs[4].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_concentric_phase_signals_seasonal(
    time_cut,
    acc_cut,
    filtered_loadcell_cut,
    velocity_raw_cut,
    velocity_cut,
    power_filtered_cut,
    power_cut,
    concentric_starts_cut,
    concentric_ends_cut,
    peak_powers,
    show: bool = False,
    save: bool = True,
    save_path: str = None,
):
    """Like the previous plot but showing BOTH velocity (trend) and velocity_raw."""
    fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)

    # ACC
    axs[0].plot(time_cut, acc_cut, color="orange")
    for i, s in enumerate(concentric_starts_cut):
        axs[0].axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(concentric_ends_cut):
        axs[0].axvline(time_cut[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("ACC Signal")
    axs[0].grid()

    # Load Cell
    axs[1].plot(time_cut, filtered_loadcell_cut, color="b")
    for i, s in enumerate(concentric_starts_cut):
        axs[1].axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
    for i, e in enumerate(concentric_ends_cut):
        axs[1].axvline(time_cut[e], color="red", linestyle="--", label="End" if i == 0 else "")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Load Cell Signal")
    axs[1].grid()

    # Velocity (trend) with peak markers
    peak_indices_velocity = []
    axs[2].plot(time_cut, velocity_cut, label="Velocity", color="royalblue", alpha=0.6)
    for cs, ce in zip(concentric_starts_cut, concentric_ends_cut):
        axs[2].axvline(time_cut[cs], color="green", linestyle="--")
        axs[2].axvline(time_cut[ce], color="red", linestyle="--")
        peak_idx_rel = np.argmax(velocity_cut[cs:ce])
        peak_idx_glob = cs + peak_idx_rel
        peak_val = velocity_cut[peak_idx_glob]
        peak_indices_velocity.append(peak_idx_glob)
        axs[2].plot(time_cut[peak_idx_glob], peak_val, "o", color="darkorange")
        axs[2].annotate(
            f"{peak_val:.2f} m/s",
            (time_cut[peak_idx_glob], peak_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="darkorange",
        )
        axs[2].set_ylabel("Velocity (m/s)")
        axs[2].set_title("Detection of Concentric Phase and Peak Velocity")
        axs[2].legend()
        axs[2].grid()

    # Velocity (raw)
    axs[3].plot(time_cut, velocity_raw_cut, label="Original Velocity", color="royalblue", alpha=0.6)
    for cs, ce in zip(concentric_starts_cut, concentric_ends_cut):
        axs[3].axvline(time_cut[cs], color="green", linestyle="--")
        axs[3].axvline(time_cut[ce], color="red", linestyle="--")
    for peak_idx_glob in peak_indices_velocity:
        peak_val_raw = velocity_raw_cut[peak_idx_glob]
        axs[3].plot(time_cut[peak_idx_glob], peak_val_raw, "o", color="darkorange")
        axs[3].annotate(
            f"{peak_val_raw:.2f} m/s",
            (time_cut[peak_idx_glob], peak_val_raw),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="darkorange",
        )
    axs[3].set_ylabel("Velocity (m/s)")
    axs[3].set_title("Raw Velocity with Concentric Phases and Peak Markers")
    axs[3].legend()
    axs[3].grid()

    # Power filtered + markers
    peak_indices_power = []
    axs[4].plot(time_cut, power_filtered_cut, label="Power", color="orchid", alpha=0.5)
    for i, (cs, ce, peak_power) in enumerate(zip(concentric_starts_cut, concentric_ends_cut, peak_powers)):
        axs[4].axvline(time_cut[cs], color="green", linestyle="--")
        axs[4].axvline(time_cut[ce], color="red", linestyle="--")
        peak_index_relative = np.argmax(power_filtered_cut[cs:ce])
        peak_index_global = cs + peak_index_relative
        peak_indices_power.append(peak_index_global)
        axs[4].plot(time_cut[peak_index_global], power_filtered_cut[peak_index_global], "o", color="purple")
        axs[4].annotate(
            f"{peak_power:.1f} W",
            (time_cut[peak_index_global], power_filtered_cut[peak_index_global]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="purple",
        )
    axs[4].set_xlabel("Time (s)")
    axs[4].set_ylabel("Power (W)")
    axs[4].set_title("Detection of Concentric Phase and Peak Power")
    axs[4].legend()
    axs[4].grid()

    # Power raw
    axs[5].plot(time_cut, power_cut, label="Power (raw)", color="darkorchid", alpha=0.6)
    for cs, ce in zip(concentric_starts_cut, concentric_ends_cut):
        axs[5].axvline(time_cut[cs], color="green", linestyle="--")
        axs[5].axvline(time_cut[ce], color="red", linestyle="--")
    for peak_idx_glob in peak_indices_power:
        peak_val = power_cut[peak_idx_glob]
        axs[5].plot(time_cut[peak_idx_glob], peak_val, "o", color="orange")
        axs[5].annotate(
            f"{peak_val:.1f} W",
            (time_cut[peak_idx_glob], peak_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="orange",
        )
    axs[5].set_ylabel("Power (W)")
    axs[5].set_title("Raw Power Signal with Peak Markers (from Filtered)")
    axs[5].legend()
    axs[5].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_full_phase_signals(
    time_cut,
    acc_cut,
    filtered_loadcell_cut,
    velocity_cut,
    power_cut,
    starts_cut,
    concentric_starts_cut,
    concentric_ends_cut,
    show=True,
    save=False,
    save_path=None,
):
    """Plot ACC, load-cell, velocity and power with eccentric/concentric markers."""
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(time_cut, acc_cut, color="orange")
    for i, (s, cs) in enumerate(zip(starts_cut, concentric_starts_cut)):
        axs[0].axvline(time_cut[s], color="red", linestyle="--", label="Eccentric Start" if i == 0 else "")
        axs[0].axvline(time_cut[cs], color="black", linestyle="-.", label="Eccentric End / Concentric Start" if i == 0 else "")
    for i, ce in enumerate(concentric_ends_cut):
        axs[0].axvline(time_cut[ce], color="red", linestyle="--", label="Concentric End" if i == 0 else "")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("Acceleration")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_cut, filtered_loadcell_cut, color="blue")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[1].axvline(time_cut[s], color="red", linestyle="--")
        axs[1].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[1].axvline(time_cut[ce], color="red", linestyle="--")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Load Cell")
    axs[1].grid()

    axs[2].plot(time_cut, velocity_cut, color="green")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[2].axvline(time_cut[s], color="red", linestyle="--")
        axs[2].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[2].axvline(time_cut[ce], color="red", linestyle="--")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Velocity")
    axs[2].grid()

    axs[3].plot(time_cut, power_cut, color="hotpink")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[3].axvline(time_cut[s], color="red", linestyle="--")
        axs[3].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[3].axvline(time_cut[ce], color="red", linestyle="--")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("Power")
    axs[3].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_normalized_signal_per_repetition(
    relative_time_percent,
    normalized_curves,
    peak_positions,
    signal_name: str = "Signal",
    unit: str = "a.u.",
    color: str = "blue",
    show: bool = True,
    save: bool = False,
    save_path: str = None,
):
    """Plot normalized 0–100% concentric curves per repetition with peak markers."""
    num_reps = len(normalized_curves)
    fig, axs = plt.subplots(num_reps, 1, figsize=(8, 2.5 * num_reps), sharex=True)

    for i, (percent, y, peak) in enumerate(zip(relative_time_percent, normalized_curves, peak_positions)):
        ax = axs[i] if num_reps > 1 else axs
        ax.plot(percent, y, color=color, label=signal_name)
        ax.axvline(peak, color="red", linestyle="--", label=f"Peak @ {peak:.1f}%")
        ax.set_ylabel(f"{signal_name} ({unit})")
        ax.set_title(f"{signal_name} – Repetition {i + 1}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    (axs if num_reps == 1 else axs[-1]).set_xlabel("Concentric Phase (% of cycle)")
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_bar_max_power(
    max_powers,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Bar chart of maximum power per repetition."""
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(max_powers) + 1), max_powers, color="mediumslateblue")
    plt.title("Maximum Power per Repetition")
    plt.xlabel("Repetition")
    plt.ylabel("Power (W)")
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_matrix(
    matrix: pd.DataFrame,
    title: str,
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Heatmap of repetition-to-repetition correlation."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu", center=0, square=True, cbar=True)
    plt.title(title)
    plt.xlabel("Repetitions")
    plt.ylabel("Repetitions")
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_mean_std_curve(
    mean,
    std,
    title: str = "Mean Curve",
    ylabel: str = "Metric",
    color: str = "darkblue",
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot mean curve with ±1 SD shaded band over normalized time (0–100%)."""
    x = np.linspace(0, 100, len(mean))
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean, color=color, label="Mean")
    plt.fill_between(x, mean - std, mean + std, color="lightblue", alpha=0.5, label="±1 SD")
    plt.title(title)
    plt.xlabel("Normalized Time (%)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_each_normalized_segment(
    normalized_data,
    ylabel: str = "Power (W)",
    save_path: str = None,
    save: bool = True,
    show: bool = False,
):
    """Plot each repetition curve as a separate subplot (normalized 0–100%)."""
    num_reps = normalized_data.shape[0]
    num_points = normalized_data.shape[1]
    fig, axs = plt.subplots(num_reps, 1, figsize=(10, 4 * num_reps), sharex=True)

    if num_reps == 1:
        axs = [axs]

    for i in range(num_reps):
        axs[i].plot(np.linspace(0, 100, num_points), normalized_data[i], color="navy", label=f"Repetition {i+1}")
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(f"Repetition {i+1}")
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Normalized Time (%)")
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_rpd_power(
    time_cut: np.ndarray,
    power_filtered_cut: np.ndarray,
    starts_cut: Sequence[int],
    concentric_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    show: bool = True,
    save: bool = False,
    save_path: str = None,
):
    """Plot Rate of Power Development (RPD = dPower/dt) alongside filtered power."""
    rpd_power = np.gradient(power_filtered_cut)
    plt.figure(figsize=(12, 6))

    plt.plot(time_cut, rpd_power * 100, label="RPD (x100)", color="red", alpha=0.5)
    plt.plot(time_cut, power_filtered_cut, label="Filtered Power", color="blue")

    for i, (s, cs) in enumerate(zip(starts_cut, concentric_starts_cut)):
        plt.axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
        plt.axvline(time_cut[cs], color="black", linestyle="-.", label="Concentric Start" if i == 0 else "")
    for i, ce in enumerate(concentric_ends_cut):
        plt.axvline(time_cut[ce], color="red", linestyle="--", label="Concentric End" if i == 0 else "")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("RPD and Power")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_power_slopes_all_reps(
    all_window_slopes: List[Sequence[float]],
    total_slopes: Sequence[float],
    window_ms: float,
    save_dir: str,
    save: bool = True,
    show: bool = False,
):
    """Plot window-by-window power slopes for all reps (one figure per rep)."""
    window_s = float(window_ms) / 1000.0
    bar_width = 0.9 * window_s  # e.g., 0.045 s when window_ms=50

    if save:
        os.makedirs(save_dir, exist_ok=True)

    for i, (slopes, slope_total) in enumerate(zip(all_window_slopes, total_slopes)):
        slopes = np.asarray(slopes, dtype=float)
        if slopes.size == 0:
            continue

        time_axis = (np.arange(slopes.size) + 0.5) * window_s  # centers: 0.5*win, 1.5*win, ...

        plt.figure(figsize=(10, 5))
        plt.bar(time_axis, slopes, width=bar_width, color="skyblue", edgecolor="black",
                label=f"Window Slopes ({int(window_ms)} ms)")
        plt.plot(time_axis, slopes, color="blue", marker="o", linewidth=2, label="Slope Trend")

        if np.isfinite(slope_total):
            plt.axhline(slope_total, color="red", linestyle="--", label=f"Total Slope = {slope_total:.2f} W/s")

        plt.title(f"Repetition {i + 1} – Slopes with Trend (Concentric Start → Peak Power)")
        plt.xlabel("Time from Concentric Start (s)")
        plt.ylabel("Slope (W/s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(save_dir, f"slopes_rep_{i + 1}.png"))
        if show:
            plt.show()
        else:
            plt.close()


######################################## EMG SIGNAL ########################################

def plot_full_phase_signals_with_emg(
    time_cut: np.ndarray,
    acc_cut: np.ndarray,
    filtered_loadcell_gaussian_cut: np.ndarray,
    velocity_cut: np.ndarray,
    power_filtered_cut: np.ndarray,
    emg_cut: np.ndarray,
    starts_cut: Sequence[int],
    concentric_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    show: bool = True,
    save: bool = False,
    save_path: str = None,
):
    """Plot ACC, load-cell, velocity, power and raw EMG with phase markers."""
    fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axs[0].plot(time_cut, acc_cut, color="orange")
    for i, (s, cs) in enumerate(zip(starts_cut, concentric_starts_cut)):
        axs[0].axvline(time_cut[s], color="red", linestyle="--", label="Eccentric Start" if i == 0 else "")
        axs[0].axvline(time_cut[cs], color="black", linestyle="-.", label="Concentric Start" if i == 0 else "")
    for i, ce in enumerate(concentric_ends_cut):
        axs[0].axvline(time_cut[ce], color="red", linestyle="--", label="Concentric End" if i == 0 else "")
    axs[0].set_ylabel("Acc (m/s²)")
    axs[0].set_title("Acceleration")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(time_cut, filtered_loadcell_gaussian_cut, color="blue")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[1].axvline(time_cut[s], color="red", linestyle="--")
        axs[1].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[1].axvline(time_cut[ce], color="red", linestyle="--")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Load Cell")
    axs[1].grid()

    axs[2].plot(time_cut, velocity_cut, color="green")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[2].axvline(time_cut[s], color="red", linestyle="--")
        axs[2].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[2].axvline(time_cut[ce], color="red", linestyle="--")
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Velocity")
    axs[2].grid()

    axs[3].plot(time_cut, power_filtered_cut, color="hotpink")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[3].axvline(time_cut[s], color="red", linestyle="--")
        axs[3].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[3].axvline(time_cut[ce], color="red", linestyle="--")
    axs[3].set_ylabel("Power (W)")
    axs[3].set_title("Power")
    axs[3].grid()

    axs[4].plot(time_cut, emg_cut, color="brown")
    for s, cs in zip(starts_cut, concentric_starts_cut):
        axs[4].axvline(time_cut[s], color="red", linestyle="--")
        axs[4].axvline(time_cut[cs], color="black", linestyle="-.")
    for ce in concentric_ends_cut:
        axs[4].axvline(time_cut[ce], color="red", linestyle="--")
    axs[4].set_ylabel("EMG (mV)")
    axs[4].set_title("Raw EMG")
    axs[4].grid()

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_emg_with_onsets_and_valid_peaks(
    time_cut: np.ndarray,
    emg_cut: np.ndarray,
    envelope: np.ndarray,
    estopt_onset_times: Sequence[float],
    starts_cut: Sequence[int],
    concentric_starts_cut: Sequence[int],
    concentric_ends_cut: Sequence[int],
    emg_peak_indices: List[List[int]],
    show: bool = True,
    save: bool = False,
    save_path: str = None,
):
    """Plot raw EMG and envelope with EstOpt onsets and accepted envelope peaks."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(time_cut, emg_cut, label="Raw EMG (mV)", color="steelblue", alpha=0.4)
    ax1.plot(time_cut, envelope * 10, label="Envelope (×10, mV²)", color="orange")

    for i, onset_time in enumerate(estopt_onset_times):
        ax1.axvline(onset_time, color="green", linestyle="--", label="EstOpt Onset" if i == 0 else "")

    for i, (s, cs) in enumerate(zip(starts_cut, concentric_starts_cut)):
        ax1.axvline(time_cut[s], color="red", linestyle="--", label="Start" if i == 0 else "")
        ax1.axvline(time_cut[cs], color="black", linestyle="-.", label="Concentric Start" if i == 0 else "")
    for i, ce in enumerate(concentric_ends_cut):
        ax1.axvline(time_cut[ce], color="red", linestyle="--", label="Concentric End" if i == 0 else "")

    for i, peaks in enumerate(emg_peak_indices):
        for j, p_idx in enumerate(peaks):
            if 0 <= p_idx < len(time_cut):
                ax1.plot(time_cut[p_idx], envelope[p_idx] * 10, "mo", label="EMG Peak" if (i == 0 and j == 0) else "")

    ax1.set_title("EMG Signal with EstOpt Onsets and Valid Envelope Peaks")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()