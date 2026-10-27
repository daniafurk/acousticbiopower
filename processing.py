"""
Contains signal processing functions for filtering, calibration, integration, and smoothing.
Includes Butterworth bandpass/lowpass filters, Gaussian smoothing, RMS/envelope generation, and
spline fitting for acceleration. Also performs integration to derive velocity and force from raw signals.
"""


import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

############################## ACC and Load Cell #######################################################################################


def transfer_weight(bit_value: np.ndarray) -> np.ndarray:
    """
    Converts load cell raw bit values to weight in kilograms.

    Args:
        bit_value (np.ndarray): Raw bit values from the load cell.

    Returns:
        np.ndarray: Weight in kilograms.
    """
    return (-0.0029 * bit_value + 94.2825) * 9.81


def get_acc_calibration(calib_file_path: str) -> tuple:
    """
    Retrieves min and max calibration values for each accelerometer axis (X, Y, Z).

    Args:
        calib_file_path (str): Full path to the calibration file.

    Returns:
        tuple: (X_min, X_max, Y_min, Y_max, Z_min, Z_max)
    """
    calib_data = np.loadtxt(calib_file_path, skiprows=1)

    C_X_min, C_X_max = np.min(calib_data[:, 2]), np.max(calib_data[:, 2])
    C_Y_min, C_Y_max = np.min(calib_data[:, 3]), np.max(calib_data[:, 3])
    C_Z_min, C_Z_max = np.min(calib_data[:, 4]), np.max(calib_data[:, 4])

    return C_X_min, C_X_max, C_Y_min, C_Y_max, C_Z_min, C_Z_max


def transfer_acc(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray,
                 calib_file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts raw accelerometer data to acceleration in m/s² using calibration values.

    Args:
        x_data, y_data, z_data (np.ndarray): Raw axis data.
        calib_file_path (str): Path to the accelerometer calibration file.

    Returns:
        tuple: Acceleration signals in m/s² for X, Y, and Z axes.
    """
    C_X_min, C_X_max, C_Y_min, C_Y_max, C_Z_min, C_Z_max = get_acc_calibration(calib_file_path)

    x = (2 * (x_data - C_X_min) / (C_X_max - C_X_min)) - 1
    y = (2 * (y_data - C_Y_min) / (C_Y_max - C_Y_min)) - 1
    z = (2 * (z_data - C_Z_min) / (C_Z_max - C_Z_min)) - 1

    return x * 9.8, y * 9.8, z * 9.8

def bandpass_filter(signal, low_cutoff, high_cutoff, fs, order):
    """
    Applies a Butterworth bandpass filter to a signal.

    Args:
        signal (np.ndarray): Input signal.
        low_cutoff (float): Low cutoff frequency in Hz.
        high_cutoff (float): High cutoff frequency in Hz.
        fs (int): Sampling frequency in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Bandpass-filtered signal.
    """
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)



def lowpass_filter(signal, cutoff, fs, order=3):
    """
    Applies a low-pass Butterworth filter.

    Args:
        signal (np.ndarray): Input signal.
        cutoff (float): Cutoff frequency in Hz.
        fs (int): Sampling frequency.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)


def savitzky_golay_filter(data, window_size=11, poly_order=2):
    """
    Applies Savitzky-Golay smoothing filter.

    Args:
        data (np.ndarray): Input signal.
        window_size (int): Size of the moving window.
        poly_order (int): Polynomial order.

    Returns:
        np.ndarray: Smoothed signal.
    """
    return savgol_filter(data, window_size, poly_order)

################################### EMG signal ##############################################3

def transfer_EMG(data):
    """
    Converts raw EMG data from bits to millivolts (mV), based on sensor specifications.

    Parameters:
        data (array): EMG signal in raw ADC units (16-bit).

    Returns:
        array: EMG signal converted to millivolts (mV).
    """
    # Formula derived from ADC resolution (16-bit), input voltage range (0–2.5 V), and gain (1100)
    data_mv = (((data / (2 ** 16 - 1)) * 2.5) / 1100) * 1000
    return data_mv


