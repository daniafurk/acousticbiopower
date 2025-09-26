"""
Configuration constants for the signal processing.
"""
############################### ACC AND LOAD CELL SIGNALS ###################################################

# Sampling rates
FS_ACC = 1000      # Accelerometer, load cell and EMG sampling rate
FS_ECG = 1000
FS_EMG = 1000

# Filter cutoffs for accelerometer
ACC_BANDPASS_LOW = 0.5     # Hz
ACC_BANDPASS_HIGH = 5.0   # Hz


# Filter orders
BANDPASS_ORDER = 3
LOWPASS_ORDER = 4

# Gaussian smoothing sigmas
SIGMA_ACC_LOW = 3
SIGMA_ACC_BAND = 3
SIGMA_ACC_BAND_STRONG = 6
SIGMA_LOADCELL = 20

# Repetition filtering and signal cropping
MIN_PEAK_VELOCITY = 0.15                 # Minimum peak velocity to accept a repetition
EXTRA_TIME_BEFORE_START = 2.0      # Extra seconds before first valid start
EXTRA_TIME_AFTER_END = 3.0         # Extra seconds after last valid end







