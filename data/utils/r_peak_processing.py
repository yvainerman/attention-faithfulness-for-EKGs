"""
data/utils/r_peak_processing.py
-------------------------------
R-peak detection and beat segmentation for 12-lead ECGs.

Matches the original implementation exactly:
  1. Per-lead Pan-Tompkins detection with a 5-15 Hz bandpass filter
  2. Multi-lead fusion: peaks are kept only if at least 6 of 12 leads
     confirm them within a 100 ms window (majority voting)
  3. Distance-only peak detection (no height threshold)
  4. RR intervals use the mean RR for the first beat, not 0

This is much more robust than single-lead detection. Inverted or noisy
R-waves on a single lead no longer ruin beat extraction for an ECG.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# =====================================================
# Pan-Tompkins on one lead
# =====================================================
def pan_tompkins_detector(ecg_signal, fs):
    """Pan-Tompkins R-peak detector on a single ECG lead."""
    # 5-15 Hz bandpass — the QRS frequency band
    lowcut, highcut = 5.0, 15.0
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    filtered_ecg = filtfilt(b, a, ecg_signal)

    # Differentiate -> square -> moving average (150 ms window)
    diff_ecg = np.diff(filtered_ecg)
    squared_ecg = diff_ecg ** 2
    window_size = int(0.150 * fs)
    mwa_ecg = np.convolve(
        squared_ecg, np.ones(window_size) / window_size, mode="same"
    )

    # Distance-only peak picking — no height threshold
    peaks, _ = find_peaks(mwa_ecg, distance=int(0.6 * fs))
    return peaks


# =====================================================
# Multi-lead fusion
# =====================================================
def multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=None):
    """
    Combine per-lead R-peak detections into a single peak list.
    A peak is kept only if at least `min_leads` leads agree on it
    (within `fusion_window` seconds). Defaults to majority (>= n_leads / 2).
    """
    n_leads = len(detected_peaks)
    if min_leads is None:
        min_leads = int(np.ceil(n_leads / 2))

    all_peaks = [(p, lead) for lead, peaks in enumerate(detected_peaks) for p in peaks]
    all_peaks.sort(key=lambda x: x[0])

    fused_peaks = []
    i = 0
    while i < len(all_peaks):
        cluster = [all_peaks[i]]
        i += 1
        while i < len(all_peaks) and all_peaks[i][0] - cluster[-1][0] <= fusion_window * fs:
            cluster.append(all_peaks[i])
            i += 1

        unique_leads = {lead for _, lead in cluster}
        if len(unique_leads) >= min_leads:
            fused_peak = int(np.median([p for (p, _) in cluster]))
            fused_peaks.append(fused_peak)

    return np.array(sorted(fused_peaks))


def detect_r_peaks(ecg_signals, fs):
    """Run Pan-Tompkins on every lead, then fuse. Default min_leads=6."""
    detected_peaks = []
    for lead in ecg_signals:
        peaks = pan_tompkins_detector(lead, fs)
        detected_peaks.append(peaks)
    return multi_lead_fusion(detected_peaks, fs, fusion_window=0.1, min_leads=6)


# =====================================================
# Beat extraction
# =====================================================
def extract_segments_around_peaks(signal, r_peaks, pre_samples, post_samples):
    """Cut fixed-length segments around each R-peak from a single lead."""
    segments = []
    for peak in r_peaks:
        start = max(0, peak - pre_samples)
        end   = min(len(signal), peak + post_samples)
        if end - start == pre_samples + post_samples:
            segments.append(signal[start:end])
    return segments


def extract_beats_multi_lead(ecg, fs, pre_samples=99, post_samples=201):
    """
    Cut a 12-lead ECG into individual beats centred on fused R-peaks.

    Parameters
    ----------
    ecg : numpy array, shape (12, L)
    fs : int
        Sampling rate in Hz.
    pre_samples, post_samples : int
        Samples before/after each R-peak. Default 99+201 = 300 (0.6 s @ 500 Hz).

    Returns
    -------
    beats : numpy array, shape (S, T, C)  — S beats, T = pre+post, C = 12 leads
    rr    : numpy array, shape (S,)       — RR intervals in seconds.
                                            First entry is the mean RR.
    Returns (None, None) if no valid beats could be cut.
    """
    ecg = np.array(ecg)

    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) == 0:
        return None, None

    # RR intervals: prepend the mean RR so every beat has a usable value
    rr_intervals = np.diff(r_peaks) / fs
    if len(rr_intervals) > 0:
        mean_rr = float(np.mean(rr_intervals))
        rr_intervals = np.insert(rr_intervals, 0, mean_rr)
    else:
        rr_intervals = np.array([0.0])

    # Drop peaks whose window would go off either end of the recording
    valid_peak_mask = []
    for peak in r_peaks:
        start = max(0, peak - pre_samples)
        end   = min(ecg.shape[1], peak + post_samples)
        valid_peak_mask.append(end - start == pre_samples + post_samples)

    valid_peaks = r_peaks[valid_peak_mask]
    rr_intervals = rr_intervals[valid_peak_mask]

    if len(valid_peaks) == 0:
        return None, None

    # For each lead, extract the same windows; then stack into (S, T, C)
    all_lead_segments = []
    for lead_signal in ecg:
        segments = extract_segments_around_peaks(
            lead_signal, valid_peaks, pre_samples, post_samples
        )
        all_lead_segments.append(segments)

    beats_arr = np.stack(
        [np.stack(segments, axis=0) for segments in all_lead_segments],
        axis=-1,
    )                                                       # (S, T, C)

    return beats_arr.astype(np.float32), rr_intervals.astype(np.float32)