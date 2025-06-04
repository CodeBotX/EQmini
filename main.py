import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wfdb

from filters import  BandPassFilter
from transforms import DiscreteFourierTransform

# hàm tính phổ biên độ - xem trong tín hiệu có những tần số nào và biên độ của chúng
def compute_magnitude_spectrum(signal, fs):
    N = len(signal)
    X = DiscreteFourierTransform(signal).transform()
    magnitude = [abs(x) for x in X[:N//2]]
    freqs = np.linspace(0, fs/2, N//2)
    return freqs, magnitude

# phát hiện đỉnh - sử dụng hàm có sẵn của scipy để phát hiện đỉnh R trong tín hiệu ECG
def detect_heart_rate(signal, fs):
    # Dùng find_peaks để phát hiện đỉnh R
    peaks, _ = find_peaks(signal, distance=fs * 0.6)  # ít nhất 0.6s giữa 2 đỉnh
    duration_sec = len(signal) / fs
    bpm = len(peaks) * 60 / duration_sec
    return bpm, peaks


def main():
    # Đọc dữ liệu từ file cục bộ
    record = wfdb.rdrecord(
        'E:/projects/EQmini/mit-bih-arrhythmia-database-1.0.0/100',
        sampto=2000
    )

    fs = record.fs
    ecg = record.p_signal[:, 0]  # Tín hiệu gốc

    # Thêm nhiễu Gaussian
    noise = 0.5 * np.random.normal(size=len(ecg))
    noisy_ecg = ecg + noise

    # Áp dụng cửa sổ Hamming trước khi lọc
    window = np.hamming(len(noisy_ecg))
    noisy_win = noisy_ecg * window

    # Lọc bằng BandPassFilter (0.5–40Hz)
    bpf = BandPassFilter(0.5, 40, fs)
    filtered = bpf.apply(noisy_win)

    # Phát hiện đỉnh R & nhịp tim
    bpm, peaks = detect_heart_rate(filtered, fs)
    print(f"Đã phát hiện {len(peaks)} đỉnh R")
    print(f"Nhịp tim trung bình: {bpm:.2f} bpm")

    # Vẽ tín hiệu theo thời gian
    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.plot(ecg, label="Tín hiệu gốc")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(noisy_ecg, label="Tín hiệu có nhiễu")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(noisy_win, label="Sau khi cộng nhiễu + Hamming")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(filtered, label="Sau lọc (0.5-40Hz)")
    plt.plot(peaks, [filtered[p] for p in peaks], "rx", label="Đỉnh R")
    plt.legend()

    # Vẽ phổ biên độ
    freqs, mag_ecg = compute_magnitude_spectrum(ecg, fs)
    _, mag_noisy = compute_magnitude_spectrum(noisy_ecg, fs)
    _, mag_filtered = compute_magnitude_spectrum(filtered, fs)

    plt.figure(figsize=(12, 6))
    plt.plot(freqs, mag_ecg, label="Phổ gốc")
    plt.plot(freqs, mag_noisy, label="Phổ có nhiễu")
    plt.plot(freqs, mag_filtered, label="Phổ sau lọc")
    plt.title("Phổ tín hiệu ECG (DFT)")
    plt.xlabel("Tần số (Hz)")
    plt.ylabel("|X(f)|")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
