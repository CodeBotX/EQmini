#Tạo các bộ lọc 

import numpy as np

class FIRFilter:
    def __init__(self, sample_rate, filter_order=64):
        self.sample_rate = sample_rate
        self.filter_order = filter_order
        
    def _create_window(self):
        """Tạo cửa sổ Hanning cho bộ lọc"""
        return np.hanning(self.filter_order)
    
    def _apply_window(self, h):
        """Áp dụng cửa sổ cho đáp ứng xung"""
        return h * self._create_window()

class LowPassFilter(FIRFilter):
    def __init__(self, sample_rate, cutoff_freq, filter_order=64):
        super().__init__(sample_rate, filter_order)
        self.cutoff_freq = cutoff_freq
        
    def design(self):
        """Thiết kế bộ lọc thông thấp"""
        # Tính toán đáp ứng xung lý tưởng
        n = np.arange(self.filter_order)
        fc = self.cutoff_freq / (self.sample_rate / 2)  # Chuẩn hóa tần số cắt
        h = np.sinc(2 * fc * (n - (self.filter_order - 1) / 2))
        
        # Áp dụng cửa sổ
        h = self._apply_window(h)
        
        # Chuẩn hóa
        h = h / np.sum(h)
        return h

class HighPassFilter(FIRFilter):
    def __init__(self, sample_rate, cutoff_freq, filter_order=64):
        super().__init__(sample_rate, filter_order)
        self.cutoff_freq = cutoff_freq
        
    def design(self):
        """Thiết kế bộ lọc thông cao"""
        # Tính toán đáp ứng xung lý tưởng
        n = np.arange(self.filter_order)
        fc = self.cutoff_freq / (self.sample_rate / 2)  # Chuẩn hóa tần số cắt
        h = np.sinc(2 * fc * (n - (self.filter_order - 1) / 2))
        
        # Chuyển đổi LPF thành HPF
        h = -h
        h[(self.filter_order - 1) // 2] += 1
        
        # Áp dụng cửa sổ
        h = self._apply_window(h)
        
        # Chuẩn hóa
        h = h / np.sum(np.abs(h))
        return h

class BandPassFilter(FIRFilter):
    def __init__(self, sample_rate, low_cutoff, high_cutoff, filter_order=64):
        super().__init__(sample_rate, filter_order)
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        
    def design(self):
        """Thiết kế bộ lọc thông dải"""
        # Tính toán đáp ứng xung lý tưởng
        n = np.arange(self.filter_order)
        fc1 = self.low_cutoff / (self.sample_rate / 2)  # Chuẩn hóa tần số cắt thấp
        fc2 = self.high_cutoff / (self.sample_rate / 2)  # Chuẩn hóa tần số cắt cao
        
        # Kết hợp hai bộ lọc thông thấp
        h = np.sinc(2 * fc2 * (n - (self.filter_order - 1) / 2)) - \
            np.sinc(2 * fc1 * (n - (self.filter_order - 1) / 2))
        
        # Áp dụng cửa sổ
        h = self._apply_window(h)
        
        # Chuẩn hóa
        h = h / np.sum(np.abs(h))
        return h

class PeakingFilter(FIRFilter):
    def __init__(self, sample_rate, center_freq, gain_db, bandwidth, filter_order=64):
        super().__init__(sample_rate, filter_order)
        self.center_freq = center_freq
        self.gain_db = gain_db
        self.bandwidth = bandwidth
        
    def design(self):
        """Thiết kế bộ lọc peaking"""
        # Tính toán các tham số
        w0 = 2 * np.pi * self.center_freq / self.sample_rate
        bw = self.bandwidth / self.sample_rate
        gain = 10 ** (self.gain_db / 20)
        
        # Tính toán đáp ứng xung
        n = np.arange(self.filter_order)
        h = np.zeros(self.filter_order)
        
        # Thiết kế bộ lọc peaking
        for i in range(self.filter_order):
            if i == (self.filter_order - 1) // 2:
                h[i] = 1 + (gain - 1) * np.exp(-bw * np.pi)
            else:
                h[i] = (gain - 1) * np.sin(bw * np.pi * (i - (self.filter_order - 1) / 2)) / \
                       (np.pi * (i - (self.filter_order - 1) / 2))
        
        # Áp dụng cửa sổ
        h = self._apply_window(h)
        
        # Chuẩn hóa
        h = h / np.sum(np.abs(h))
        return h

    def process(self, x):
        """Xử lý tín hiệu với bộ lọc peaking"""
        h = self.design()
        return np.convolve(x, h, mode='same') 
