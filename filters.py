from transforms import DiscreteFourierTransform, InverseDiscreteFourierTransform


class LowPassFilter:
    def __init__(self, cutoff_freq, sample_rate):
        self.cutoff = cutoff_freq
        self.fs = sample_rate

    def apply(self, signal):
        N = len(signal)
        dft = DiscreteFourierTransform(signal)
        X = dft.transform()

        # Áp dụng LPF: chỉ giữ các tần số thấp hơn ngưỡng
        filtered_X = []
        for k in range(N):
            freq = k * self.fs / N
            if freq <= self.cutoff or freq >= (self.fs - self.cutoff):
                filtered_X.append(X[k])
            else:
                filtered_X.append(0j)

        idft = InverseDiscreteFourierTransform(filtered_X)
        return [x.real for x in idft.transform()]

class HighPassFilter:
    def __init__(self, cutoff_freq, sample_rate):
        self.cutoff = cutoff_freq
        self.fs = sample_rate

    def apply(self, signal):
        N = len(signal)
        dft = DiscreteFourierTransform(signal)
        X = dft.transform()

        # HPF: loại bỏ tần số thấp hơn ngưỡng
        filtered_X = []
        for k in range(N):
            freq = k * self.fs / N
            if freq >= self.cutoff and freq <= (self.fs - self.cutoff):
                filtered_X.append(X[k])
            else:
                filtered_X.append(0j)

        idft = InverseDiscreteFourierTransform(filtered_X)
        return [x.real for x in idft.transform()]

class BandPassFilter:
    def __init__(self, low_cut, high_cut, sample_rate):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.fs = sample_rate

    def apply(self, signal):
        N = len(signal)
        dft = DiscreteFourierTransform(signal)
        X = dft.transform()

        # BPF: chỉ giữ các tần số nằm giữa low_cut và high_cut
        filtered_X = []
        for k in range(N):
            freq = k * self.fs / N
            if self.low_cut <= freq <= self.high_cut or \
               (self.fs - self.high_cut <= freq <= self.fs - self.low_cut):
                filtered_X.append(X[k])
            else:
                filtered_X.append(0j)

        idft = InverseDiscreteFourierTransform(filtered_X)
        return [x.real for x in idft.transform()]
