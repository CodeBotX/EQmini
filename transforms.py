import cmath


class DiscreteFourierTransform:
    def __init__(self, xn):
        """
        xn: list hoặc iterable các mẫu tín hiệu (real hoặc complex)
        """
        self.xn = xn
        self.N = len(xn)

    def transform(self):
        """
        Tính DFT: X[k] = sum_{n=0}^{N-1} xn[n] * exp(-j*2*pi*k*n/N)
        Trả về danh sách X (complex)
        """
        N = self.N
        X = []
        for k in range(N):
            total = 0j
            for n in range(N):
                angle = -2j * cmath.pi * k * n / N
                total += self.xn[n] * cmath.exp(angle)
            X.append(total)
        return X

class InverseDiscreteFourierTransform:
    def __init__(self, X):
        """
        X: list hoặc iterable các giá trị phức của phổ
        """
        self.X = X
        self.N = len(X)

    def transform(self):
        """
        Tính IDFT: xn[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(j*2*pi*k*n/N)
        Trả về danh sách xn (complex, có thể gần như thực nếu input chuẩn)
        """
        N = self.N
        xn = []
        for n in range(N):
            total = 0j
            for k in range(N):
                angle = 2j * cmath.pi * k * n / N
                total += self.X[k] * cmath.exp(angle)
            xn.append(total / N)
        return xn

class ZTransform:
    def __init__(self, xn):
        """
        xn: list hoặc iterable các mẫu tín hiệu (real hoặc complex)
        """
        self.xn = xn

    def transform(self, z):
        """
        Tính Z-transform tại điểm z (complex):
        X(z) = sum_{n=0}^{N-1} xn[n] * z^{-n}
        Trả về giá trị complex
        """
        total = 0j
        for n, val in enumerate(self.xn):
            total += val * (z ** (-n))
        return total
