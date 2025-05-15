#Tạo các phép biến đổi

import numpy as np

# 1. **Discrete Fourier Transform (DFT)**
def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
