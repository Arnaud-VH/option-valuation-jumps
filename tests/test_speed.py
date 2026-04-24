"""
Speed comparison between fd_solver (dense) and fft_solver (iterative splitting).
Run from project root: python -m tests.test_speed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from model_config import MERTON_MODEL, MERTON_PARAMS
from solvers.fd_solver import solve
from solvers.fft_solver import solve_fft

CONFIGS = [
    (32,   5),
    (64,   10),
    (128,  20),
    (256,  40),
    (512,  80),
    (1024, 160),
    (2048, 320),
]

K = MERTON_PARAMS['K']
T = MERTON_PARAMS['T']
r = MERTON_PARAMS['r']

print(f"{'M':>6}  {'n':>6}  {'Dense (s)':>12}  {'FFT (s)':>12}  {'Speedup':>10}  {'Max diff':>12}")
print("-" * 70)

for M, N in CONFIGS:
    n = 2 * M + 1

    t0 = time.perf_counter()
    X1, u1 = solve(model=MERTON_MODEL, K=K, T=T, r=r, M=M, N=N)
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    X2, u2 = solve_fft(model=MERTON_MODEL, K=K, T=T, r=r, M=M, N=N)
    t_fft = time.perf_counter() - t0

    speedup = t_dense / t_fft
    diff = np.max(np.abs(u1 - u2))

    print(f"{M:>6}  {n:>6}  {t_dense:>12.4f}  {t_fft:>12.4f}  {speedup:>10.2f}x  {diff:>12.2e}")    