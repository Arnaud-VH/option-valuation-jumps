import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from model_config import MERTON_MODEL, MERTON_PARAMS
from solvers.fd_solver import solve
from solvers.fft_solver import solve_fft

X1, u1 = solve(model=MERTON_MODEL, K=MERTON_PARAMS['K'], T=MERTON_PARAMS['T'], r=MERTON_PARAMS['r'], M=64, N=10)
X2, u2 = solve_fft(model=MERTON_MODEL, K=MERTON_PARAMS['K'], T=MERTON_PARAMS['T'], r=MERTON_PARAMS['r'], M=64, N=10)

print(f"Max difference: {np.max(np.abs(u1 - u2)):.2e}")