from solvers.fd_solver import solve
from models.merton import MertonModel
from models.kou import KouModel
from models.black_scholes import bs_call
import numpy as np

bs = bs_call(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)

# Merton FD with lam=0
merton_model = MertonModel(sigma=0.2, lam=0.0, mu_J=0.0, sigma_J=0.5)
X, u_merton = solve(model=merton_model, K=1.0, T=1.0, r=0.0,M=128, N=20, x_star=4.0)
idx = np.argmin(np.abs(X - 0.0))
merton_fd = u_merton[idx]

# Kou FD with lam=0
kou_model = KouModel(sigma=0.2, lam=0.0, p=0.5, alpha1=3.0, alpha2=2.0)
X, u_kou = solve(model=kou_model, K=1.0, T=1.0, r=0.0,M=128, N=20, x_star=4.0)
kou_fd = u_kou[idx]

print(f"Black-Scholes : {bs:.8f}")
print(f"Merton FD : {merton_fd:.8f} diff={abs(bs-merton_fd):.2e}")
print(f"Kou FD : {kou_fd:.8f} diff={abs(bs-kou_fd):.2e}")

#Run from project root with python -m tests.test_fd_scheme