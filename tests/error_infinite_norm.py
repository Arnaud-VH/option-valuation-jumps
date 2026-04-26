from models.merton import MertonModel, merton_analytical
from solvers.fd_solver import solve
import numpy as np

def compute_error_at_money(M=64, N=10, T=1.0, K=1.0, r=0.0):
    """
    Compares the FD numerical price to the analytical price at x=0 (S=K).
    Returns the difference.
    """
    model = MertonModel(sigma=0.2, lam=0.1, mu_J=0.0, sigma_J=0.5)

    analytical = merton_analytical(model=model, S0=K, K=K, T=T, r=r)

    X, u_final = solve(model=model, K=K, T=T, r=r, M=M, N=N)

    # Find the grid point closest to x=0 (at the money)
    atm_index = np.argmin(np.abs(X))
    numerical = u_final[atm_index]

    error = analytical - numerical
    print(f"Analytical:  {analytical:.10f}")
    print(f"Numerical:   {numerical:.10f}")
    print(f"Error:       {error:.2e}")
    return error
    
def compute_infinite_norm_error(M=64, N=10, T=1.0, K=1.0, r=0.0):
    """
    Returns the infinite norm
    """
    model = MertonModel(sigma=0.2, lam=0.1, mu_J=0.0, sigma_J=0.5)
    
    X, u_final = solve(model=model, K=K, T=T, r=r, M=M, N=N)
    S = np.exp(X)
    
    analytical = merton_analytical(model=model, S0=S, K=K, T=T, r=r)
    error_inf = np.max(np.abs(analytical[1:-1] - u_final[1:-1]))
    print(f"Error_inf:   {error_inf:.2e}")
    return error_inf

compute_error_at_money()
compute_infinite_norm_error()

#From root --> python -m tests.error_infinite_norm
