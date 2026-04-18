import numpy as np
from models.merton import merton_analytical
from solvers.fd_solver import solve

def run_convergence_analysis(model, K, T, r, analytical_price, configs, x_star):
   """
   Run the FD solver at multiple grid resolutions and compute the pointwise error at x_K = log(K) against the analytical solution. 

   Parameters:
   params : dict of model parameters (K, T, r, sigma, lam, mu_J, sigma_J)
   configs : list of (M, N) tuples
   x_star : truncation point (for reporting h)

   Returns:
   results: list of dicts with keys: n, k, h, fd_price, error, ratio
   """
   x_K = np.log(K)
   results = []
   prev_error = None

   for M, N in configs:
      n = 2 * M + 1
      h = (2.0 * x_star) / (n - 1)
      k = T / N
      X, u_final = solve(model=model, K=K, T=T, r=r, M=M, N=N, x_star=x_star)

      #find the grid point closest to x_K
      index = np.argmin(np.abs(X - x_K))
      fd_price = u_final[index]
      error = abs(fd_price - analytical_price)
      ratio = prev_error / error if prev_error is not None else float('nan')

      results.append(dict(n=n, k=k, h=h, fd_price=fd_price, error=error, ratio=ratio, ))

      prev_error = error
   
   return results

def print_convergence_table(analytical, results):
   """
   Function to display the convergence results more clearly
   """
   print(f"Analytical price at x_K: {analytical:.9f}\n")
   print(f"{'n':>6} {'k':>8} {'FD price':>12} {'error':>12} {'ratio':>8}")
   print("-" * 52)
   for r in results:
      ratio_str = f"{r['ratio']:8.2f}" if not np.isnan(r['ratio']) else "     ---"
      print(f"{r['n']:>6} {r['k']:>8.4f} {r['fd_price']:>12.8f} " f"{r['error']:>12.2e} {ratio_str}")