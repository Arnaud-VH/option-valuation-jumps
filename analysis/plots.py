"""
File containing functions to generate plots needed for the report and analysis. 
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_option_price(X, u_final, K, save_path=None):
   """
   Plots the option price under the Merton-Model
   """
   S = np.exp(X)
   payoff = np.maximum(S - K, 0.0)
   fig, ax = plt.subplots(figsize=(7, 4))
   ax.plot(S, payoff,  '--', color='steelblue', label='Payoff $(t=T)$')
   ax.plot(S, u_final, '-',  color='darkorange', label='Option value $(t=0)$')
   ax.axvline(K, color='gray', linestyle=':', alpha=0.6, label=f'Strike $K={K}$')
   ax.set_xlim(0, 3 * K)
   ax.set_ylim(-0.05 * K, 2.5 * K)
   ax.set_xlabel('Asset price $S$')
   ax.set_ylabel('Option value')
   ax.set_title("European Call — Merton Jump-Diffusion")
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   if save_path:
      plt.savefig(save_path, dpi=150)
   plt.show()

def plot_convergence(results, save_path=None):
   """
   log-log plot of error vs h, with reference slope-2 line.
   """
   hs     = [r['h']     for r in results]
   errors = [r['error'] for r in results]

   fig, ax = plt.subplots(figsize=(7, 4))
   ax.loglog(hs, errors, 'o-', color='darkorange', label='Error at $x_K$')

   h0, e0 = hs[0], errors[0]
   ref = [e0 * (h / h0)**2 for h in hs]
   ax.loglog(hs, ref, '--', color='gray', label='$O(h^2)$ reference')

   ax.set_xlabel('Spatial step $h$')
   ax.set_ylabel('Error $|u^h(x_K) - u(x_K)|$')
   ax.set_title('Convergence of FD scheme')
   ax.legend()
   ax.grid(True, alpha=0.3, which='both')
   plt.tight_layout()
   if save_path:
      plt.savefig(save_path, dpi=150)
   plt.show()

def plot_error_profile(X, u_final, fn_analytical, model, params, save_path=None):
   """
   Plots the spatial error u_num - u_analytical across domain, recreating figure 1 from paper. 
   """
   S = np.exp(X)
   K = params['K']
   
   u_exact = np.array([fn_analytical(model=model, S0=s, K=K, T=params['T'], r=params['r'], n_terms=50) for s in S])

   error = u_final - u_exact
   fig,ax = plt.subplots(figsize=(7, 4))
   ax.plot(X, error, color='darkorange')
   ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
   ax.axvline(np.log(K), color='gray', linestyle=':', alpha=0.5, label=f'$x_K = \\ln K = {np.log(K):.2f}$')
   ax.set_xlabel('Log-price $x$')
   ax.set_ylabel('Error $u^h - u$')
   ax.set_title('Spatial error profile')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   if save_path:
      plt.savefig(save_path, dpi=150)
   plt.show()