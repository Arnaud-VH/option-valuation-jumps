"""
Main that runs numerical expirements and produces plots. 
"""

import numpy as np
from model_config import (KOU_CONVERGENCE_CONFIGS, MERTON_PARAMS, KOU_PARAMS, KOU_MODEL, CONVERGENCE_CONFIGS, X_STAR, MERTON_MODEL)
from solvers.fd_solver import solve
from analysis.convergence import run_convergence_analysis, print_convergence_table
from analysis.plots import plot_convergence, plot_option_price, plot_error_profile
from models.merton import merton_analytical
from models.kou import kou_analytical


def main():
   #Option Price plot
   print("---Running FD solver for option price plot---\n")
   X, u_final = solve(model=MERTON_MODEL, K=MERTON_PARAMS['K'], T=MERTON_PARAMS['T'], r=MERTON_PARAMS['r'], M=64, N=10)
   plot_option_price(X, u_final, K=MERTON_PARAMS['K'])

   #Convergence study
   print("---Running convergence study---\n")
   analytical_merton = merton_analytical(S0=MERTON_PARAMS['K'],**MERTON_PARAMS)
   merton_results = run_convergence_analysis(model=MERTON_MODEL, K=MERTON_PARAMS['K'], T=MERTON_PARAMS['T'], r=MERTON_PARAMS['r'], analytical_price=analytical_merton, configs = CONVERGENCE_CONFIGS, x_star = X_STAR)
   print_convergence_table(analytical_merton, merton_results)
   plot_convergence(merton_results)

   #Spatial error
   print("---Generating spatial error profile")
   plot_error_profile(X, u_final, merton_analytical, MERTON_PARAMS)


   #Kou's option price
   print("---Running FD solver for option price plot---\n")
   X, u_final_kou = solve(model=KOU_MODEL, K=KOU_PARAMS['K'], T=KOU_PARAMS['T'], r=KOU_PARAMS['r'], M=64, N=10, x_star=6.0)
   plot_option_price(X, u_final_kou, K=KOU_PARAMS['K'])

   #Kou's convergence study
   print("--- Kou: convergence study ---\n")
   analytical_kou = kou_analytical(
      x_K    = np.log(KOU_PARAMS['K']),
      tau    = KOU_PARAMS['T'],
      sigma  = KOU_PARAMS['sigma'],
      lam    = KOU_PARAMS['lam'],
      p      = KOU_PARAMS['p'],
      alpha1 = KOU_PARAMS['alpha1'],
      alpha2 = KOU_PARAMS['alpha2'],
      zeta   = KOU_MODEL.compensator(),
      r      = KOU_PARAMS['r'],
    )
   
   kou_results = run_convergence_analysis(
      model            = KOU_MODEL,
      K                = KOU_PARAMS['K'],
      T                = KOU_PARAMS['T'],
      r                = KOU_PARAMS['r'],
      analytical_price = analytical_kou,
      configs          = KOU_CONVERGENCE_CONFIGS,
      x_star           = 6.0,
   )
   print_convergence_table(analytical_kou, kou_results)
   plot_convergence(kou_results)
   
if __name__ == "__main__":
   main()