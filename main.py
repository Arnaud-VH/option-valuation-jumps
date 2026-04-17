"""
Main that runs numerical expirements and produces plots. 
"""

from model_config import MERTON_PARAMS, CONVERGENCE_CONFIGS, X_STAR
from solvers.fd_solver import solve
from analysis.convergence import run_convergence_analysis, print_convergence_table
from analysis.plots import plot_convergence, plot_option_price, plot_error_profile
from models.merton import merton_analytical

def main():
   #Option Price plot
   print("---Running FD solver for option price plot---\n")
   X, u_final = solve(M=64, N=10, **MERTON_PARAMS)
   plot_option_price(X, u_final, K=MERTON_PARAMS['K'])

   #Convergence study
   print("---Running convergence study---\n")
   analytical, results = run_convergence_analysis(params = MERTON_PARAMS, configs= CONVERGENCE_CONFIGS, x_star=X_STAR)
   print_convergence_table(analytical=analytical, results=results)
   plot_convergence(results)

   #Spatial error
   print("---Generating spatial error profile")
   plot_error_profile(X, u_final, merton_analytical, MERTON_PARAMS)

if __name__ == "__main__":
   main()