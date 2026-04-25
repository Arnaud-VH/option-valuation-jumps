"""
Numerical experiments: option pricing, convergence studies, and error profiles.
"""

import numpy as np
from model_config import (
    KOU_CONVERGENCE_CONFIGS, MERTON_PARAMS, KOU_PARAMS,
    KOU_MODEL, CONVERGENCE_CONFIGS, X_STAR, MERTON_MODEL,
)
from solvers.fd_solver import solve
from solvers.fft_solver import solve_fft
from analysis.convergence import run_convergence_analysis, print_convergence_table
from analysis.plots import plot_convergence, plot_option_price, plot_error_profile
from models.merton import merton_analytical
from models.kou import kou_analytical


def _run_merton():
    #Option Price plot
    print("--- Merton: Running FD solver for option price plot ---\n")
    X, u_final = solve_fft(
        model=MERTON_MODEL, 
        K=MERTON_PARAMS['K'], 
        T=MERTON_PARAMS['T'], 
        r=MERTON_PARAMS['r'], 
        M=64, 
        N=10,
        )
    plot_option_price(X, u_final, K=MERTON_PARAMS['K'])

    #Convergence study
    print("--- Merton: Running convergence study ---\n")
    analytical = merton_analytical(
        model=MERTON_MODEL,
        S0=MERTON_PARAMS['K'],
        K=MERTON_PARAMS['K'],
        T=MERTON_PARAMS['T'],
        r=MERTON_PARAMS['r'],
    )
    results = run_convergence_analysis(
        model=MERTON_MODEL,
        K=MERTON_PARAMS['K'],
        T=MERTON_PARAMS['T'],
        r=MERTON_PARAMS['r'],
        analytical_price=analytical,
        configs=CONVERGENCE_CONFIGS,
        x_star=X_STAR,
    )
    print_convergence_table(analytical, results)
    plot_convergence(results, "Convergence of Merton FD Scheme")

    #Spatial error
    print("--- Merton: Generating spatial error profile ---\n")
    plot_error_profile(X, u_final, merton_analytical, MERTON_MODEL, MERTON_PARAMS, "Merton Spatial Error Profile")


def _run_kou():
    #Kou's option price
    print("--- Kou: Running FD solver for option price plot ---\n")
    X, u_final = solve(
        model=KOU_MODEL,
        K=KOU_PARAMS['K'],
        T=KOU_PARAMS['T'],
        r=KOU_PARAMS['r'],
        M=64,
        N=10,
        x_star=6.0,
    )
    plot_option_price(X, u_final, K=KOU_PARAMS['K'])

    #Kou's convergence study
    print("--- Kou: convergence study ---\n")
    analytical = kou_analytical(
        model=KOU_MODEL,
        x_K=np.log(KOU_PARAMS['K']),
        tau=KOU_PARAMS['T'],
        r=KOU_PARAMS['r'],
    )
    results = run_convergence_analysis(
        model=KOU_MODEL,
        K=KOU_PARAMS['K'],
        T=KOU_PARAMS['T'],
        r=KOU_PARAMS['r'],
        analytical_price=analytical,
        configs=KOU_CONVERGENCE_CONFIGS,
        x_star=6.0,
    )
    print_convergence_table(analytical, results)
    plot_convergence(results, "Convergence of Kou FD Scheme")


def run_all():
    _run_merton()
    _run_kou()