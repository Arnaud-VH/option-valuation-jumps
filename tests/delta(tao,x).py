from models.merton import merton_analytical
from merton_numerical_class import MertonNumericalClass
import numpy as np
import matplotlib.pyplot as plt
n = 129
M = int((n-1)/2)
k = 0.1
K = 1.0
Xmax = 4
T = 1.0
def plot_delta_tao_x(x=4, tao=0.2):
    Xmax = x
    N = int(T/k)
    X_grid = np.linspace(-Xmax,Xmax,n)
    S_grid = np.exp(X_grid)
    # analytical
    analytical_tao = np.zeros(n)
    for i in range(n):
        analytical_tao[i] = merton_analytical(
            S0=S_grid[i], 
            K=K, 
            T=tao, 
            r=0.0,
            sigma=0.2, 
            lam=0.1, 
            mu_J=0.0, 
            sigma_J=0.5
        )


    mnc = MertonNumericalClass(
        K=K, 
        T=T, 
        r=0.0, 
        sigma=0.2, 
        lam=0.1, 
        sigma_J=0.5, 
        mu_J=0.0,
        M=M, 
        N=N
    )
    numerical_tao = mnc.getPriceTao(0.2)
    X_grid = mnc.getGrid_X()
    delta = analytical_tao - numerical_tao

    # plot

    plt.plot(X_grid, delta)
    plt.title(f"Error at tao={tao} (analytic - numeric)")
    plt.xlabel("x = ln(S/K)")
    plt.show()

def plotAnalytical(tao):
    X_grid = np.linspace(-Xmax,Xmax,n)
    S_grid = np.exp(X_grid)
    # analytical
    analytical_tao = np.zeros(n)
    for i in range(n):
        analytical_tao[i] = merton_analytical(
            S0=S_grid[i], 
            K=K, 
            T=tao, 
            r=0.0,
            sigma=0.2, 
            lam=0.1, 
            mu_J=0.0, 
            sigma_J=0.5
        )
    plt.plot(X_grid, analytical_tao)
    plt.title(f"analytical price at tao={tao} ")
    plt.xlabel("x = ln(S/K)")
    plt.show()
    return X_grid, analytical_tao

def plotNumerical(tao):
    mnc = MertonNumericalClass(
        K=K, 
        T=T, 
        r=0.0, 
        sigma=0.2, 
        lam=0.1, 
        sigma_J=0.5, 
        mu_J=0.0,
        M=M, 
        N=int(T/k),
        x_star=4
    )
    numerical_tao = mnc.getPriceTao(0.2)
    X_grid=mnc.getGrid_X()
    # plot
    plt.plot(X_grid, numerical_tao)
    plt.title(f"Numerical Option price at tao={tao}")
    plt.xlabel("x = ln(S/K)")
    plt.show()
    return X_grid, mnc.numerical_tao

if __name__ == "__main__":
    plot_delta_tao_x(4,0.2)
    plotAnalytical(0.2)
    plotNumerical(0.2)