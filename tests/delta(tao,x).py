from models.merton import merton_analytical
from discretization_function import price_european_call_merton
import numpy as np
import matplotlib.pyplot as plt

T = 1.0
M = 64
N = 10

# discretization
S_grid, numerical_T = price_european_call_merton(
    K=1.0, 
    T=T, 
    r=0.0, 
    sigma=0.2, 
    lam=0.1, 
    sigma_J=0.5, 
    mu_J=0.0,
    M=M, 
    N=N
)

# analytical
analytical_T = np.zeros_like(S_grid)
for i in range(len(S_grid)):
    analytical_T[i] = merton_analytical(
        S0=S_grid[i], 
        K=1.0, 
        T=1.0, 
        r=0.0,
        sigma=0.2, 
        lam=0.1, 
        mu_J=0.0, 
        sigma_J=0.5
    )

delta = analytical_T - numerical_T

# plot

plt.plot(np.log(S_grid/1.0), delta, linewidth=2)
plt.title("Error at T=1 (analytic - numeric)")
plt.xlabel("x = ln(S/K)")
plt.grid(True)
plt.show()