import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def price_european_call_merton(K, T, r, sigma, lam, sigma_J, mu_J=0.0, M=20, N=20):
    """
    Prices a European Call option under Merton's jump-diffusion model using
    finite differences (BDF2 in time, composite trapezoidal for the integral).
    
    Reference: Almendral & Oosterlee (2005)
    
    Parameters:
    K       : Strike price
    T       : Time to maturity (years)
    r       : Risk-free interest rate
    sigma   : Continuous diffusion volatility
    lam     : Jump intensity (lambda)
    mu_J    : Mean of the jump size
    sigma_J : Standard deviation of the jump size (for Merton it is 0.0)
    M       : Half the number of spatial grid intervals
    N       : Number of time steps
    
    Returns:
    S       : Array of asset prices
    u_final : Array of option prices at t=0
    """
    # 1. Model compensator
    zeta = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # 2. Space-Time Grid Setup
    Xmax = np.log(8.0 * K)  # Truncation boundary
    h = Xmax / M            # Spatial step size
    k = T / N                 # Time step size
    
    # Spatial grid goes from -Xmax to Xmax
    n_nodes = 2 * M + 1
    X = np.linspace(-Xmax, Xmax, n_nodes)
    S = np.exp(X)

    # 3. Helper Functions
    def jump_density(y):
        """Gaussian jump size density f_m(y)."""
        variance = sigma_J**2
        normalization = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_J)
        return normalization * np.exp(-((y - mu_J)**2) / (2.0 * variance))

    def tail_correction(tau, xi):
        """Tail integral epsilon(tau, xi, Xmax)."""
        d1 = (xi - Xmax + sigma_J**2) / sigma_J
        d2 = (xi - Xmax) / sigma_J
        return np.exp(xi + 0.5 * sigma_J**2) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    # 4. Matrix C (Black-Scholes Differential Operator)
    C = np.zeros((n_nodes, n_nodes))
    alpha = (k * sigma**2) / (2.0 * h**2)
    beta = k * (r - 0.5 * sigma**2 - lam * zeta) / (2.0 * h)

    for i in range(1, n_nodes - 1):
        C[i, i-1] = -alpha - beta                # Lower diagonal
        C[i, i]   = 2.0 * alpha + (r + lam) * k  # Main diagonal
        C[i, i+1] = -alpha + beta                # Upper diagonal

    # 5. Matrix D (Jump Integral Operator)
    D = np.zeros((n_nodes, n_nodes))
    for i in range(1, n_nodes - 1):
        for j in range(n_nodes):
            dy = X[j] - X[i]
            gamma = -k * h * lam * jump_density(dy)
            
            # Composite trapezoidal rule weights (halved at boundaries)
            if j == 0 or j == n_nodes - 1:
                D[i, j] = gamma / 2.0
            else:
                D[i, j] = gamma

    # 6. Time Stepping (BDF2)
    u = np.zeros((N + 1, n_nodes))
    u[0, :] = np.maximum(S - K, 0.0)  # Payoff at maturity
    
    b = np.zeros(n_nodes)
    I = np.eye(n_nodes)

    for m in range(1, N + 1):
        tau_m = m * k
        
        # BDF2 weights (Implicit Euler for m=1)
        w0 = 1.0 if m == 1 else 1.5
        w1 = 1.0 if m == 1 else 2.0
        w2 = 0.0 if m == 1 else -0.5
        
        # Matrix A (used to solve for u)
        A_m = (w0 * I) + C + D
        
        # Vector b (interior)
        for i in range(1, n_nodes - 1):
            jump_tail = k * lam * tail_correction(tau_m, X[i])
            history = w1 * u[m-1, i] + w2 * u[m-2, i]
            b[i] = jump_tail + history
            
        # Boundary conditions
        b[0] = 0.0
        b[-1] = w0 * (np.exp(Xmax) - K * np.exp(-r * tau_m))
        
        # Solve the linear system
        u[m, :] = np.linalg.solve(A_m, b)

    return S, u[-1, :]



if __name__ == "__main__":
    K = 5.0
    S_grid, u_price = price_european_call_merton(
        K=K, T=1.0, r=0.02, sigma=0.3, 
        lam=0.01, mu_J=0.0, sigma_J=0.5, 
        M=20, N=20
    )
    
    payoff = np.maximum(S_grid - K, 0.0)
    
    # Plot with Payoff and Option Value
    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, payoff, label="Payoff (t=T)", linestyle='--')
    plt.plot(S_grid, u_price, label="Option Value/Price (t=0)")
    plt.xlabel("Asset Price (S)")
    plt.ylabel("Option Value")
    plt.title("European Call Option under Merton's Jump-Diffusion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()