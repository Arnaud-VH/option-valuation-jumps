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
    sigma_J : Standard deviation of the jump size
    mu_J    : Mean of the jump size (default 0.0)
    M       : Half the number of spatial grid intervals
    N       : Number of time intervals
    
    Returns:
    S       : Array of asset prices
    u_final : Array of option prices at t=0
    """
    # 1. Model compensator
    zeta = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0

    # 2. Space-Time Grid Setup
    x_star = np.log(8.0 * K)  # Truncation boundary [-x*, x*]
    
    n = 2 * M + 1             # Total number of spatial points
    q = N + 1                 # Total number of time steps
    
    # Grid step sizes
    h = (2.0 * x_star) / (n - 1)  
    k = T / (q - 1)               
    
    # Spatial grid (X[0] is -x* and X[n-1] is x*)
    X = np.linspace(-x_star, x_star, n)
    S = np.exp(X)

    # 3. Helper Functions
    def jump_density(y):
        """Gaussian jump size density f_m(y)."""
        variance = sigma_J**2
        normalization = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_J)
        return normalization * np.exp(-((y - mu_J)**2) / (2.0 * variance))

    def tail_correction(tau, xi):
        """Analytical tail integral epsilon(tau, xi, x_star)."""
        d1 = (xi - x_star + sigma_J**2) / sigma_J
        d2 = (xi - x_star) / sigma_J
        return np.exp(xi + 0.5 * sigma_J**2) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    # 4. Matrix C (Black-Scholes Differential Operator)
    C = np.zeros((n, n))
    alpha = (k * sigma**2) / (2.0 * h**2)
    beta = k * (r - 0.5 * sigma**2 - lam * zeta) / (2.0 * h)

    # Loop through strictly interior nodes
    for i in range(1, n - 1):
        C[i, i-1] = -alpha - beta                # Lower diagonal
        C[i, i]   = 2.0 * alpha + (r + lam) * k  # Main diagonal
        C[i, i+1] = -alpha + beta                # Upper diagonal

    # 5. Matrix D (Jump Integral Operator)
    D = np.zeros((n, n))
    for i in range(1, n - 1):
        for j in range(n):
            dy = X[j] - X[i]
            trapezoid = -k * h * lam * jump_density(dy)
            
            # Composite trapezoidal rule weights (halved at boundaries)
            if j == 0 or j == n - 1:
                D[i, j] = trapezoid / 2.0
            else:
                D[i, j] = trapezoid

    # 6. Time Stepping (BDF2)
    u = np.zeros((q, n))
    u[0, :] = np.maximum(S - K, 0.0)  # Payoff at maturity (tau=0)
    
    b = np.zeros(n)
    I = np.eye(n)

    # Step through time (tau_1 to tau_q-1)
    for m in range(1, q):
        tau_m = m * k
        
        # BDF2 weights (Implicit Euler for m=1, BDF2 for m>=2)
        w0 = 1.0 if m == 1 else 1.5
        w1 = 1.0 if m == 1 else 2.0
        w2 = 0.0 if m == 1 else -0.5
        
        # Matrix A (used to solve for u^m)
        A_m = (w0 * I) + C + D
        
        # Vector b (interior)
        for i in range(1, n - 1):
            jump_tail = k * lam * tail_correction(tau_m, X[i])
            history = w1 * u[m-1, i] + w2 * u[m-2, i]
            b[i] = jump_tail + history
            
        # Dirichlet Boundary conditions
        b[0] = 0.0
        b[-1] = w0 * (np.exp(x_star) - K * np.exp(-r * tau_m))
        
        # Solve the linear system A * u^m = b^m
        u[m, :] = np.linalg.solve(A_m, b)

    return S, u[-1, :]


if __name__ == "__main__":
    K = 5.0
    S_grid, u_price = price_european_call_merton(
        K=K, T=1.0, r=0.02, sigma=0.3, 
        lam=0.01, sigma_J=0.5, mu_J=0.0, 
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