"""
BDF2 finite difference solver with FFT-accelerated iterative splitting.
Follows Section 6 of Almendral & Oosterlee (2005).
"""

import numpy as np
from scipy.linalg import solve_banded
from models.base import JumpDiffusionModel


def solve_fft(model: JumpDiffusionModel, K, T, r, M=64, N=20, x_star=4.0, tol=1e-8, max_iter=100):
    """
    Prices a European Call option under a jump-diffusion model.
    Same interface as fd_solver.solve(), with two extra optional parameters.

    Parameters:
        model    : JumpDiffusionModel instance (Merton or Kou)
        K        : Strike price
        T        : Time to maturity (years)
        r        : Risk-free interest rate
        M        : Half the number of spatial grid intervals
        N        : Number of time intervals
        x_star   : Truncation boundary
        tol      : Convergence tolerance for splitting iteration (eq. 67)
        max_iter : Maximum iterations per time step

    Returns:
        X       : Array of log-prices
        u_final : Array of option prices at t=0
    """
    zeta = model.zeta

    # Grid setup 
    n = 2 * M + 1
    q = N + 1
    h = (2.0 * x_star) / (n - 1)
    k = T / (q - 1)
    X = np.linspace(-x_star, x_star, n)
    S = np.exp(X)

    # FD coefficients
    alpha = (k * model.sigma**2) / (2.0 * h**2)
    beta = k * (r - 0.5 * model.sigma**2 - model.lam * zeta) / (2.0 * h)

    # Jump density at all shifts d*h, for d = -(n-1)...(n-1)
    # length 2n-1
    shifts = np.arange(-(n - 1), n) * h  
    f_all = model.jump_density(shifts)
    # f_all[d + (n-1)] = f(d*h)

    # Build Q: tridiagonal part of A = w0*I + C + D 
    # C is tridiagonal from the diffusion operator. D is dense from the jump integral, but its three main diagonals

    # D's tridiagonal values
    d_m1 = -k * h * model.lam * f_all[n - 2]  
    d_0 = -k * h * model.lam * f_all[n - 1]   
    d_p1 = -k * h * model.lam * f_all[n]  

    # Trapezoidal half-weight at boundaries (j=0 or j=n-1)
    d_m1_boundary = d_m1 / 2.0
    d_p1_boundary = d_p1 / 2.0

    q_lower = np.zeros(n) # Q[i, i-1] (without w0)
    q_main = np.zeros(n) # Q[i, i] (without w0)
    q_upper = np.zeros(n) # Q[i, i+1] (without w0)

    for i in range(1, n - 1):
        # C contributions
        q_lower[i] = (-alpha + beta)
        q_main[i] = (2.0 * alpha + (r + model.lam) * k)
        q_upper[i] = (-alpha - beta)

        # D tridiagonal contributions
        q_main[i] += d_0

        if i == 1:
            q_lower[i] += d_m1_boundary # j=0 is boundary: half weight
        else:
            q_lower[i] += d_m1

        if i == n - 2:
            q_upper[i] += d_p1_boundary # j=n-1 is boundary: half weight
        else:
            q_upper[i] += d_p1

    # Building R, off-tridiagonal part of -D, as a Toeplitz matrix
    r_toep = k * h * model.lam * f_all.copy()

    # Zero out the three main diagonals (these are in Q, not R)
    r_toep[n - 2] = 0.0  
    r_toep[n - 1] = 0.0  
    r_toep[n] = 0.0      

    # Circulant embedding
    circ_col = np.concatenate([
        r_toep[n - 1::-1],
        r_toep[2 * n - 2:n - 1:-1],
    ])
    # Precompute FFT of circulant column (done once, reused every iteration)
    r_hat = np.fft.fft(circ_col)

    # Boundary trapezoidal correction vectors
    f_corr_left = np.zeros(n) # correction for j = 0
    f_corr_right = np.zeros(n) # correction for j = n-1
    for i in range(1, n - 1):
        if abs(i) > 1:         
            f_corr_left[i] = 0.5 * k * h * model.lam * f_all[-i + (n - 1)]
        if abs(i - (n - 1)) > 1:
            f_corr_right[i] = 0.5 * k * h * model.lam * f_all[(n - 1) - i + (n - 1)]

    # Time stepping
    u = np.zeros((q, n))
    u[0, :] = np.maximum(S - K, 0.0)

    for m in range(1, q):
        tau_m = m * k

        # BDF weights
        if m == 1:
            w0, w1, w2 = 1.0, 1.0, 0.0
        else:
            w0, w1, w2 = 1.5, 2.0, -0.5

        # Banded matrix for Q (add w0 to diagonal)
        # Format for solve_banded with (lower=1, upper=1):
        ab = np.zeros((3, n))
        ab[0, 1:] = q_upper[:-1]
        ab[1, :] = w0 + q_main
        ab[2, :-1] = q_lower[1:]

        # Right-hand side b^m
        b = np.zeros(n)
        for i in range(1, n - 1):
            u_prev2 = u[m - 2, i] if m >= 2 else 0.0
            tail = k * model.lam * model.tail_correction(tau_m, X[i], x_star, K, r)
            history = w1 * u[m - 1, i] + w2 * u_prev2
            b[i] = tail + history

        # C and D only fill interior rows, so boundary rows reduce to w0.
        b[0] = 0.0
        b[-1] = w0 * model.right_boundary(x_star, K, r, tau_m)

        # Splitting iteration
        v = np.zeros(n)

        for iteration in range(max_iter):
            # Toeplitz product R*v via FFT 
            v_padded = np.zeros(2 * n - 1)
            v_padded[:n] = v
            Rv = np.real(np.fft.ifft(r_hat * np.fft.fft(v_padded)))[:n]

            # Subtract boundary trapezoidal correction
            Rv -= f_corr_left * v[0] + f_corr_right * v[n - 1]

            # Boundary rows of R are zero
            Rv[0] = 0.0
            Rv[-1] = 0.0

            # Solve tridiagonal system: Q v_new = Rv + b
            v_new = solve_banded((1, 1), ab, Rv + b)

            # Convergence check 
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new

        u[m, :] = v

    return X, u[-1, :]