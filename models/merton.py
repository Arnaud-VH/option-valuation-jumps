import numpy as np
from math import factorial, exp, log, sqrt
from models.black_scholes import bs_call

def merton_analytical(S0, K, T, r, sigma, lam, mu_J, sigma_J, n_terms=5):
   """
   Analytical price for a European call under Merton's jump-diffusion model. 

   Parameters:
   S0       : Initial asset price
   K        : Strike price
   T        : time to maturity
   r        : risk-free rate
   sigma    : diffusion volatility
   lam      : jump intensity
   mu_J     : mean of log jump size
   sigma_J  : standard deviation of log jump size
   n_terms  : number of series terms
   
   Returns:
   option price
   """
   zeta = exp(mu_J + 0.5 * sigma_J**2) - 1.0
   lam_prime = lam * (1.0 + zeta)
   S0 = np.asarray(S0)
   price = np.zeros_like(S0, dtype=float)

   for m in range(n_terms):
      poisson_weight = (exp(-lam_prime * T) * (lam_prime * T)**m / factorial(m))
      sigma_m = sqrt(sigma**2 + m * sigma_J**2 / T)
      r_m = r - lam * zeta + m * log(1.0 + zeta) / T
      
      bs_price = bs_call(S0, K, T, r_m, sigma_m)

      price += poisson_weight * bs_price
   
   return price
