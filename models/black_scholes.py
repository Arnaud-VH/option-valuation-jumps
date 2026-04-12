import numpy as np
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
   """
   Black-Scholes price for a European call option. 

   Parameters:
   S        : Current asset price
   K        : Strike price
   T        : Time to maturity (years)
   r        : Risk-free rate
   sigma    : volatility

   Returns:
   Call option price
   """
   if (T <= 0 or sigma <= 0):
      return np.maximum(S - K, 0)

   d1 = (np.log(S / K) + (r + 0.5  * sigma**2)  * T) / (sigma * np.sqrt(T))
   d2 = d1 - sigma * np.sqrt(T)

   return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

