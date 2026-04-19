import numpy as np
from math import factorial, exp, log, sqrt
from scipy.stats import norm

from models.base import JumpDiffusionModel
from models.black_scholes import bs_call

class MertonModel(JumpDiffusionModel):
   """
   Merton jump-diffusion model with Gaussian jump sizes. 

   Parameters:
   sigma - diffusion volatility
   lam - jump intensity
   mu_J - mean of log jump size
   sigma_J - std dev of log jump size
   """

   def __init__(self, sigma, lam, mu_J, sigma_J):
      self._sigma = sigma
      self._lam = lam
      self.mu_J = mu_J
      self.sigma_J = sigma_J
   
   @property
   def sigma(self):
      return self._sigma

   @property
   def lam(self):
      return self._lam
   
   def _compute_compensator(self):
      return exp(self.mu_J + 0.5 * self.sigma_J**2) - 1.0
   
   def jump_density(self, y):
      y = np.asarray(y, dtype=float)
      return (1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_J) * np.exp(-((y - self.mu_J)**2) / (2.0 * self.sigma_J**2)))
   
   def tail_correction(self, tau, xi, x_star, K, r):
      d1 = (xi - x_star + self.sigma_J**2) / self.sigma_J
      d2 = (xi - x_star) / self.sigma_J
      return (np.exp(xi + 0.5 * self.sigma_J**2) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)) 
   
   def right_boundary(self, x_star, K, r, tau):
      return np.exp(x_star) - K * np.exp(-r * tau)

def merton_analytical(model, S0, K, T, r, n_terms=5):
   """
   Analytical price for a European call under Merton's jump-diffusion model. 

   Parameters:
   model    : MertonModel instance
   S0       : Initial asset price
   K        : Strike price
   T        : Time to maturity
   r        : Risk-free rate
   n_terms  : Number of series terms
   
   Returns:
   option price
   """
   lam_prime = model.lam * (1.0 + model.zeta)
   S0 = np.asarray(S0)
   price = np.zeros_like(S0, dtype=float)

   for m in range(n_terms):
      poisson_weight = exp(-lam_prime * T) * (lam_prime * T)**m / factorial(m)
      sigma_m = sqrt(model.sigma**2 + m * model.sigma_J**2 / T)
      r_m = r - model.lam * model.zeta + m * log(1.0 + model.zeta) / T
      
      price += poisson_weight * bs_call(S0, K, T, r_m, sigma_m)
   
   return price
