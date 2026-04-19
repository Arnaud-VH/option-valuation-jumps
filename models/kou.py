"""
Kou's double exponential jump-diffusion model.
"""

import numpy as np
from math import exp

from models.base import JumpDiffusionModel

class KouModel(JumpDiffusionModel):
   """
   Kou double exponential jump diffusion model. 
   """

   def __init__(self, sigma, lam, p, alpha1, alpha2):
      self._sigma = sigma
      self._lam = lam
      self.p = p
      self.q = 1.0 - p
      self.alpha1 = alpha1
      self.alpha2 = alpha2

   @property
   def sigma(self):
      return self._sigma

   @property
   def lam(self):
      return self._lam
   
   def _compute_compensator(self):
      """
      zeta = E[e^Y - 1] for double exponential y
      referencing eq 9 in the paper. 
      """
      return (self.p * self.alpha1 / (self.alpha1 - 1.0) + self.q * self.alpha2 / (self.alpha2 + 1.0) - 1.0)
   
   def jump_density(self, y):
      """
      Double exponential density. 
      referencing eq 8 in the paper. 
      """
      y = np.asarray(y, dtype=float)
      return np.where(y >= 0, self.p * self.alpha1 * np.exp(-self.alpha1 * y), self.q * self.alpha2 * np.exp(self.alpha2 * y))
   
   def tail_correction(self, tau, xi, x_star, K, r):
      delta = x_star - xi
      term_exp = (self.p * self.alpha1 / (self.alpha1 -1.0) * np.exp(xi - (self.alpha1 - 1.0) * delta))
      term_K = (self.p * np.exp(-self.alpha1 * delta) * K * np.exp(-r * tau))
      return term_exp - term_K
   
   def right_boundary(self, x_star, K, r, tau):
      return np.exp(x_star) - K * np.exp(-r * tau)
   
def kou_analytical(model, x_K, tau, r=0.0, n_points=512, rho=1.5):
   """
   European call price under Kou's model following the Almendral & Oosterlee (2005) paper. 
   Follows with Equations (65) and (66) of the paper. 
   """
   q = 1.0 - model.p

   def levy_exponent(z):
      """
      Levy-Khintchine exponent from equation (66).
      """
      diffusion = -0.5 * model.sigma**2 * z**2
      #Use of imaginary number 1j here. 
      drift = -(model.lam * model.zeta + 0.5 * model.sigma**2) * 1j * z
      jump = model.lam * ((model.p * model.alpha1) / (model.alpha1 - 1j * z) + (q * model.alpha2) / (model.alpha2 + 1j * z) - 1.0)
      return diffusion + drift + jump
   
   #Use Simpon's rule for the integral as mentioned in paper. 
   y_max = 30.0
   y = np.linspace(0, y_max, n_points)
   dy = y[1] - y[0]
   z = y + rho * 1j
   integrand = np.real(np.exp(-1j * z * x_K + tau * levy_exponent(-z)) / (z**2 - 1j * z))
   w = np.ones(n_points)
   w[1:-1:2] = 4.0
   w[2:-2:2] = 2.0

   integral = np.dot(w,integrand) * dy / 3.0

   return -exp(-rho * x_K) / np.pi * integral