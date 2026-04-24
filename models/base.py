"""
Abstract base class for jump-diffusion models. 
All models must implement these methods to work with the FD solver. 
"""

from abc import ABC, abstractmethod
from functools import cached_property

class JumpDiffusionModel(ABC):
   """
   Abstract base class that defines the interface that any jump diffusion model should implement
   to be compatible with the FD solver. 

   Subclasses must implement the following functions:
      - _compute_compensator()
      - jump_density(y)
      - tail_correction(tau, xi, x_star, K, r)
      - right_boundary(x_star, K, r, tau) 
   """

   @property
   @abstractmethod
   def sigma(self):
      """
      Diffusion volatility
      """
      pass

   @property
   @abstractmethod
   def lam(self):
      """
      Jump intensity
      """
      pass
   
   @cached_property
   def zeta(self):
      """Compensator E[e^Y - 1]. Computed once from _compute_compensator()."""
      return self._compute_compensator()

   @abstractmethod
   def _compute_compensator(self):
      """Subclasses implement the actual compensator formula here."""
      pass

   @abstractmethod
   def jump_density(self, y):
      """
      Jump size density f(y) evaluated at y. 
      """
      pass

   @abstractmethod
   def tail_correction(self, tau, xi, x_star, K, r):
      """
      Analytical correction for the integral over R\\Omega*. Represents the contribution from jumps outside domain. 
      
      Parameters:
      tau - current forward time
      xi - log-price at grid node i
      x_star - truncation point
      K - strike price
      r - risk-free rate
      """
      pass

   @abstractmethod
   def right_boundary(self, x_star, K, r, tau):
      """
      Dirichlet boundary value at x = x_star
      Parameters:
      x_star - truncation point
      K - strike price
      r - risk-free rate
      tau - current forward time
      """
      pass

