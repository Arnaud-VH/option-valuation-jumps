"""
Config file for numerical experiments.
Parameters chosen in accordance with Almendral & Oosterlee (2005) paper. 
"""

MERTON_PARAMS = dict(
   K = 1.0, 
   T = 1.0, 
   r = 0.0, 
   sigma = 0.2, 
   lam = 0.1, 
   mu_J = 0.1,
   sigma_J = 0.5,
)

X_STAR = 4.0 #Truncation point
N_TERMS = 50 #Terms in Merton analytical series

CONVERGENCE_CONFIGS = [
   (32, 5), 
   (64, 10), 
   (128, 20), 
   (256, 40), 
   (512, 80),
]