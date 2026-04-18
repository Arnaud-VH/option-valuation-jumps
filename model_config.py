"""
Config file for numerical experiments.
Parameters chosen in accordance with Almendral & Oosterlee (2005) paper. 
"""
from models.merton import MertonModel
from models.kou import KouModel

MERTON_PARAMS = dict(
   K = 1.0, 
   T = 1.0, 
   r = 0.0, 
   sigma = 0.2, 
   lam = 0.1, 
   mu_J = 0.1,
   sigma_J = 0.5,
)

MERTON_MODEL = MertonModel(
   sigma = MERTON_PARAMS['sigma'],
   lam = MERTON_PARAMS['lam'], 
   mu_J = MERTON_PARAMS['mu_J'], 
   sigma_J = MERTON_PARAMS['sigma_J'],
)

X_STAR = 6.0 #Truncation point
N_TERMS = 10 #Terms in Merton analytical series

CONVERGENCE_CONFIGS = [
   (32, 5), 
   (64, 10), 
   (128, 20), 
   (256, 40), 
   (512, 80),
]

KOU_PARAMS = dict(
   K = 1.0,
   T = 0.2,
   r = 0.0,
   sigma = 0.2,
   lam = 0.2,
   p = 0.5,
   alpha1 = 3.0,
   alpha2 = 2.0,
)

KOU_MODEL = KouModel(
   sigma = KOU_PARAMS['sigma'],
   lam = KOU_PARAMS['lam'],
   p = KOU_PARAMS['p'],
   alpha1 = KOU_PARAMS['alpha1'],
   alpha2 = KOU_PARAMS['alpha2']
)

KOU_CONVERGENCE_CONFIGS = [
   (32,   10),
   (64,   20),
   (128,  40),
   (256,  80),
]