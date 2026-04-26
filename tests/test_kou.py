from models.black_scholes import bs_call
from models.kou import KouModel, kou_analytical
from solvers.fd_solver import solve
import numpy as np

bs = bs_call(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)

kou_model = KouModel(sigma=0.2, lam=0.0, p=0.5, alpha1=3.0, alpha2=2.0)
kou_price = kou_analytical(
   model = kou_model,
   x_K = 0.0,
   tau = 1.0,
   r = 0.0
)

print(f"Black Scholes: {bs:.8f}")
print(f"Kou Analytical: {kou_price:.8f}")
print(f"Difference: {abs(bs - kou_price):.4f}")