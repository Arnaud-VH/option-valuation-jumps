from models.black_scholes import bs_call
from models.merton import merton_analytical

#Check that with lambda = 0, we recover the Black-Scholes
bs = bs_call(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)
merton = merton_analytical(S0=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2, lam=0.0, mu_J=0.0, sigma_J=0.5)

print(f"Black Scholes: {bs}")
print(f"Merton: {merton}")
print(f"Difference: {abs(bs - merton)}")
