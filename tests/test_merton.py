from models.black_scholes import bs_call
from models.merton import MertonModel, merton_analytical

# With lam=0, Merton should reduce to Black-Scholes
bs = bs_call(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)

model = MertonModel(sigma=0.2, lam=0.0, mu_J=0.0, sigma_J=0.5)
merton = merton_analytical(model=model, S0=1.0, K=1.0, T=1.0, r=0.0)

print(f"Black Scholes: {bs}")
print(f"Merton: {merton}")
print(f"Difference: {abs(bs - merton)}")

#To run this you should use --> python -m tests.test_merton -- from the project root