from models.merton import merton_analytical
from merton_numerical_class import MertonNumericalClass
def computeError_X0(M=64,N=10,tao=0.2,maturity=1.0,sigma=0.2,lam=0.1,K=1.0,r=0.0,mu_J=0.0,sigma_J=0.5):
    analytical = merton_analytical(
            S0=1, 
            K=K, 
            T=tao, 
            r=r,
            sigma=sigma, 
            lam=lam, 
            mu_J=mu_J, 
            sigma_J=sigma_J
    )
    mnc = MertonNumericalClass(
        K=K, 
        T=maturity, 
        r=r, 
        sigma=sigma, 
        lam=lam, 
        sigma_J=sigma_J, 
        mu_J=mu_J,
        M=M, 
        N=N
    )
    if maturity==tao:
        numerical = mnc.getPriceMaturity()[mnc.M]
    else:
        numerical = mnc.getPriceTaoX0(tao)
    delta = analytical - numerical
    return delta
print(computeError_X0())
