import numpy as np
from scipy.stats import norm
class MertonNumericalClass:
    def __init__(self, K, T, r, sigma, lam, sigma_J, mu_J=0.0, M=20, N=20, x_star=4):
        self.K=K
        self.T=T
        self.r=r
        self.sigma=sigma
        self.lam=lam
        self.sigma_J=sigma_J
        self.mu_J=mu_J
        self.M=M
        self.N=N
        self.x_star=x_star
        self.doOtherInit()
        pass

    def doOtherInit(self):
        self.zeta = np.exp(self.mu_J + 0.5 * self.sigma_J**2) - 1.0
        # 2. Space-Time Grid Setup
        self.n = 2 * self.M + 1             # Total number of spatial points
        self.q = self.N + 1                 # Total number of time steps
        
        # Grid step sizes
        self.h = (2.0 * self.x_star) / (self.n - 1)  
        self.k = self.T / self.N
        
        # Spatial grid (X[0] is -x* and X[n-1] is x*)
        self.X = np.linspace(-self.x_star, self.x_star, self.n)
        self.S = np.exp(self.X)
        def jump_density(y):
            """Gaussian jump size density f_m(y)."""
            variance = self.sigma_J**2
            normalization = 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma_J)
            return normalization * np.exp(-((y - self.mu_J)**2) / (2.0 * variance))

        def tail_correction(tau, xi):
            """Analytical tail integral epsilon(tau, xi, x_star)."""
            d1 = (xi - self.x_star + self.sigma_J**2) / self.sigma_J
            d2 = (xi - self.x_star) / self.sigma_J
            return np.exp(xi + 0.5 * self.sigma_J**2) * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        C = np.zeros((self.n, self.n))
        alpha = (self.k * self.sigma**2) / (2.0 * self.h**2)
        beta = self.k * (self.r - 0.5 * self.sigma**2 - self.lam * self.zeta) / (2.0 * self.h)
        # Loop through strictly interior nodes
        for i in range(1, self.n - 1):
            C[i, i-1] = -alpha - beta                # Lower diagonal
            C[i, i]   = 2.0 * alpha + (self.r + self.lam) * self.k  # Main diagonal
            C[i, i+1] = -alpha + beta                # Upper diagonal

        # 5. Matrix D (Jump Integral Operator)
        D = np.zeros((self.n, self.n))
        for i in range(1, self.n - 1):
            for j in range(self.n):
                dy = self.X[j] - self.X[i]
                trapezoid = -self.k * self.h * self.lam * jump_density(dy)
                
                # Composite trapezoidal rule weights (halved at boundaries)
                if j == 0 or j == self.n - 1:
                    D[i, j] = trapezoid / 2.0
                else:
                    D[i, j] = trapezoid

        # 6. Time Stepping (BDF2)
        self.u = np.zeros((self.q, self.n))
        self.u[0, :] = np.maximum(self.S - self.K, 0.0)  # Payoff at maturity (tau=0)
        
        b = np.zeros(self.n)
        I = np.eye(self.n)

        # Step through time (tau_1 to tau_q-1)
        for m in range(1, self.q):
            tau_m = m * self.k
            
            # BDF2 weights (Implicit Euler for m=1, BDF2 for m>=2)
            w0 = 1.0 if m == 1 else 1.5
            w1 = 1.0 if m == 1 else 2.0
            w2 = 0.0 if m == 1 else -0.5
            
            # Matrix A (used to solve for u^m)
            A_m = (w0 * I) + C + D
            
            # Vector b (interior)
            for i in range(1, self.n - 1):
                jump_tail = self.k * self.lam * tail_correction(tau_m, self.X[i])
                history = w1 * self.u[m-1, i] + w2 * self.u[m-2, i]
                b[i] = jump_tail + history
                
            # Dirichlet Boundary conditions
            b[0] = 0.0
            b[-1] = w0 * (np.exp(self.x_star) - self.K * np.exp(-self.r * tau_m))
            
            # Solve the linear system A * u^m = b^m
            self.u[m, :] = np.linalg.solve(A_m, b)
    
    def getGrid_S(self):
        return self.S
    
    def getGrid_X(self):
        return self.X
    
    def getPrice0(self):
        # Array of option prices at t=0
        return self.u[-1, :]
    
    def getPriceTaoX0(self,tao):
        # Array of option prices at time to maturity tao=T-t
        if(tao==0): # tao=0,  t=T-tao=T
            return self.getPriceMaturity()[self.M]
        else:
            return self.u[int(round(tao/self.k)), :][self.M]
        
    def getPriceTao(self, tao):
        # Array of option prices at time to maturity tao=T-t
        if(tao==0): # tao=0,  t=T-tao=T
            return self.getPriceMaturity()
        else:
            return self.u[int(round(tao/self.k)), :]
    
    def getPriceMaturity(self):
        # payoff at time t=T
        return np.maximum(self.S - self.K, 0.0)
        # OR self.u[0]
    
    def getNumericalSolutionMatrix(self):
        return self.u