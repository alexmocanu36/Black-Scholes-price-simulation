

import numpy as np
import matplotlib.pyplot as plt


def simulation_black_scholes_indep(n_assets, T, k, S0, mu, sigma):
    dt = T / k
    a = mu - 0.5 * sigma**2
    S = np.zeros((n_assets, k+1))
    S[:, 0] = S0
    
    for j in range(1, k+1):
        Z = np.random.normal(0, np.sqrt(dt), size=n_assets) 
        for i in range(n_assets):
            S[i, j] = S[i, j-1] * np.exp(a[i] * dt + sigma[i] * Z[i])
    
    return np.linspace(0, T, k+1), S

def simulation_black_scholes_dependent(n_assets, T, k, S0, mu, sigma_matrix):
    
    dt = T / k
    times = np.linspace(0, T, k+1)
    
   
    a = mu - 0.5 * np.sum(sigma_matrix**2, axis=1)
    
    S = np.zeros((n_assets, k+1))
    S[:, 0] = S0
    
    for j in range(1, k+1):
        Z = np.random.normal(0, np.sqrt(dt), size=n_assets)
        
        
        for i in range(n_assets):
            shock = np.dot(sigma_matrix[i], Z)
            S[i, j] = S[i, j-1] * np.exp(a[i] * dt + shock)
    
    return times, S

n_assets = 5
T = 1  # 1 year
k = 252   # trading days
S0 = np.array([100, 90, 120, 80, 70]) #initial price
mu = np.array([0.05, 0.04, 0.06, 0.03, 0.07])
sigma = np.array([0.2, 0.25, 0.15, 0.3, 0.22]) #we assume the stocks are independent (volatility matrix is diagonal)
sigma_matrix = np.array([
    [0.20, 0.05, 0.02, 0.01, 0.00],
    [0.04, 0.25, 0.03, 0.02, 0.01],
    [0.01, 0.02, 0.15, 0.05, 0.02],
    [0.00, 0.01, 0.04, 0.30, 0.03],
    [0.02, 0.00, 0.01, 0.03, 0.22]
]) #correlated stocks

time_grid, S_paths =simulation_black_scholes_indep(n_assets, T, k, S0, mu, sigma)


plt.figure(figsize=(10, 6))
for i in range(n_assets):
    plt.plot(time_grid, S_paths[i], label=f"Asset {i+1}")
plt.title("Simulation of 5 Independent Assets ")
plt.xlabel("Time (Years)")
plt.ylabel("Asset Price")
plt.legend()
plt.grid(True)
plt.show()


times, S_paths = simulation_black_scholes_dependent(n_assets, T, k, S0, mu, sigma_matrix)


plt.figure(figsize=(10, 6))
for i in range(n_assets):
    plt.plot(times, S_paths[i], label=f"Asset {i+1}")
plt.title("Simulation of 5 dependent assets")
plt.xlabel("Time (Years)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
