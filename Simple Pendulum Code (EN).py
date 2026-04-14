import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sc
from scipy.special import ellipj

# Parameters
L = 0.01 # m
g = 9.81 # m/s^2
theta0 = np.pi/3 # rad
k = np.sin(theta0/2)

# Jacobi
t_max = 3
N = 1000
t = np.linspace(0,t_max,N)
u = np.sqrt(g/L) * t
sn, cn, dn, ph = ellipj(u,k**2)
theta_exact = 2 * np.arcsin(k * sn)

# Odeint
def deriv(y,t):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g/L) * np.sin(theta)
    return [dtheta_dt,domega_dt]
y0 = [0,2*k*np.sqrt(g/L)]
sol = sc.odeint(deriv,y0,t)
theta_num = sol[:,0]

# Plot
plt.figure(figsize=(10,6))
plt.plot(t,theta_num,label="odeint",linewidth=3)
plt.plot(t,theta_exact,'--',label="Exact solution (Jacobi)",linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Theta angle (rad)")
plt.title("Simple pendulum : Exact solution vs odeint")
plt.legend()
plt.grid(True)
plt.show()

# Fonction angle exact
def sol(time):
    sn1, cn1, dn1, ph1 = ellipj(np.sqrt(g/L) * time,k**2)
    return 2 * np.arcsin(k * sn1)
