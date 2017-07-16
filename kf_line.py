#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def kf(Phi, G, H, P, Q, R, x, z):
    xm = Phi @ x
    Pm = Phi @ P @ Phi.T + G @ Q @ G.T
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    xp = xm + K @ (z - xm)
    Pp = (np.eye(x.size) - K @ H) @ Pm

    return xp, Pp

itr = 300
system_sd = 10
system_vars = np.array([system_sd])**2
system_noise = np.random.normal(0, system_sd, size=itr)
measure_sd = 100
measure_vars = np.array([measure_sd])**2
measure_noise = np.random.normal(0, measure_sd, size=itr)

x = np.zeros( (itr, 1, 1) ) # true: position, velocity
y = np.zeros( (itr, 1, 1) ) # true: position, velocity
xh = np.zeros( (itr, 1, 1) ) # predected: position, velocity

Phi = np.array([
    1
    ]).reshape(1,1)
G = np.array([
    1
    ]).reshape(1,1)
H = np.array([
    1
    ]).reshape(1,1)

Q = np.diag(system_vars)
R = np.diag(measure_vars)

y[0] = H @ x[0] + measure_noise[0]
for i in range(1, itr):
    x[i] = Phi @ x[i-1] + system_noise[i-1]
    y[i] = H @ x[i] + measure_noise[i]
    pass

xh[0] = 0
P = np.array( [
    0
    ]).reshape(1,1)

for i in range(1, itr):
    xh[i], P = kf(Phi, G, H, P, Q, R, x[i], y[i])
    pass

plt.figure()
plt.plot(y.reshape(y.size))
plt.plot(x.reshape(x.size))
plt.plot(xh.reshape(xh.size))

plt.show()
