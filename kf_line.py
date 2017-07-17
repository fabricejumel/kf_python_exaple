#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def kf(Phi, G, H, P, Q, R, x, z):
    xm = Phi @ x
    Pm = Phi @ P @ Phi.T + G @ Q @ G.T
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    xp = xm + K @ (z - H @ xm)
    Pp = (np.eye(x.size) - K @ H) @ Pm

    return xp, Pp, K

itr = 500                           # iteration
x_num = 2                           # state vector
y_num = 1                           # observation vector

system_sds = np.array([[
    10,
    ]]).T
measure_sds = np.array([[
    100,
    ]]).T

Phi = np.array([
    1, 0,
    0, 1,
    ]).reshape(x_num, x_num)
G = np.array([
    1, 0,
    ]).reshape(x_num, system_sds.size)
H = np.array([
    1, 1,
    ]).reshape(y_num, x_num)

x = np.zeros( (itr, x_num, 1) )     # true state
xh = np.zeros( x.shape )            # predected state
y = np.zeros( (itr, y_num, 1) )     # observation array
P = np.zeros( (itr, x_num, x_num) )
K = np.zeros( (itr, x_num, y_num) )

Q = np.diag(system_sds.flatten()**2)
R = np.diag(measure_sds.flatten()**2)

system_noise = np.random.randn(itr, system_sds.size, 1) * system_sds
measure_noise = np.random.randn(itr, measure_sds.size, 1) * measure_sds

x[0,1] = 500.   # Bias
y[0] = H @ x[0] + measure_noise[0]
for i in range(1, itr):
    x[i] = Phi @ x[i-1] + G @ system_noise[i-1]
    y[i] = H @ x[i] + measure_noise[i]
    pass

xh[0] = 0
P[0] = 0
K[0] = 0
for i in range(1, itr):
    xh[i], P[i], K[i] = kf(Phi, G, H, P[i-1], Q, R, xh[i-1], y[i])
    pass

plt.subplot(3,1,1)
plt.plot(y[:,0,:].flatten())
plt.plot(x[:,0,:].flatten())
plt.plot(xh[:,0,:].flatten())

plt.subplot(3,1,2)
plt.plot(K[:,0,0].flatten())

plt.subplot(3,1,3)
plt.plot(P[:,0,0].flatten())

plt.show()
