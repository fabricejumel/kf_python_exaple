#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def kf(Phi, G, H, P, Q, R, z, x, u=0):
    xp = Phi @ x + u;
    Pm = Phi @ P @ Phi.T + G @ Q @ G.T
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    xf = xp + K @ (z - H @ xp)
    Pp = (np.eye(x.size) - K @ H) @ Pm

    return xp, xf, Pp, K

itr = 100                          # iteration
x_num = 2                           # state vector
y_num = 1                           # observation vector

delta = 0.01

system_sds = np.array([[
    100,
    ]]).T
measure_sds = np.array([[
    1,
    ]]).T

Phi = np.array([
    1, delta,
    0, 1,
    ]).reshape(x_num, x_num)
G = np.array([
    (delta**2)/2,
    delta
    ]).reshape(x_num, system_sds.size)
H = np.array([
    1, 0,
    ]).reshape(y_num, x_num)

x = np.zeros( (itr, x_num, 1) )     # true state
xm = np.zeros( x.shape )            # predected state
xh = np.zeros( x.shape )            # predected state
y = np.zeros( (itr, y_num, 1) )     # observation array
P = np.zeros( (itr, x_num, x_num) )
K = np.zeros( (itr, x_num, y_num) )

Q = np.diag(system_sds.flatten()**2)
R = np.diag(measure_sds.flatten()**2)

system_noise = np.random.randn(itr, system_sds.size, 1) * system_sds
measure_noise = np.random.randn(itr, measure_sds.size, 1) * measure_sds

y[0] = H @ x[0] + measure_noise[0]
for i in range(1, itr):
    x[i] = Phi @ x[i-1] + G @ system_noise[i-1]
    y[i] = H @ x[i] + measure_noise[i]
    pass

xh[0] = 0
P[0] = 0
K[0] = 0
for i in range(1, itr):
    xm[i], xh[i], P[i], K[i] = kf(Phi, G, H, P[i-1], Q, R, y[i], xh[i-1])
    pass

subplt_num = 4
subplt_loc = 0;

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(y[:,0,:].flatten(), label="Observed")
plt.plot(x[:,0,:].flatten(), label="Truth")
plt.plot(xm[:,0,:].flatten(), label="Predicted")
plt.plot(xh[:,0,:].flatten(), label="Filtered")
plt.legend()

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(x[:,1,:].flatten(), label="Truth")
plt.plot(xm[:,1,:].flatten(), label="Predicted")
plt.plot(xh[:,1,:].flatten(), label="Filtered")
plt.legend()

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(K[:,0,0].flatten())

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(P[:,0,0].flatten())

plt.show()
