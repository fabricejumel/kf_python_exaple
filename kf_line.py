#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

itr = 1000
p_sd = 10 # 10[m/s^2]
p_noise = np.random.normal(0, p_sd, size=itr)
p_noise[0] = 0.
m_sd = 0.01 # 10[cm]
m_noise = np.random.normal(0, m_sd, size=itr)

dt = 0.01   # 1 msec
x = np.zeros( (2,itr) ) # true: position, velocity
y = np.zeros( (1,itr) ) # true: position, velocity
x_ = np.zeros( (2,itr) ) # predected: position, velocity
F = np.array([
    [1, dt],
    [0, 1],
    ])
G = np.array([
    [dt**2/2],
    [dt],
    ])
H = np.array([
    [1, 0],
    ])

P = np.array( [
    [0, 0],
    [0, 0],
    ])
Q = np.diag([p_sd**2])
R = np.diag([m_sd**2])
I = np.eye(2,2)

for i in range(1, itr):
    # preparation
    x[:,i] = (F @ x[:,i-1].reshape(2,1) + G * p_noise[i]).flatten()
    xv = x_[:,i-1].reshape(2,1)
    # predict
    xp = F @ xv
    P = F @ P @ F.T + G @ Q @ G.T
    # observe and update
    y[:,i] = H @ x[:,i].reshape(2,1) + m_noise[i]
    e = y[:,i] - H @ xp
    S = R + H @ P @ H.T
    K = P @ H.T @ np.linalg.inv(S)
    x_[:,i] = (xv + K @ e).flatten()
    P = (I - K @ H) @ P

#plt.plot(y[0])
plt.plot(x[1])
plt.plot(x_[1])
plt.show()
