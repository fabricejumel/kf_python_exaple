#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from kf_func import kf

endtime = 2.                        # target end time [sec]
delta = 0.001                       # sampling time [sec]
x_num = 4                           # num of state vector
z_num = 2                           # num of observation vector


# Setting Parameters
system_sds = np.array([[
    0.0,
    ]]).T
measure_sds = np.array([[
    0.01,
    0.01,
    ]]).T

F = np.array([
    0., 0., 0., 0.,
    0., 0., 0., 0.,
    0., 0., 0., 0.,
    0., 0., 0., 0.,
    ]).reshape(x_num, x_num)
G = np.array([
    0.,
    0.,
    1.,
    0.,
    ]).reshape(x_num, system_sds.size)
H = np.array([
    1., 0., 0., 0.,
    0., 1., 0., 0.,
    ]).reshape(z_num, x_num)

itr = (int)(endtime / delta)
x = np.zeros( (itr, x_num, 1) )     # true state
xm = np.zeros( x.shape )            # predected state in previous
xh = np.zeros( x.shape )            # predected state
z = np.zeros( (itr, z_num, 1) )     # observation array
P = np.zeros( (itr, x_num, x_num) )
K = np.zeros( (itr, x_num, z_num) )

Q = np.diag(system_sds.flatten()**2)
R = np.diag(measure_sds.flatten()**2)

# Create simulation data
system_noise = np.random.randn(itr, system_sds.size, 1) * system_sds
measure_noise = np.random.randn(itr, measure_sds.size, 1) * measure_sds

x[:,3] = np.pi/2.
for i in range(1, itr):
    x[i,2] = x[i-1,2] + delta * x[i-1,3] + system_noise[i-1]
    x[i,0] = np.cos(x[i,2])
    x[i,1] = np.sin(x[i,2])
    z[i] = H @ x[i] + measure_noise[i]
    pass

# Initialize
xh[0] = 0
P[0] = 10
# Start filtering
for i in range(1, itr):
    F[0,2] = -np.sin(x[i-1,2])
    F[0,3] = -delta * np.sin(x[i-1,2])
    F[1,2] = np.cos(x[i-1,2])
    F[1,3] = delta * np.cos(x[i-1,2])
    F[2,3] = delta
    xm[i], xh[i], P[i], K[i] = kf(F, G, H, P[i-1], Q, R, z[i], xh[i-1])
    pass

def plot_results(id, start, end):
    subplt_num = 1
    subplt_loc = 0;
    s_idx = int(start/delta)
    e_idx = int(end/delta)
    t = np.arange(start, end, delta)

    plt.figure(id)
    subplt_loc += 1;
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(x[s_idx:e_idx,0,:], x[s_idx:e_idx,1,:].flatten(),    "--",   label="Truth")
    plt.plot(xm[s_idx:e_idx,0,:], xm[s_idx:e_idx,1,:].flatten(),   "-.",   label="Predicted")
    plt.plot(xh[s_idx:e_idx,0,:], xh[s_idx:e_idx,1,:].flatten(),   "-",    label="Filtered")
    plt.plot(z[s_idx:e_idx,0,:], z[s_idx:e_idx,1,:].flatten(),    ":",    label="Observed")
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.legend()

plot_results(0, 0., endtime)
plt.show()
