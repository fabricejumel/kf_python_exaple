#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from kf_func import kf
from kf_func import ekf

endtime = 5.                        # target end time [sec]
delta = 0.1                       # sampling time [sec]
x_num = 1                           # num of state vector
z_num = 1                           # num of observation vector


# Setting Parameters
system_sds = np.array([[
    1.,
    ]]).T
measure_sds = np.array([[
    10.,
    ]]).T

F = np.array([
    0.,
    ]).reshape(x_num, x_num)
G = np.array([
    1.,
    ]).reshape(x_num, system_sds.size)
H = np.array([
    1.,
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

def f(x):
    return x + 3 * np.cos(x/10.)

def h(x):
    return x**3

x[0,0] = 10
for i in range(1, itr):
    x[i,0] = f(x[i-1,0]) + system_noise[i-1]
    z[i] = h(x[i])
    pass

# Initialize
xh[0] = 11.0
#xh[:,3] = np.pi/2.
P[0,0,0] = 1.0
# Start filtering
for i in range(1, itr):
    F[0,0] = - 3.0 * np.sin(xh[i-1,0]/10.0) / 10.0
    H[0,0] = 3.0 * xh[i-1,0] ** 2.0
    xp = f(xh[i-1])
    zp = h(xp)
    xm[i], xh[i], P[i], K[i] = ekf(F, G, H, P[i-1], Q, R, z[i], zp, xp)
    pass

def plot_results(start, end):
    s_idx = int(start/delta)
    e_idx = int(end/delta)
    t = np.arange(start, end, delta)

    subplt_num = 2
    subplt_loc = 0
    plt.figure(plot_results.id); plot_results.id += 1
    subplt_loc += 1
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(x[s_idx:e_idx,0,:],     "--",   label="Truth")
    plt.plot(xh[s_idx:e_idx,0,:],    "--",    label="Filtered")
    plt.legend()

    subplt_loc += 1
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(z[s_idx:e_idx,0,:],    "-",    label="Observed")
    plt.xlabel("Sample")
    plt.legend()

plot_results.id=0
plot_results(0., endtime)
plt.show()
