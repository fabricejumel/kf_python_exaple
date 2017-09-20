#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from kf_func import kf

endtime = 2.                        # target end time [sec]
delta = 0.001                       # sampling time [sec]
omega = 0.01                        # omega
x_num = 3                           # num of state vector
z_num = 2                           # num of observation vector


# Setting Parameters
system_sds = np.array([[
    0.,         # sigma
    0.,         # sigma
    1.,         # sigma
    ]]).T
measure_sds = np.array([[
    0.1,        # sigma
    0.1,        # sigma
    ]]).T

Phi = np.array([
    0., 0., 0.,
    0., 0., 0.,
    0., 0., 1,
    ]).reshape(x_num, x_num)
G = np.array([
    (delta**2)/2,
    delta,
    ]).reshape(x_num, system_sds.size)
H = np.array([
    1, 0,
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

for i in range(1, itr):
    x[i] = Phi @ x[i-1] + G @ system_noise[i-1]
    z[i] = H @ x[i] + measure_noise[i]
    pass

# Initialize
xh[0] = 0
P[0] = 0
# Start filtering
for i in range(1, itr):
    xm[i], xh[i], P[i], K[i] = kf(Phi, G, H, P[i-1], Q, R, z[i], xh[i-1])
    pass

def plot_results(id, start, end):
    subplt_num = 4
    subplt_loc = 0;
    s_idx = int(start/delta)
    e_idx = int(end/delta)
    t = np.arange(start, end, delta)

    plt.figure(id)
    subplt_loc += 1;
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(t, x[s_idx:e_idx,0,:].flatten(),    "--",   label="Truth")
    plt.plot(t, xm[s_idx:e_idx,0,:].flatten(),   "-.",   label="Predicted")
    plt.plot(t, xh[s_idx:e_idx,0,:].flatten(),   "-",    label="Filtered")
    plt.plot(t, z[s_idx:e_idx,0,:].flatten(),    ":",    label="Observed")
    plt.ylabel("Position [m]")
    plt.legend()

    subplt_loc += 1;
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(t, x[s_idx:e_idx,1,:].flatten(),    "--",   label="Truth")
    plt.plot(t, xm[s_idx:e_idx,1,:].flatten(),   "-.",   label="Predicted")
    plt.plot(t, xh[s_idx:e_idx,1,:].flatten(),   "-",    label="Filtered")
    plt.ylabel("Velocity [m/sec]")
    plt.legend()

    subplt_loc += 1;
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(t, K[s_idx:e_idx,0,0].flatten(), label="for position")
    plt.plot(t, K[s_idx:e_idx,1,0].flatten(), label="for velocity")
    plt.ylabel("K Gain")
    plt.legend()

    subplt_loc += 1;
    plt.subplot(subplt_num,1,subplt_loc)
    plt.plot(t, P[s_idx:e_idx,0,0].flatten(), label="for position")
    plt.plot(t, P[s_idx:e_idx,1,1].flatten(), label="for velocity")
    plt.xlabel("Time [sec]")
    plt.ylabel("Estimate Covariance")
    plt.legend()

plot_results(0, 0., endtime)
plot_results(1, 0., 0.2)
plot_results(2, 0.2, 0.4)
plt.show()
