#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import kf_wiki_sample as wiki

itr = 50                         # iteration
x_num = 4                        # state vector
z_num = 1                        # observation vector

system_sds = np.array([[
    0.0,
    0.0,
    0.0,
    0.0,
    ]]).T
measure_sds = np.array([[
    0.2,
    ]]).T

Phi = np.eye(x_num)
G = np.eye(x_num)
H = np.array([
    0., 0., 0., 0.,
    ]).reshape(z_num, x_num)

x = np.zeros( (itr, x_num, 1) )     # true state
xm = np.zeros( x.shape )            # predected state in previous
xh = np.zeros( x.shape )            # predected state
z = np.zeros( (itr, z_num, 1) )     # observation array
zh = np.zeros( (itr, z_num, 1) )     # observation array
P = np.zeros( (itr, x_num, x_num) )
K = np.zeros( (itr, x_num, z_num) )

Q = np.diag(system_sds.flatten()**2)
R = np.diag(measure_sds.flatten()**2)

system_noise = np.random.randn(itr, system_sds.size, 1) * system_sds
measure_noise = np.random.randn(itr, measure_sds.size, 1) * measure_sds
u = np.sign( np.random.randn(itr,1, 1) )

x[:,0,0] = -1.5
x[:,1,0] = 0.7
x[:,2,0] = 1.
x[:,3,0] = 0.5
#x += system_noise
#xh[:,0,:], xh[:,1,:], xh[:,2,:], xh[:,3,:] = (-1.5, 0.7, 1., 0.5)

# Initialize
z[0] = H @ x[0] + measure_noise[0]
P[0][np.diag_indices(x_num)] = 0.5
for i in range(1, itr):
    # Update with previous value
    H[0,1] = H[0,0]
    H[0,3] = H[0,2]
    H[0,0] = -z[i-1]
    H[0,2] = u[i-1]
    z[i] = H @ x[i] + measure_noise[i]
    xm[i], xh[i], P[i], K[i] = wiki.kf(Phi, G, H, P[i-1], Q, R, z[i], xh[i-1])
    zh[i] = H @ xh[i]
    pass

subplt_num = 2
subplt_loc = 0

subplt_loc += 1
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(z[:,0,:].flatten(), "--", label="Observed")
plt.plot(zh[:,0,:].flatten(), label="Predicted")
plt.ylabel("Signal")
plt.legend()

subplt_loc += 1
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(x[:,0,:].flatten(), "--", label="a1 truth")
plt.plot(x[:,1,:].flatten(), "--", label="a2 truth")
plt.plot(x[:,2,:].flatten(), "--", label="b1 truth")
plt.plot(x[:,3,:].flatten(), "--", label="b2 truth")
plt.plot(xh[:,0,:].flatten(), label="a1 predicted")
plt.plot(xh[:,1,:].flatten(), label="a2 predicted")
plt.plot(xh[:,2,:].flatten(), label="b1 predicted")
plt.plot(xh[:,3,:].flatten(), label="b2 predicted")
plt.ylabel("Parameters")
plt.legend(loc=4)

plt.show()
