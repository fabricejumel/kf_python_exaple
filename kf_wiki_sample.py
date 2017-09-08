#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

f_control_input = True

def kf(Phi, G, H, P, Q, R, z, x, u=0):
    """ Kalman Filtering function
        x(k) = Phi@x(k-1) + u(k-1) + w(k-1)
        z(k) = R@x(k) + v(k)
        w(k) := N(0, Q)
        v(k) := N(0, R)
    Args:
        Phi: dynamics
        G: System noise model
        H: Observation model
        P: Current error covariance matrix
        Q: System noise covariance matrix
        R: Observation noise covariance matrix
        z: Observation vector
        x: State vector
        u: Control input
    Returns: xp, xf, newP, K
        xp: Predicted state
        xf: Filtered state
        newP: Updated P
        K: Kalman gain
    """
    xp = Phi @ x + u;
    Pm = Phi @ P @ Phi.T + G @ Q @ G.T
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    xf = xp + K @ (z - H @ xp)
    newP = (np.eye(x.size) - K @ H) @ Pm

    return xp, xf, newP, K

itr = 300                         # iteration
x_num = 2                           # state vector
z_num = 1                           # observation vector

delta = 0.01

system_sds = np.array([[
    100,     # sigma = 10 [m/s^2]
    ]]).T
measure_sds = np.array([[
    1,      # sigma = 1 [m]
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
    ]).reshape(z_num, x_num)

x = np.zeros( (itr, x_num, 1) )     # true state
xm = np.zeros( x.shape )            # predected state in previous
xh = np.zeros( x.shape )            # predected state
z = np.zeros( (itr, z_num, 1) )     # observation array
P = np.zeros( (itr, x_num, x_num) )
K = np.zeros( (itr, x_num, z_num) )

Q = np.diag(system_sds.flatten()**2)
R = np.diag(measure_sds.flatten()**2)

system_noise = np.random.randn(itr, system_sds.size, 1) * system_sds
measure_noise = np.random.randn(itr, measure_sds.size, 1) * measure_sds
if (f_control_input):   # control input [m/s^2]
    u = 100 * np.cos( np.linspace( 0, itr*2*np.pi/100., itr ) ).reshape(itr, 1, 1)

z[0] = H @ x[0] + measure_noise[0]
for i in range(1, itr):
    x[i] = Phi @ x[i-1] + G @ system_noise[i]
    if (f_control_input):
        x[i] = x[i] + G @ u[i]
    z[i] = H @ x[i] + measure_noise[i]
    pass

# Initialize
xh[0] = 0
P[0] = 0
K[0] = 0
for i in range(1, itr):
    if (f_control_input):
        xm[i], xh[i], P[i], K[i] = kf(Phi, G, H, P[i-1], Q, R, z[i], xh[i-1], G@u[i])
    else:
        xm[i], xh[i], P[i], K[i] = kf(Phi, G, H, P[i-1], Q, R, z[i], xh[i-1])
    pass

subplt_num = 4
subplt_loc = 0;

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(z[:,0,:].flatten(), label="Observed")
plt.plot(xm[:,0,:].flatten(), label="Predicted")
plt.plot(xh[:,0,:].flatten(), label="Filtered")
plt.plot(x[:,0,:].flatten(), label="Truth")
plt.ylabel("Position [m]")
plt.legend()

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(xm[:,1,:].flatten(), label="Predicted")
plt.plot(xh[:,1,:].flatten(), label="Filtered")
plt.plot(x[:,1,:].flatten(), label="Truth")
plt.ylabel("Velocity [m]")
plt.legend()

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(K[:,0,0].flatten(), label="for position")
plt.plot(K[:,1,0].flatten(), label="for velocity")
plt.ylabel("K Gain")
plt.legend()

subplt_loc += 1;
plt.subplot(subplt_num,1,subplt_loc)
plt.plot(P[:,0,0].flatten(), label="for position")
plt.plot(P[:,1,1].flatten(), label="for velocity")
plt.ylabel("Covariance")
plt.legend()

plt.show()
