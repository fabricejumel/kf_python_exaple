import numpy as np
import matplotlib.pyplot as plt

def kf(F, G, H, P, Q, R, z, x, u=0):
    """ Kalman Filtering function
        x(k) = F@x(k-1) + u(k-1) + w(k-1)
        z(k) = R@x(k) + v(k)
        w(k) := N(0, Q)
        v(k) := N(0, R)
    Args:
        F: dynamics
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
    I = np.eye(x.size)
    xp = (I+F) @ x + u;
    Pm = (I+F) @ P @ (I+F).T + G @ Q @ G.T
    S = H @ Pm @ H.T + R
    K = Pm @ H.T @ np.linalg.inv(S)
    xf = xp + K @ (z - H @ xp)
    newP = (np.eye(x.size) - K @ H) @ Pm

    return xp, xf, newP, K
