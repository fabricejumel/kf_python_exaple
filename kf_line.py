#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

p_sd = 10
p_noise = np.random.normal(0, p_sd, size=10000) # 10 m/s^2
m_sd = 0.1
m_noise = np.random.normal(0, m_sd, size=10000) # 10 cm

dt = 0.001   # 1 msec
x = np.matrix([[0], [0]]) # state: [position, velocity]
F = np.matrix([[1, dt], [0, 1]])
G = np.matrix([[dt**2/2], [dt]])
H = np.matrix([[1, 0]])

P = np.matrix([[0, 0], [0, 0]])
Q = p_sd**2 * (G * G.T)
R = m_sd**2

plt.plot(samples)
plt.show()
