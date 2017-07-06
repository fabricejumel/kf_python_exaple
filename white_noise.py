#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

mean = 0
std = 1
num_samples = 1000
samples = np.random.normal(mean, std, size=num_samples)

plt.plot(samples)
plt.show()
