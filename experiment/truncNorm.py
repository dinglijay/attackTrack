import torch
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

import numpy as np  
  
x = np.linspace(-2, 5, 100)  
     
# Varying positional arguments  
a, b = -1, 2
y1 = truncnorm.pdf(x, a, b)  

a, b = -1, 1
y2 = truncnorm.pdf(x, a, b)
plt.plot(x, y1, "*", x, y2, "r--")  
plt.show()


# R = truncnorm .rvs(a, b, size = 10)  