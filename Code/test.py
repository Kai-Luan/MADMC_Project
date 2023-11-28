import numpy as np
import matplotlib.pyplot as plt
from utils import *

n= 100
data = np.random.randint(100, size=(n,2))

plt.scatter(*data.T)
plt.show()




