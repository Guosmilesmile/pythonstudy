import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
x1 = [1, 2, 3, 4, 5,5]# Make x, y arrays for each graph
y1 = [1, 4, 9, 16, 25]
x2 = [1, 2, 4, 6, 8]
y2 = [2, 4, 8, 12, 16]
data = np.array(x1)
print(data)
plt.hist(data)
plt.show()