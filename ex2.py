import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

raw_data = open("iris.data.txt", "r")

lines = raw_data.readlines()
lines.pop()
for i in range(0, len(lines)):
	lines[i] = lines[i].strip().split(",")

for line in lines:
	for i in range(0, len(line)-1):
		line[i] = float(line[i])

df = pd.DataFrame(lines)

data = lines

cols = [[],[],[]]
for d in data:
	if d[4] == 'Iris-setosa':
		cols[0].append(d[0])
	elif d[4] == "Iris-versicolor":
		cols[1].append(d[0])
	elif d[4] == "Iris-virginica":
		cols[2].append(d[0])
arr_cols = np.array(cols)
arr_cols = arr_cols.transpose()
plt.boxplot(arr_cols)
plt.show()
