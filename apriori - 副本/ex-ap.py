import apriori as ap
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

df = pd.read_excel("Transactions.xls")
columns = df.columns
values = df.values
data = []
for i in range(len(values)):
	temp = []
	for j in range(len(values[0])):
		if values[i][j] == 1:
			temp.append(j)
	data.append(temp)
counts = []
for index in columns:
	line = df[index]
	count = 0
	for i in range(len(line)):
		if line[i]==1:
			count += 1
	counts.append((count))
plt.plot(counts)
plt.show()
counts.sort()
minSupport = counts[len(counts)*1/5]/10000
print minSupport
# df = pd.DataFrame(counts)
# df.to_excel("out.xls")


