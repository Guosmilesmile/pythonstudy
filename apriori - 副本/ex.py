import apriori as ap
import fpGrowth as fp
import pandas as pd
import numpy as np

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
	counts.append((float)(count)/10000)
counts.sort()
minSupport = counts[len(counts)*1/5]

#use apriori 
L,supportData = ap.apriori(data,minSupport)
rules = ap.generateRules(L,supportData,minConf=0.4)

#use fpGrowth

minSup = minSupport*10000
simpDat = data
initSet = fp.createInitSet(simpDat)
myFPtree, myHeaderTab = fp.createTree(initSet, minSup)
myFreqList = []
fp.mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
print myFreqList