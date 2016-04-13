import fpGrowth as fp
import pandas as pd
import numpy as np

# simpDat = fp.loadSimpDat()
# initset = fp.createInitSet(simpDat)
# myfptree,myheaderTab = fp.createTree(initset,3)
# freqItems = []
# fp.mineTree(myfptree,myheaderTab,3,set[()],freqItems)
# print freqItems
df = pd.read_excel("Transactions.xls")
values = df.values
data = []
for i in range(len(values)):
	temp = []
	for j in range(len(values[0])):
		if values[i][j] == 1:
			temp.append(j)
	data.append(temp)

minSup = 313.0
#simpDat = fp.loadSimpDat()
simpDat = data
initSet = fp.createInitSet(simpDat)
myFPtree, myHeaderTab = fp.createTree(initSet, minSup)
myFreqList = []
fp.mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
print myFreqList