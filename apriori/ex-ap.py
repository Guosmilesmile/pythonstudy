import apriori as ap
import pandas as pd
import numpy as np
# IS_DEBUG = 1;
# file_object = open("kosarak.dat")
# lines = file_object.readlines();
# alldata = []
# for line in lines:
# 	alldata.append(line.strip().split(" "))
# if IS_DEBUG == 1:
# 	alldata = alldata[0:50000]
# #c1 = ap.createC1(alldata)
# #d1 = map(set,alldata)
# #L1,supportData0 = ap.scanD(d1,c1,0.4)
# #print L1
# L,supportData = ap.apriori(alldata,0.2)
# rules = ap.generateRules(L,supportData,minConf=0.2)
# print L
df = pd.read_excel("Transactions.xls")
values = df.values
data = []
for i in range(len(values)):
	temp = []
	for j in range(len(values[0])):
		if values[i][j] == 1:
			temp.append(j)
	data.append(temp)
print data[1]
L,supportData = ap.apriori(data,0.05)
rules = ap.generateRules(L,supportData,minConf=0.3)
