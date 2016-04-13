#-*- coding=utf-8 -*-
import jieba
import sys
import numpy as np
import operator
reload(sys)
sys.setdefaultencoding('utf-8')

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  
    #print sortedDistIndicies
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print sortedClassCount
    return sortedClassCount[0][0]

def getfilecontent(filename):
	filesource = open(filename,"r")
	file_lines = filesource.readlines()
	file_list = [] 
	for file_line in file_lines:
		file_list.append(file_line.strip())
	return file_list


u_stop_words = getfilecontent("stop_source.txt")
u_source_words = getfilecontent("source.txt")
source_classfication = getfilecontent("classfication.txt")
source_trainning = getfilecontent("testtraining.txt")

#dict of the keyword of source
source_dict = []
af_remove_stop = []
#temp_str = ''
for line_text in u_source_words:
	temp_list = []
	segs = jieba.cut(line_text,cut_all=False)
	for seg in segs:
		if seg not in u_stop_words:
			#temp_str += seg.encode("utf-8")
			temp_list.append(seg.encode("utf-8"))
			source_dict.append(seg.encode("utf-8"))
	if len(temp_list):
		if temp_list not in af_remove_stop:
			af_remove_stop.append(temp_list)
	#temp_str = ''
#print af_remove_stop[0][0]

af_remove_stop_training = []
for line in source_trainning:
	temp_list = []
	segs = jieba.cut(line,cut_all=False)
	for seg in segs:
		if seg not in u_stop_words:
			temp_list.append(seg.encode("utf-8"))
	if len(temp_list):
		if temp_list not in af_remove_stop_training:
			af_remove_stop_training.append(temp_list)

matri = []
for i in range(0,len(af_remove_stop)):
	temp_list = []
	for count in range(0,len(source_dict)):
		temp_list.append(0)
	for j in range(0,len(af_remove_stop[i])):
		for k in range(0,len(source_dict)):
			temp_word = source_dict[k]
			if temp_word  == af_remove_stop[i][j]:
				#print temp_word,af_remove_stop[i][j],k
				temp_list[k] += 1
				break
	matri.append(temp_list)

#print matri	
#print len(matri[0])

inX = []
result_classfication = []
for count in range(0,len(source_dict)):
		inX.append(0)

output = open('result.txt', 'w')
#print af_remove_stop_training[1]
for i in range(len(af_remove_stop_training)):
	for j in range(len(af_remove_stop_training[i])):
		if af_remove_stop_training[i][j] in source_dict:
			#print source_dict.index(af_remove_stop_training[i][j])
			inX[source_dict.index(af_remove_stop_training[i][j])] += 1
	#print inX
	result_classfication.append(classify(np.array(inX),np.array(matri),source_classfication,3)) 
	for count in range(0,len(source_dict)):
		inX[count] = 0
print result_classfication
output.writelines(result_classfication)

#print inX
