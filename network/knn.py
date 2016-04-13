#-*- coding=utf-8 -*-
import jieba
import jieba.analyse
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
fp = open("chinese_stopword.txt","r")
lines = fp.readlines()

for line in lines:
	line = line.encode("utf-8")


lines.pop()

for line in lines:
	print line.decode("utf-8"),
# seg_list = jieba.cut("我来到北京清华大学",cut_all=True)
# print "Full Mode:", "/ ".join(seg_list) #全模式

# seg_list = jieba.cut("我来到北京清华大学",cut_all=False)
# print "Default Mode:", "/ ".join(seg_list) #精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦".decode("utf-8")) #默认是精确模式
# print ", ".join(seg_list)


