#coding=utf-8
import urllib as ub
import re

BookIndex = 1
BookChannel = 21
BookUrl = "http://all.qidian.com/Book/BookStore.aspx?ChannelId="+bytes(BookChannel)+"&SubCategoryId=-1&Tag=all&Size=-1&Action=-1&OrderId=6&P=all&PageIndex="+bytes(BookIndex)+"&update=-1&Vip=-1&Boutique=-1&SignStatus=-1"

def getHtml(url):
    page = ub.urlopen(url)
    html = page.read()
    return html
def getImg(html):
    reg = r'src="(.+?\.png)" '
    imgre = re.compile(reg)
    imglist = re.findall(imgre,html)
    return imglist     
def gethtmlBookName(html):
	#reg = r'\<span\sclass="swbt"\>.*?\<a.*?\<\/span\>';
	#reg = r'<span\sclass="swbt">.*<a.*?>(.*?)<\/a><\/span>';
	reg = reg = r'<span\sclass="swbt"\><a.*?<\/span>';
	namere = re.compile(reg)
	namelist = re.findall(namere,html)
	return namelist 
def getBookList(htmllist):
	booklist = []
	for item in htmllist:
		index_one = item.find('target="_blank">')
		index_two = item.find('</a>')
		bookname = item[index_one+16:index_two]
		booklist.append(bookname)
	return booklist
html = getHtml(BookUrl)
htmlbooklist = gethtmlBookName(html)
booklist = getBookList(htmlbooklist)
print booklist
