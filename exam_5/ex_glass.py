import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

file_object = open("glass.data")
try:
	all_lines = file_object.readlines()
	all_lines.pop(-1)
	print all_lines[0]
finally:
	file_object.close()

