from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.datasets import load_boston
# -*- coding: utf-8 -*-
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

# Only needed for data generation and graphical output
from pylab import ion, ioff, figure, draw, contourf, clf, show, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from random import normalvariate
import pandas as pd
import numpy as np
def getTargetData(fielname):
    file_object  = open(fielname);
    all_text_lines = file_object.readlines()
    all_text = []
    train_text = []
    train_classfi = []
    for line in all_text_lines:
        all_text.append(line.strip().split(","))
    train_classfi  = all_text[0]
    for i in range(len(all_text)):
        if(i!=0):
            train_text.append(all_text[i])
    train_text = np.array(train_text)
    train_classfi = np.array(train_classfi)
    for i in range(len(train_text)):
        for j in range(len(train_text[0])):
            train_text[i][j] = float(train_text[i][j])
    train_classfi_number = []
    for i in range(len(train_classfi)):
        features = train_text[i]
        if train_classfi[i]=="lumina" :
            klass = 0
            train_classfi_number.append(klass)
        elif train_classfi[i]=="ERBB2" :
            klass = 1
            train_classfi_number.append(klass)
        elif train_classfi[i]=="basal" :
            klass = 2
            train_classfi_number.append(klass)
        elif train_classfi[i]=="normal" :
            klass = 3
            train_classfi_number.append(klass)
        elif train_classfi[i]=="cell_lines" :
            klass = 4
            train_classfi_number.append(klass)
    train_feature_name = []
    for i in range(len(train_text)):
        train_feature_name.append(i)
    return train_text.transpose(),train_classfi_number,train_classfi,train_feature_name


def gettestbyindex(filename,index,number):
    ss = []
    count = 0
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData(filename)
    df = pd.DataFrame(train_text)
    temp=index
    for line in temp:
        count == 0
        count += 1
        ss.append(df[line[1]].values)
        if(count==number):
            break
    train_text = np.array(ss).transpose()
    return train_text

def RF(trainfilename,testfilename,number):
    train_text,train_classfi_number,train_classfi,train_feature_name = getTargetData(trainfilename)
    X = train_text
    Y = train_classfi_number
    names = train_feature_name
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    temp = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True) 
    ss = []
    count = 0
    df = pd.DataFrame(train_text)
    for line in temp:
        count == 0
        count += 1
        ss.append(df[line[1]].values)
        if(count==number):
            break
    train_text = np.array(ss).transpose()
    test_text = gettestbyindex(testfilename,temp,number)
    return train_text,test_text

