#!/usr/bin/env python
import subprocess
from time import sleep
import threading
import csv
from trackETFpar import thePar
import math
import pandas as pd
import datetime, dateutil.parser
import numpy as np
import pandas_datareader.data as web
import statsmodels.tsa.stattools as ts
from pandas_datareader import data, wb
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import style
import statsmodels.formula.api as lm
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm
from decimal import Decimal
from pymongo import MongoClient
from sklearn import linear_model, datasets
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn import linear_model, datasets
import imp
from sklearn.naive_bayes import GaussianNB
from pymongo import MongoClient
import collections, re
import nltk
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn import neighbors, datasets

thePath = 'C:/Users/xilin/gitHubCode/parallelAnalysis/'
start = datetime.datetime(2000,1,1)
end = datetime.datetime(2016,7,12)

mongo = MongoClient('127.0.0.1', 27017)
mongoDb = mongo['priceData']
mongoColl = mongoDb['crspData'] #price data
mongoInsert = mongoDb['dailyRun'] #write what assets to trade

theTickers = np.sort(np.array(mongoColl.distinct('ticker')))
theTickers = [s.strip('$') for s in theTickers]
numTickers = len(theTickers)

numThreads = 4
    
tickerSlice = numTickers/numThreads 
print(tickerSlice) 

theWindow = 10
testSize = .95
postThresh = .8
baseTicker = "^VIX"

#theTickers, start, end, postThresh, theWindow, testSize, thePath
def seekOne():
    thePar(theTickers[0:tickerSlice], baseTicker, start, end, postThresh, theWindow, testSize, tickerSlice, "one", thePath, mongoInsert)
one_thread = threading.Thread(target=seekOne)
one_thread.start()

def seekTwo():
    thePar(theTickers[tickerSlice:2*tickerSlice], baseTicker, start, end, postThresh, theWindow, testSize, tickerSlice, "two", thePath, mongoInsert)
two_thread = threading.Thread(target=seekTwo)
two_thread.start()

def seekThree():
    thePar(theTickers[2*tickerSlice:3*tickerSlice], baseTicker, start, end, postThresh, theWindow, testSize, tickerSlice, "three", thePath, mongoInsert)
three_thread = threading.Thread(target=seekThree)
three_thread.start()

def seekFour():
    thePar(theTickers[3*tickerSlice:len(theTickers)], baseTicker, start, end, postThresh, theWindow, testSize, tickerSlice, "four", thePath, mongoInsert)
three_thread = threading.Thread(target=seekFour)
three_thread.start()

