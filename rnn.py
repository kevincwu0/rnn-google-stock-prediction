# Recurrent Neural Network
# Part 1 - Data preprocessing

# CS 229 Stanford 
# http://cs229.stanford.edu/proj2012/BernalFokPidaparthi-FinancialMarketTimeSeriesPredictionwithRecurrentNeural.pdf
# train LSTM on five years of Google 
# Supervised Deep Learning

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv("Google_Stock_Price_Train.csv")

# input/ouputs of recurrent neural (input != date, stock price)
# but stock price at time t for input, and stock price t+1 for the output
# create a set only with the "Open" Google stock price, extract that column
# two-dimensional numpy array
training_set = training_set.iloc[:,1:2].values