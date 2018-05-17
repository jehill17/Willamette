# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:45:49 2018

@author: Joy Hill
"""

#Willamette historical generation for 2005

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import ExcelWriter
import numpy as np
import scipy as sp
from scipy.interpolate import interp2d
from sklearn import linear_model
from sklearn.metrics import r2_score
#from xml.dom import minidom
#import xml.etree.ElementTree as ET
#import untangle as unt
import xmltodict as xmld
#from collections import OrderedDict
#import dict_digger as dd
import datetime as dt
import Williamette_model as inner #reading in the inner function
import os


Willamette_gen_2005 = pd.read_excel('Data/Williamette_historical_hydropower_gen.xlsx',sheetname='2005',skiprows=4) #this is hourly
Willamette_gen_2005.columns = ['CGR','DET','DEX','FOS','GPR','HCR','LOP','BCL']

CGR_2005 = Willamette_gen_2005['CGR']
CGR_2005_hourly = np.mean(np.array(CGR_2005).reshape(-1,24),axis=1)