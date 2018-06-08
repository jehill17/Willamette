# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 16:44:20 2018

@author: sdenaro
"""
# this file runs the Willamette operational model for years 1989-2007 
# and plots/validates the results

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas import ExcelWriter
import numpy as np
import scipy as sp
from scipy import stats
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
import sys

#%%

runfile("Willamette_outer.py", "Flow_1989.xml")

#%%    
#OUTFLOW VALIDATION


#CGR outflows
x = (CGR5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,7]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
CGR.R2 = r_val**2
print(CGR.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Cougar Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(120,40,'$R^2$={}'.format(round(CGR.R2,4)),fontsize=15)




  
#DET outflows
x = (DET5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,11]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DET.R2 = r_val**2
print(DET.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Detroit Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,150,'$R^2$={}'.format(round(DET.R2,4)),fontsize=15)




  
#FOS outflows
x = (FOS5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,10]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FOS.R2 = r_val**2
print(FOS.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Foster Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(175,150,'$R^2$={}'.format(round(FOS.R2,4)),fontsize=15)










#GPR outflows
x = (GPR5H[0:6938,1].astype('float'))
y = outflows_all[0:6938,9]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
GPR.R2 = r_val**2
print(GPR.R2)


plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Green Peter Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(GPR.R2,4)),fontsize=15)




#HCR OUTFLOWS
x = (HCR5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.R2 = r_val**2
print(HCR.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Hills Creek Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,60,'$R^2$={}'.format(round(HCR.R2,4)),fontsize=15)





#LOP outflows validation
x = (LOP5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.R2 = r_val**2
print(LOP.R2)
plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Lookout Point Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(LOP.R2,4)),fontsize=15)



#BCL outflows valid
x = (BCL5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,12]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BCL.R2 = r_val**2
print(BCL.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Big Cliff Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(BCL.R2,4)),fontsize=15)


#nonhydro outflows

#COT
x = (COT5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
COT.R2 = r_val**2
print(COT.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Cottage Grove Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,30,'$R^2$={}'.format(round(COT.R2,4)),fontsize=15)


#DOR
x = (DOR5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,4]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DOR.R2 = r_val**2
print(DOR.R2)
plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Dorena Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,80,'$R^2$={}'.format(round(DOR.R2,4)),fontsize=15)



#FRN
x = (FRN5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,6]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FRN.R2 = r_val**2
print(FRN.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Fern Ridge Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')


#FAL
x = (FAL5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,3]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FAL.R2 = r_val**2
print(FAL.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Fall Creek Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(160,40,'$R^2$={}'.format(round(FAL.R2,4)),fontsize=15)

#BLU
x = (BLU5H[0:6939,1].astype('float'))
y = outflows_all[0:6939,8]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BLU.R2 = r_val**2
print(BLU.R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Blue River Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(160,60,'$R^2$={}'.format(round(BLU.R2,4)),fontsize=15)




#%%
#aggregate outflows and hydropower validation
#outflows_minus_FRN = np.delete(outflows_all[0:364],6,1)
outflows_aggr = np.sum(outflows_all[0:6939],axis=1)

outflows_aggr_1989 = np.sum(outflows_all_hist[1:6940],axis=1).astype(float)

x = outflows_aggr_1989
y = outflows_aggr
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
outflows_R2 = r_val**2
print(outflows_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Aggregate Outflows 89-07')
plt.legend(['Historical aggregate outflows','Simulated aggregate outflows'])
plt.xlabel('doy')
plt.ylabel('Outflows ($m^3$/s)')
plt.text(150,1200,'$R^2$={}'.format(round(outflows_R2,4)),fontsize=12)


#%%