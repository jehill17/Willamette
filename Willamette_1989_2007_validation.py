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
import Willamette_model as inner #reading in the inner function
import os
import pylab as py

#%%

runfile("Willamette_outer.py", "Flow_1989.xml")

#%%    
#OUTFLOW VALIDATION

##PLOT
for i in range (0, len(res_list)):
    x = (RES[i].dataOUT[0:T-1,1].astype('float'))
    y = outflows_all[0:T-1,i]
    slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
    RES[i].R2 = r_val**2
    print(RES[i].R2)
#    plt.figure()
#    plt.plot(x)
#    plt.plot(y)
#    plt.title(RES[i].name,fontsize=20)
#    plt.legend(['historical outflows','simulated outflows'],fontsize=12, loc=1)
#    plt.xlabel('doy')
#    plt.ylabel('Outflows 1989-2007 (cubic meters/second)')
#    plt.text(150,max(outflows_all[:,i])-60,'$R^2$={}'.format(round(RES[i].R2,4)),fontsize=15)
#    #figure_name=os.path.join('Figures/',(RES[i].name + '.png'))
#    #py.savefig(figure_name, bbox_inches='tight')
#
##%%
##aggregate outflows and hydropower validation
##outflows_minus_FRN = np.delete(outflows_all[0:364],6,1)
outflows_aggr = np.sum(outflows_all[0:T-1],axis=1)
#
outflows_all_hist=np.zeros((6940,n_res))
for i in range(0,n_res):
    outflows_all_hist[:,i] = RES[i].dataOUT[:,1]
#    
outflows_aggr_1989=np.sum(outflows_all_hist[1:T],axis=1).astype(float)
#
#
x = outflows_aggr_1989
y = outflows_aggr
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
outflows_R2 = r_val**2
print(outflows_R2)
#
#plt.figure()
#plt.plot(x)
#plt.plot(y)
#plt.title('Aggregate Outflows 89-07')
#plt.legend(['Historical aggregate outflows','Simulated aggregate outflows'])
#plt.xlabel('doy')
#plt.ylabel('Outflows ($m^3$/s)')
#plt.text(150,3000,'$R^2$={}'.format(round(outflows_R2,4)),fontsize=12)
#py.savefig('Figures/aggregated_89_07.png', bbox_inches='tight')
#

#%% CP
cp_simulated=pd.DataFrame(cp_discharge_all,columns=cp_list)
cp_hist=pd.DataFrame(np.column_stack((CP[0].dis,CP[1].dis , CP[2].dis,CP[3].dis,CP[4].dis,CP[5].dis,CP[6].dis,CP[7].dis,CP[8].dis,CP[9].dis,CP[10].dis,CP[11].dis)),columns=cp_list)
#for cp in cp_list:
#    plt.figure()
#    plt.plot(cp_simulated[cp])
#    plt.plot(cp_hist[cp])
#    plt.title('Discharge at %s 89-07'%cp)
#    plt.legend(['Simulated','Historical'])
#    plt.xlabel('doy')
#    plt.ylabel('cms')
#    py.savefig('Figures/%s_89_07.png'%cp, bbox_inches='tight')

#%%SAVE results
writer = pd.ExcelWriter('Output/ControlPoints_89_07.xlsx')
cp_simulated.to_excel(writer,'simulated')
cp_hist.to_excel(writer,'historical')
writer.save()

res_simulated=pd.DataFrame(outflows_all[0:T-1,:],columns=res_list)
res_hist=pd.DataFrame(np.column_stack((RES[0].dataOUT[0:T-1,1],RES[1].dataOUT[0:T-1,1] , RES[2].dataOUT[0:T-1,1],RES[3].dataOUT[0:T-1,1],RES[4].dataOUT[0:T-1,1],RES[5].dataOUT[0:T-1,1],RES[6].dataOUT[0:T-1,1],RES[7].dataOUT[0:T-1,1],RES[8].dataOUT[0:T-1,1],RES[9].dataOUT[0:T-1,1],RES[10].dataOUT[0:T-1,1],RES[11].dataOUT[0:T-1,1],RES[12].dataOUT[0:T-1,1])),columns=res_list)

writer = pd.ExcelWriter('Output/Res_outflows_89_07.xlsx')
res_simulated.to_excel(writer,'simulated')
res_hist.to_excel(writer,'historical')
writer.save()


