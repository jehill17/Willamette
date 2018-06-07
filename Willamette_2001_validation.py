# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 18:25:43 2018

@author: sdenaro
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 07 16:44:20 2018

@author: sdenaro
"""
# this file runs the Willamette operational model for year 2001 
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

runfile("Willamette_outer.py", "Flow_2001.xml")

#%%    
#historical validation -- HYDRO AND OUTFLOWS
Willamette_gen_2001 = pd.read_excel('Data/Williamette_historical_hydropower_gen.xlsx',sheetname='2001',skiprows=4) #this is hourly
Willamette_gen_2001.columns = ['CGR','DET','DEX','FOS','GPR','HCR','LOP','BCL']

#CGR hydro
CGR_2001 = Willamette_gen_2001['CGR']
CGR_2001_daily = np.mean(np.array(CGR_2001).reshape(-1,24),axis=1)

x = (CGR_2001_daily[1:365].astype('float'))
y = hydropower_all[1:365,3]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
CGR.hydro_R2 = r_val**2
print(CGR.hydro_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Cougar Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(100,30,'$R^2$={}'.format(round(CGR.hydro_R2,4)),fontsize=12)


#CGR outflows
x = (CGR5H[0:364,1].astype('float'))
y = outflows_all[0:364,7]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
CGR.R2 = r_val**2
print(CGR.R2)

plt.figure()
plt.plot(CGR5H[:,1])
plt.plot(outflows_all[0:364,7])
plt.title('Cougar Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(120,40,'$R^2$={}'.format(round(CGR.R2,4)),fontsize=15)


#DET hydro
DET_2001 = Willamette_gen_2001['DET']
DET_2001_daily = np.mean(np.array(DET_2001).reshape(-1,24),axis=1)

x = (DET_2001_daily[3:365].astype('float'))
y = hydropower_all[3:365,6]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DET.hydro_R2 = r_val**2
print(DET.hydro_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Detroit Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(200,70,'$R^2$={}'.format(round(DET.hydro_R2,4)),fontsize=12)


  
#DET outflows
x = (DET5H[0:363,1].astype('float'))
y = outflows_all[0:363,11]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DET.R2 = r_val**2
print(DET.R2)

plt.figure()
plt.plot(DET5H[:,1])
plt.plot(outflows_all[0:364,11])
plt.title('Detroit Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,150,'$R^2$={}'.format(round(DET.R2,4)),fontsize=15)


#DEX hydro
DEX_2001 = Willamette_gen_2001['DEX']
DEX_2001_daily = np.mean(np.array(DEX_2001).reshape(-1,24),axis=1)  

x = (DEX_2001_daily[1:365].astype('float'))
y = hydropower_all[1:365,2]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DEX.hydro_R2 = r_val**2
print(DEX.hydro_R2)

plt.figure()
plt.plot(DEX_2001_daily[1:365])
plt.plot(hydropower_all[1:365,2])
plt.title('Dexter Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(150,15,'$R^2$={}'.format(round(DEX.hydro_R2,4)),fontsize=12)



FOS_2001 = Willamette_gen_2001['FOS']
FOS_2001_daily = np.mean(np.array(FOS_2001).reshape(-1,24),axis=1)  

x = (FOS5H[0:364,1].astype('float'))
y = outflows_all[0:364,10]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FOS.R2 = r_val**2
#print(FOS.R2)
#FOS outflows
plt.figure()
plt.plot(FOS5H[0:364,1])
plt.plot(outflows_all[0:364,10])
plt.title('Foster Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(175,150,'$R^2$={}'.format(round(FOS.R2,4)),fontsize=15)

#FOS hydropower
x = (FOS_2001_daily[3:365].astype('float'))
y = hydropower_all[3:365,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FOS.hydro_R2 = r_val**2
print(FOS.hydro_R2)

plt.figure()
plt.plot(FOS_2001_daily[3:365])
plt.plot(hydropower_all[3:365,5])
plt.title('Foster Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(200,25,'$R^2$={}'.format(round(FOS.hydro_R2,4)),fontsize=12)





GPR_2001 = Willamette_gen_2001['GPR']
GPR_2001_daily = np.mean(np.array(GPR_2001).reshape(-1,24),axis=1)  

x = (GPR_2001_daily[3:365].astype('float'))
y = hydropower_all[3:365,4]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
GPR.hydro_R2 = r_val**2
print(GPR.hydro_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Green Peter Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(200,50,'$R^2$={}'.format(round(GPR.hydro_R2,4)),fontsize=12)

x = (GPR5H[0:363,1].astype('float'))
y = outflows_all[0:363,9]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
GPR.R2 = r_val**2
print(GPR.R2)

#GPR outflows
plt.figure()
plt.plot(GPR5H[0:363,1])
plt.plot(outflows_all[0:363,9])
plt.title('Green Peter Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(GPR.R2,4)),fontsize=15)



#HCR
HCR_2001 = Willamette_gen_2001['HCR']
HCR_2001_daily = np.mean(np.array(HCR_2001).reshape(-1,24),axis=1)  
x = (HCR_2001_daily[1:365].astype('float'))
y = hydropower_all[1:365,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.hydro_R2 = r_val**2
print(HCR.hydro_R2)

plt.figure()
plt.plot(HCR_2001_daily[1:365])
plt.plot(hydropower_all[1:365,0])
plt.title('Hills Creek Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(150,40,'$R^2$={}'.format(round(HCR.hydro_R2,4)),fontsize=12)

#HCR OUTFLOWS
x = (HCR5H[0:365,1].astype('float'))
y = outflows_all[0:365,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.R2 = r_val**2
print(HCR.R2)

plt.figure()
plt.plot(HCR5H[:,1])
plt.plot(outflows_all[0:364,0])
plt.title('Hills Creek Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,60,'$R^2$={}'.format(round(HCR.R2,4)),fontsize=15)


#LOP hydro validation
LOP_2001 = Willamette_gen_2001['LOP']
LOP_2001_daily = np.mean(np.array(LOP_2001).reshape(-1,24),axis=1)
x = (LOP_2001_daily[1:365].astype('float'))
y = hydropower_all[1:365,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.hydro_R2 = r_val**2
print(LOP.hydro_R2)

plt.figure()
plt.plot(LOP_2001_daily[1:365])
plt.plot(hydropower_all[1:365,1])
plt.title('Lookout Point Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(150,65,'$R^2$={}'.format(round(LOP.hydro_R2,4)),fontsize=12)  


#LOP outflows validation
x = (LOP5H[0:364,1].astype('float'))
y = outflows_all[0:364,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.R2 = r_val**2
print(LOP.R2)
plt.figure()
plt.plot(LOP5H[:,1])
plt.plot(outflows_all[0:364,1])
plt.title('Lookout Point Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(LOP.R2,4)),fontsize=15)



#BCL hydro valid
BCL_2001 = Willamette_gen_2001['BCL']
BCL_2001_daily = np.mean(np.array(BCL_2001).reshape(-1,24),axis=1)  

x = (BCL_2001_daily[3:365].astype('float'))
y = hydropower_all[3:365,7]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BCL.hydro_R2 = r_val**2
print(BCL.hydro_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Big Cliff Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(200,17,'$R^2$={}'.format(round(BCL.hydro_R2,4)),fontsize=12)

#BCL outflows valid
x = (BCL5H[0:363,1].astype('float'))
y = outflows_all[0:363,12]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BCL.R2 = r_val**2
print(BCL.R2)

plt.figure()
plt.plot(BCL5H[0:363,1])
plt.plot(outflows_all[0:363,12])
plt.title('Big Cliff Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(150,150,'$R^2$={}'.format(round(BCL.R2,4)),fontsize=15)


#nonhydro outflows

#COT
x = (COT5H[0:365,1].astype('float'))
y = outflows_all[0:365,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
COT.R2 = r_val**2
print(COT.R2)

plt.figure()
plt.plot(COT5H[0:365,1])
plt.plot(outflows_all[0:365,5])
plt.title('Cottage Grove Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,30,'$R^2$={}'.format(round(COT.R2,4)),fontsize=15)


#DOR
x = (DOR5H[0:365,1].astype('float'))
y = outflows_all[0:365,4]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DOR.R2 = r_val**2
print(DOR.R2)
plt.figure()
plt.plot(DOR5H[0:365,1])
plt.plot(outflows_all[0:365,4])
plt.title('Dorena Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(200,80,'$R^2$={}'.format(round(DOR.R2,4)),fontsize=15)



#FRN
x = (FRN5H[0:364,1].astype('float'))
y = outflows_all[0:364,6]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FRN.R2 = r_val**2
print(FRN.R2)

plt.figure()
plt.plot(FRN5H[0:364,1])
plt.plot(outflows_all[0:364,6])
plt.title('Fern Ridge Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')


#FAL
x = (FAL5H[0:365,1].astype('float'))
y = outflows_all[0:365,3]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FAL.R2 = r_val**2
print(FAL.R2)

plt.figure()
plt.plot(FAL5H[0:365,1])
plt.plot(outflows_all[0:365,3])
plt.title('Fall Creek Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(160,40,'$R^2$={}'.format(round(FAL.R2,4)),fontsize=15)

#BLU
x = (BLU5H[0:364,1].astype('float'))
y = outflows_all[0:364,8]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BLU.R2 = r_val**2
print(BLU.R2)

plt.figure()
plt.plot(BLU5H[0:364,1])
plt.plot(outflows_all[0:364,8])
plt.title('Blue River Reservoir Outflows',fontsize=20)
plt.legend(['historical outflows','simulated outflows'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Outflows (cubic meters/second)')
plt.text(160,60,'$R^2$={}'.format(round(BLU.R2,4)),fontsize=15)

#%%
#volume validation

#%%
#volume validation

#HCR
x = RES[0].histVol
y = volumes_all[0:365,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.vol_R2 = r_val**2
print(HCR.vol_R2)

plt.figure()
plt.plot(RES[0].histVol)
plt.plot(volumes_all[0:365,0])
plt.title('Hills Creek Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(20,3.8e8,'$R^2$={}'.format(round(HCR.vol_R2,4)),fontsize=12)


#LOP
x = RES[1].histVol
y = volumes_all[0:365,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.vol_R2 = r_val**2
print(LOP.vol_R2)

plt.figure()
plt.plot(RES[1].histVol)
plt.plot(volumes_all[0:365,1])
plt.title('Lookout Point Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(20,4.5e8,'$R^2$={}'.format(round(LOP.vol_R2,4)),fontsize=12)

#FAL
x = RES[3].histVol
y = volumes_all[0:361,3]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FAL.vol_R2 = r_val**2
print(FAL.vol_R2)

plt.figure()
plt.plot(RES[3].histVol)
plt.plot(volumes_all[0:361,3])
plt.title('Fall Creek Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,1.4e8,'$R^2$={}'.format(round(FAL.vol_R2,4)),fontsize=12)


#DOR
x = RES[4].histVol
y = volumes_all[0:363,4]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DOR.vol_R2 = r_val**2
print(DOR.vol_R2)

plt.figure()
plt.plot(RES[4].histVol)
plt.plot(volumes_all[0:363,4])
plt.title('Dorena Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,0.8e8,'$R^2$={}'.format(round(DOR.vol_R2,4)),fontsize=12)

#COT
x = RES[5].histVol
y = volumes_all[0:363,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
COT.vol_R2 = r_val**2
print(COT.vol_R2)

plt.figure()
plt.plot(RES[5].histVol)
plt.plot(volumes_all[0:363,5])
plt.title('Cottage Grove Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,3.5e7,'$R^2$={}'.format(round(COT.vol_R2,4)),fontsize=12)

#FRN
x = RES[6].histVol
y = volumes_all[0:363,6]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FRN.vol_R2 = r_val**2
print(FRN.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Fern Ridge Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,1.0e8,'$R^2$={}'.format(round(FRN.vol_R2,4)),fontsize=12)

#CGR
x = RES[7].histVol
y = volumes_all[0:365,7]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
CGR.vol_R2 = r_val**2
print(CGR.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Cougar Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,2.0e8,'$R^2$={}'.format(round(CGR.vol_R2,4)),fontsize=12)

#BLU
x = RES[8].histVol
y = volumes_all[0:364,8]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
BLU.vol_R2 = r_val**2
print(BLU.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Blue River Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,0.9e8,'$R^2$={}'.format(round(BLU.vol_R2,4)),fontsize=12)

#GPR
x = RES[9].histVol
y = volumes_all[0:345,9]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
GPR.vol_R2 = r_val**2
print(GPR.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Green Peter Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,4.5e8,'$R^2$={}'.format(round(GPR.vol_R2,4)),fontsize=12)


#FOS
x = RES[10].histVol[0:365]
y = volumes_all[0:365,10]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FOS.vol_R2 = r_val**2
print(FOS.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Foster Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,0.7e8,'$R^2$={}'.format(round(FOS.vol_R2,4)),fontsize=12)

#DET
x = RES[11].histVol[0:365]
y = volumes_all[0:365,11]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DET.vol_R2 = r_val**2
print(DET.vol_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Detroit Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,5.0e8,'$R^2$={}'.format(round(DET.vol_R2,4)),fontsize=12)


#%%
#aggregate outflows and hydropower validation
#outflows_minus_FRN = np.delete(outflows_all[0:364],6,1)
outflows_aggr = np.sum(outflows_all[0:364],axis=1)

outflows_aggr_2001 = np.sum(outflows_all_hist[1:365],axis=1).astype(float)

x = outflows_aggr_2001
y = outflows_aggr
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
outflows_R2 = r_val**2
print(outflows_R2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.title('Aggregate Outflows 2001')
plt.legend(['Historical aggregate outflows','Simulated aggregate outflows'])
plt.xlabel('doy')
plt.ylabel('Outflows ($m^3$/s)')
plt.text(150,1200,'$R^2$={}'.format(round(outflows_R2,4)),fontsize=12)


hydro = np.sum(np.array(Willamette_gen_2001),axis=1) #this is hourly
hydro_aggr_2001 = np.mean(hydro.reshape(-1,24),axis=1) 

hydro_aggr = np.sum(hydropower_all[3:365],axis=1) #this is only 362 days

plt.figure()
plt.plot(hydro_aggr_2001)
plt.plot(hydro_aggr)
plt.legend(['Historical aggregate hydro 2001','Simulated aggregate hydro 2001'])

#%%
