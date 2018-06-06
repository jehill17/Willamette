# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:09:37 2018

@authors: Joy Hill, Simona Denaro
"""

#this file will be the outer shell of the Willamette code

#this code will take the inflows as an input (at various timesteps)
#use balance eqns for reservoirs and control points

#initialize reservoirs
#running the inner function for each reservoir along with the timing of the routing

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

#%%  load reservoirs and control point infos
with open('Flow.xml') as fd:
    flow = xmld.parse(fd.read())

flow_model = flow["flow_model"]

controlPoints = flow_model["controlPoints"]['controlPoint']

reservoirs = flow_model["reservoirs"]['reservoir']

#Create Reservoir class
class Reservoir:
    def __init__(self, ID):
        self.ID=ID
        self.Restype=[]
        self.AreaVolCurve=pd.DataFrame
        self.RuleCurve=pd.DataFrame
        self.Composite=pd.DataFrame
        self.RO=pd.DataFrame
        self.RulePriorityTable=pd.DataFrame
        self.Buffer=pd.DataFrame
        self.Spillway=pd.DataFrame
        self.ruleDir =str()
        self.InitVol=[]
        self.init_outflow =[]
        self.minOutflow=[]
        self.maxVolume=[]
        self.Td_elev=[]
        self.inactive_elev=[]
        self.Fc1_elev=[]
        self.Fc2_elev=[]
        self.Fc3_elev=[]
        self.GateMaxPowerFlow=[]
        self.maxPowerFlow=[]
        self.maxRO_Flow =[]
        self.maxSpillwayFlow=[]
        self.minPowerFlow=0
        self.minRO_Flow =0
        self.minSpillwayFlow=0
        self.Tailwater_elev=[]
        self.Turbine_eff=[]
        self.R2 = float()
        self.zone=1
   

#Create ControlPoint class
class ControlPoint:
    def __init__(self, ID):
        self.ID=ID
        self.COMID=str()
        self.influencedReservoirs=[]



#RESERVOIR rules
#in order of RES ID
res_list =('HCR', 'LOP', 'DEX', 'FAL', 'DOR', 'COT', 'FRN', 'CGR', 'BLU', 'GPR', 'FOS', 'DET', 'BCL')
RES = [Reservoir(id) for id in range(1,len(res_list)+1)]

for res in RES:
    id = res.ID
    res.name = res_list[id-1]
    res.Restype = str(reservoirs[id-1]['@reservoir_type'])
    res.AreaVolCurve=pd.read_csv(os.path.join('Area_Capacity_Curves/', str(reservoirs[id-1]['@area_vol_curve'])))
    res.Composite=pd.read_csv(os.path.join('Rel_Cap/', str(reservoirs[id-1]['@composite_rc'])))
    res.RO=pd.read_csv(os.path.join('Rel_Cap/', str(reservoirs[id-1]['@RO_rc'])))
    res.Spillway=pd.read_csv(os.path.join('Rel_Cap/', str(reservoirs[id-1]['@spillway_rc'])))
    res.InitVol=float(reservoirs[id-1]["@initVolume"])
    #res.InitVol=float(reservoirs[id-1]["@initOutflow"]) TO BE ADDED in the xml file
    res.minOutflow=float(reservoirs[id-1]["@minOutflow"])
    res.inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
    res.GateMaxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"])
    res.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
    res.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
    if res.Restype != "RunOfRiver":
        res.ruleDir=str(reservoirs[id-1]["@rp_dir"])
        res.cpDir=str(reservoirs[id-1]["@cp_dir"])
        res.RuleCurve=pd.read_csv(os.path.join('Rule_Curves/', str(reservoirs[id-1]['@rule_curve'])))
        res.RulePriorityTable=pd.read_csv(os.path.join('Rule_Priorities/', str(reservoirs[id-1]['@rule_priorities'])))
        res.Buffer=pd.read_csv(os.path.join('Rule_Curves/', str(reservoirs[id-1]['@buffer_zone'])))
        res.maxVolume=float(reservoirs[id-1]["@maxVolume"])
        res.Td_elev=float(reservoirs[id-1]["@td_elev"])
        res.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
        res.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
        res.Fc2_elev=float(reservoirs[id-1]["@fc3_elev"])
       
cp_list =['SAL', 'ALB', 'JEF', 'MEH', 'HAR', 'VID', 'JAS', 'GOS', 'WAT', 'MON', 'FOS']
CP = [ControlPoint(id) for id in range(1, len(cp_list)+1)]

for cp in CP:
    id = cp.ID
    cp.name = cp_list[id-1]
    cp.influencedReservoirs =np.array((str(controlPoints[id-1]["@reservoirs"]).split(',')))
    cp.COMID=str(controlPoints[id-1]["@location"])


#import control point historical data-- shifted one day before

#convert data
cfs_to_cms = 0.0283168
 

#reservoirs:
#read in historical reservoir inflows -- this will contain the array of 'dates' to use
BLU5Ad = pd.read_excel('Data/BLU5A_daily.xls',skiprows=27942,skip_footer =1004) #only using data from 2005
BLU5Ad.columns = ['Date','Inflow']
BLU5A = pd.read_excel('Data/BLU5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
CGR5A = pd.read_excel('Data/CGR5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
DET5A = pd.read_excel('Data/DET5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
DOR5A = pd.read_excel('Data/DOR5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
FAL5A = pd.read_excel('Data/FAL5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
FOS5A = pd.read_excel('Data/FOS5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
FRN5M = pd.read_excel('Data/FRN5M_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
GPR5A = pd.read_excel('Data/GPR5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
HCR5A = pd.read_excel('Data/HCR5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
LOP5A = pd.read_excel('Data/LOP5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
LOP5E = pd.read_excel('Data/LOP5E_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
COT5A = pd.read_excel('Data/COT5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
FOS_loc = pd.read_excel('Data/FOS_loc.xls',usecols = [0,3],skiprows=27942,skip_footer =1004)*cfs_to_cms
LOP_loc = pd.read_excel('Data/LOP_loc.xls',usecols = [0,3],skiprows=27942,skip_footer =1004)*cfs_to_cms


#historical outflows 
BLU5H = np.array(pd.read_excel('Data/BLU5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms) #only using data from 2005
BCL5H = np.array(pd.read_excel('Data/BCL5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms) #only using data from 2005
CGR5H = np.array(pd.read_excel('Data/CGR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DET5H = np.array(pd.read_excel('Data/DET5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DEX5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DOR5H = np.array(pd.read_excel('Data/DOR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FAL5H = np.array(pd.read_excel('Data/FAL5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FRN5H = np.array(pd.read_excel('Data/FRN5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
GPR5H = np.array(pd.read_excel('Data/GPR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
HCR5H = np.array(pd.read_excel('Data/HCR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
COT5H = np.array(pd.read_excel('Data/COT5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)

outflows_2005_all = np.stack((HCR5H[:,1],LOP5H[:,1],DEX5H[:,1],FAL5H[:,1],DOR5H[:,1],COT5H[:,1],FRN5H[:,1],CGR5H[:,1],BLU5H[:,1],GPR5H[:,1],FOS5H[:,1],DET5H[:,1],BCL5H[:,1]),axis=1)
outflows_2005_wo_FRN = np.stack((HCR5H[:,1],LOP5H[:,1],DEX5H[:,1],FAL5H[:,1],DOR5H[:,1],COT5H[:,1],CGR5H[:,1],BLU5H[:,1],GPR5H[:,1],FOS5H[:,1],DET5H[:,1],BCL5H[:,1]),axis=1)


#reading in historical volumes
M3_PER_ACREFT = 1233.4

#HCR
HCR2005_vol = pd.read_excel('Data/HCRvolume_2005.xlsx')
HCR2005_vol.columns = ['Date','Time','Storage(AF)']
HCR2005vol_d = np.array(HCR2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#LOP
LOP2005_vol = pd.read_excel('Data/LOPvolume_2005.xlsx')
LOP2005_vol.columns = ['Date','Time','Storage(AF)']
LOP2005vol_d = np.array(LOP2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#FAL
FAL2005_vol = pd.read_excel('Data/FALvolume_2005.xlsx')
FAL2005_vol.columns = ['Date','Time','Storage(AF)']
FAL2005vol_d = np.array(FAL2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#DOR
DOR2005_vol = pd.read_excel('Data/DORvolume_2005.xlsx')
DOR2005_vol.columns = ['Date','Time','Storage(AF)']
DOR2005vol_d = np.array(DOR2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#COT
COT2005_vol = pd.read_excel('Data/COTvolume_2005.xlsx')
COT2005_vol.columns = ['Date','Time','Storage(AF)']
COT2005vol_d = np.array(COT2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#FRN
FRN2005_vol = pd.read_excel('Data/FRNvolume_2005.xlsx')
FRN2005_vol.columns = ['Date','Time','Storage(AF)']
FRN2005vol_d = np.array(FRN2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#after looking at the query, I have discovered that 2005 was quite an unusual year for FRN. Volume curve does not follow normal pattern.

#CGR
CGR2005_vol = pd.read_excel('Data/CGRvolume_2005.xlsx')
CGR2005_vol.columns = ['Date','Time','Storage(AF)']
CGR2005vol_d = np.array(CGR2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#BLU
BLU2005_vol = pd.read_excel('Data/BLUvolume_2005.xlsx')
BLU2005_vol.columns = ['Date','Time','Storage(AF)']
BLU2005vol_d = np.array(BLU2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)


#DET
DET2005_vol = pd.read_excel('Data/DETvolume_2005.xlsx')
DET2005_vol.columns = ['Date','Time','Storage(AF)']
DET2005vol_d = np.array(DET2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#BCL
BCL2005_vol = pd.read_excel('Data/BCLvolume_2005.xlsx') #this should be relatively constant, but need to extract initial values
BCL2005_vol.columns = ['Date','Time','Storage(AF)']
BCL2005vol_d = np.array(BCL2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#FOS
FOS2005_vol = pd.read_excel('Data/FOSvolume_2005.xlsx')
FOS2005_vol.columns = ['Date','Time','Storage(AF)']
FOS2005vol_d = np.array(FOS2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#GPR
GPR2005_vol = pd.read_excel('Data/GPRvolume_2005.xlsx')
GPR2005_vol.columns = ['Date','Time','Storage(AF)']
GPR2005vol_d = np.array(GPR2005_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)



#control points
#cp_hist discharge: start this at 12/31/2004
filename='Data/Control point historical discharge 2005.xlsx'
SAL_2005 = pd.read_excel(filename,sheetname='Salem')
SAL_2005_dis = np.array(SAL_2005['Discharge'])*cfs_to_cms
ALB_2005 = pd.read_excel(filename,sheetname='Albany')
ALB_2005_dis = np.array(ALB_2005['Discharge'])*cfs_to_cms
JEF_2005 = pd.read_excel(filename,sheetname='Jefferson')
JEF_2005_dis = np.array(JEF_2005['Discharge'])*cfs_to_cms
MEH_2005 = pd.read_excel(filename,sheetname='Mehama')
MEH_2005_dis = np.array(MEH_2005['Discharge'])*cfs_to_cms
HAR_2005 = pd.read_excel(filename,sheetname='Harrisburg')
HAR_2005_dis = np.array(HAR_2005['Discharge'])*cfs_to_cms
VID_2005 = pd.read_excel(filename,sheetname='Vida')
VID_2005_dis = np.array(VID_2005['Discharge'])*cfs_to_cms
JAS_2005 = pd.read_excel(filename,sheetname='Jasper')
JAS_2005_dis = np.array(JAS_2005['Discharge'])*cfs_to_cms
GOS_2005 = pd.read_excel(filename,sheetname='Goshen')
GOS_2005_dis = np.array(GOS_2005['Discharge'])*cfs_to_cms
WAT_2005 = pd.read_excel(filename,sheetname='Waterloo')
WAT_2005_dis = np.array(WAT_2005['Discharge'])*cfs_to_cms
MON_2005 = pd.read_excel(filename,sheetname='Monroe')
MON_2005_dis = np.array(MON_2005['Discharge'])*cfs_to_cms
FOS_2005 = pd.read_excel(filename,sheetname='Foster')
FOS_2005_dis = np.array(FOS_2005['Discharge'])*cfs_to_cms
cp_discharge_2005_all = np.stack((SAL_2005_dis,ALB_2005_dis,JEF_2005_dis,MEH_2005_dis,HAR_2005_dis,VID_2005_dis,JAS_2005_dis,GOS_2005_dis,WAT_2005_dis,MON_2005_dis,FOS_2005_dis),axis=1)    



#cp local flows, starts at 1/1/2005
ALB_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Albany',skiprows=5844,skip_footer=730)
ALB_loc.columns = ['Date','Local Flow']
SAL_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Salem',skiprows=5844,skip_footer=730)
SAL_loc.columns = ['Date','Local Flow']
JEF_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Jefferson',skiprows=5844,skip_footer=730)
JEF_loc.columns = ['Date','Local Flow']
MEH_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Mehama',skiprows=5844,skip_footer=730)
MEH_loc.columns = ['Date','Local Flow']
HAR_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Harrisburg',skiprows=5844,skip_footer=730)
HAR_loc.columns = ['Date','Local Flow']
VID_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Vida',skiprows=5844,skip_footer=730)
VID_loc.columns = ['Date','Local Flow']
JAS_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Jasper',skiprows=5844,skip_footer=730)
JAS_loc.columns = ['Date','Local Flow']
GOS_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Goshen',skiprows=5844,skip_footer=730)
GOS_loc.columns = ['Date','Local Flow']
WAT_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Waterloo',skiprows=5844,skip_footer=730)
WAT_loc.columns = ['Date','Local Flow']
MON_loc = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname='Monroe',skiprows=5844,skip_footer=730)
MON_loc.columns = ['Date','Local Flow']

dates = np.array(BLU5Ad['Date'])

#%% Allocate and initialize
T = 365 # Set the simulation horizon

n_res=13
n_HPres=8
n_cp = 11


#=======
# allocate output 
outflows_all = np.full((T+2,n_res),np.nan) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.full((T+2,n_HPres), np.nan)
volumes_all = np.full((T+2,n_res),np.nan)
elevations_all = np.full((T+2,n_res),np.nan)
cp_discharge_all = np.full((T+2,(n_cp)),np.nan)

#initialize values
for  i in range(0,n_res):
    outflows_all[0:3,i] = outflows_2005_all[0:3,i] #remember to stack outflows historical values
    volumes_all[0:3,i] = np.tile(RES[i].InitVol,(3)) #TO BE CHANGED!
    elevations_all[0:3,i]=inner.GetPoolElevationFromVolume(volumes_all[0:3,i],RES[i])
    
volumes_all[0:3,9] = GPR2005vol_d[0:3]
volumes_all[0:3,10] = FOS2005vol_d[0:3]
volumes_all[0:3,11] = DET2005vol_d[0:3]
volumes_all[0:3,12] = BCL2005vol_d[0:3]



for  i in range(0,n_cp):
     cp_discharge_all[0,i] = cp_discharge_2005_all[0,i]

#define an outer fnt here that takes date, name, vol as inputs?

InitwaterYear = 1.2
waterYear = InitwaterYear
#%% Daily loop
for t in range(1,T+2):    
    doy = inner.DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')
    
    if doy==120:
        waterYear = inner.UpdateReservoirWaterYear(doy,t,volumes_all) #function missing
    
    #COTTAGE GROVE ID=6 count=5 NO HYDROPOWER
    COT = RES[5]
    COT_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,COT.ID-1],COT)
    [COT_outflow, _,_,_, COT.zone] = inner.GetResOutflow(COT,volumes_all[t-1,COT.ID-1],COT5A.iloc[t,1],outflows_all[t-1,COT.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],COT.zone)
    [COT_volume,COT_elevation] = inner.UpdateVolume_elev (COT, COT5A.iloc[t,1], COT_outflow, volumes_all[t-1,COT.ID-1])
    
    outflows_all[t,COT.ID-1] = COT_outflow 
    volumes_all[t,COT.ID-1] =  COT_volume
    elevations_all[t,COT.ID-1]=  COT_elevation


    #DORENA ID=5 count=4 NO HYDROPOWER
    DOR = RES[4]
    DOR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,DOR.ID-1],DOR)
    [DOR_outflow, _,_,_, DOR.zone] = inner.GetResOutflow(DOR,volumes_all[t-1,DOR.ID-1],DOR5A.iloc[t,1],outflows_all[t-1,DOR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],DOR.zone)
    [DOR_volume,DOR_elevation] = inner.UpdateVolume_elev (DOR, DOR5A.iloc[t,1], DOR_outflow,volumes_all[t-1,DOR.ID-1])
    
    outflows_all[t,DOR.ID-1] = DOR_outflow 
    volumes_all[t,DOR.ID-1] =  DOR_volume
    elevations_all[t,DOR.ID-1]=  DOR_elevation
    
    #FERN RIDGE ID=7 count=6 NO HYDROPOWER
    FRN = RES[6]
    FRN_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,FRN.ID-1],FRN)
    [FRN_outflow,_,_,_, FRN.zone] = inner.GetResOutflow(FRN,volumes_all[t-1,FRN.ID-1],FRN5M.iloc[t,1],outflows_all[t-1,FRN.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],FRN.zone)
    [FRN_volume,FRN_elevation] = inner.UpdateVolume_elev (FRN, FRN5M.iloc[t,1], FRN_outflow,volumes_all[t-1,FRN.ID-1])
    
    outflows_all[t,FRN.ID-1] = FRN_outflow 
    volumes_all[t,FRN.ID-1] =  FRN_volume
    elevations_all[t,FRN.ID-1]=  FRN_elevation
    
    #HILLS CREEK ID=1 count =0
    HCR = RES[0]
    HCR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,HCR.ID-1],HCR)
    [HCR_outflow,powerFlow,RO_flow,spillwayFlow, HCR.zone] = inner.GetResOutflow(HCR,volumes_all[t-1,HCR.ID-1],HCR5A.iloc[t,1],outflows_all[t-1,HCR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],HCR.zone)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    [HCR_volume,HCR_elevation] = inner.UpdateVolume_elev (HCR, HCR5A.iloc[t,1], HCR_outflow,volumes_all[t-1,HCR.ID-1])
    
    outflows_all[t,HCR.ID-1] = HCR_outflow 
    volumes_all[t,HCR.ID-1] =  HCR_volume
    elevations_all[t,HCR.ID-1]=  HCR_elevation
    hydropower_all[t,0] = HCR_power_output
    
    #LOOKOUT POINT ID=2 count=1
    LOP = RES[1]
    LOP_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,LOP.ID-1],LOP)
    LOP_inflow =  HCR_outflow + LOP_loc.iloc[t,1] + LOP5E.iloc[t,1] #balance equation
    [LOP_outflow,powerFlow,RO_flow,spillwayFlow, LOP.zone]  = inner.GetResOutflow(LOP,volumes_all[t-1,LOP.ID-1],LOP_inflow,outflows_all[t-1,LOP.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],LOP.zone)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    [LOP_volume,LOP_elevation] = inner.UpdateVolume_elev (LOP, LOP5A.iloc[t,1], LOP_outflow,volumes_all[t-1,LOP.ID-1])
    
    outflows_all[t,LOP.ID-1] = LOP_outflow 
    volumes_all[t,LOP.ID-1] =  LOP_volume
    elevations_all[t,LOP.ID-1]=  LOP_elevation
    hydropower_all[t,1] = LOP_power_output
    
    #DEXTER ID=3 count=2
    DEX = RES[2]
    DEX_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,DEX.ID-1],DEX)
    [DEX_outflow,powerFlow,RO_flow,spillwayFlow, DEX.zone] = inner.GetResOutflow(DEX,volumes_all[t-1,DEX.ID-1],LOP_outflow,outflows_all[t-1,DEX.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],DEX.zone)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    [DEX_volume,DEX_elevation] = inner.UpdateVolume_elev (DEX, LOP_outflow, DEX_outflow,volumes_all[t-1,DEX.ID-1])
    
    outflows_all[t,DEX.ID-1] = DEX_outflow 
    volumes_all[t,DEX.ID-1] =  DEX_volume
    elevations_all[t,DEX.ID-1]=  DEX_elevation
    hydropower_all[t,2] = DEX_power_output   
    
    #FALL CREEK ID=4 count=3 NO HYDROPOWER
    FAL = RES[3]
    FAL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,FAL.ID-1],FAL)
    [FAL_outflow, _,_,_, FAL.zone] = inner.GetResOutflow(FAL,volumes_all[t-1,FAL.ID-1],FAL5A.iloc[t,1],outflows_all[t-1,FAL.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],FAL.zone)
    [FAL_volume,FAL_elevation] = inner.UpdateVolume_elev (FAL, FAL5A.iloc[t,1], FAL_outflow,volumes_all[t-1,FAL.ID-1])
    
    outflows_all[t,FAL.ID-1] = FAL_outflow 
    volumes_all[t,FAL.ID-1] =  FAL_volume
    elevations_all[t,FAL.ID-1]=  FAL_elevation
    
    #COUGAR ID=8 count=7
    CGR = RES[7]
    CGR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,CGR.ID-1],CGR)
    [CGR_outflow,powerFlow,RO_flow,spillwayFlow, CGR.zone] = inner.GetResOutflow(CGR,volumes_all[t-1,CGR.ID-1],CGR5A.iloc[t,1],outflows_all[t-1,CGR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],CGR.zone)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    [CGR_volume,CGR_elevation] = inner.UpdateVolume_elev (CGR, CGR5A.iloc[t,1], CGR_outflow,volumes_all[t-1,CGR.ID-1])
    
    outflows_all[t,CGR.ID-1] = CGR_outflow 
    volumes_all[t,CGR.ID-1] =  CGR_volume
    elevations_all[t,CGR.ID-1]=  CGR_elevation
    hydropower_all[t,3] = CGR_power_output  

    
    #BLUE RIVER ID=9 count= 8 NO HYDROPOWER
    BLU = RES[8]
    BLU_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,BLU.ID-1],BLU)
    [BLU_outflow,_,_,_, BLU.zone]  = inner.GetResOutflow(BLU,volumes_all[t-1,BLU.ID-1],BLU5A.iloc[t,1],outflows_all[t-1,BLU.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],BLU.zone)
    [BLU_volume,BLU_elevation] = inner.UpdateVolume_elev (BLU, BLU5A.iloc[t,1], BLU_outflow,volumes_all[t-1,BLU.ID-1])
    
    outflows_all[t,BLU.ID-1] = BLU_outflow 
    volumes_all[t,BLU.ID-1] =  BLU_volume
    elevations_all[t,BLU.ID-1]=  BLU_elevation    
    
    
    #the next reservoirs are at time "t+2" #check!!!
    #GREEN PETER ID=10 count=9
    GPR = RES[9]
    GPR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,GPR.ID-1],GPR)
    [GPR_outflow, powerFlow,RO_flow,spillwayFlow,GPR.zone] = inner.GetResOutflow(GPR,volumes_all[t+1,GPR.ID-1],GPR5A.iloc[t+2,1],outflows_all[t+1,GPR.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],GPR.zone)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    [GPR_volume,GPR_elevation] = inner.UpdateVolume_elev (GPR, GPR5A.iloc[t+2,1], GPR_outflow,volumes_all[t+1,GPR.ID-1])
    
    outflows_all[t+2,GPR.ID-1] = GPR_outflow 
    volumes_all[t+2,GPR.ID-1] =  GPR_volume
    elevations_all[t+2,GPR.ID-1]=  GPR_elevation
    hydropower_all[t+2,4] = GPR_power_output      
    
    
    #FOSTER ID=11 count=10
    FOS = RES[10]
    FOS_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,FOS.ID-1],FOS)
    FOS_inflow =GPR_outflow + FOS_loc.iloc[t+2,1] #balance equation
    [FOS_outflow, powerFlow,RO_flow,spillwayFlow, FOS.zone] = inner.GetResOutflow(FOS,volumes_all[t+1,FOS.ID-1],FOS_inflow,outflows_all[t+1,FOS.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],FOS.zone)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    [FOS_volume,FOS_elevation] = inner.UpdateVolume_elev (FOS, FOS5A.iloc[t+2,1], FOS_outflow,volumes_all[t+1,FOS.ID-1])
    
    outflows_all[t+2,FOS.ID-1] = FOS_outflow 
    volumes_all[t+2,FOS.ID-1] =  FOS_volume
    elevations_all[t+2,FOS.ID-1]=  FOS_elevation
    hydropower_all[t+2,5] = FOS_power_output   
    
    
    #DETROIT ID=12 count=11
    DET = RES[11]
    DET_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,DET.ID-1],DET)
    [DET_outflow, powerFlow,RO_flow,spillwayFlow, DET.zone] = inner.GetResOutflow(DET,volumes_all[t+1,DET.ID-1],DET5A.iloc[t+2,1],outflows_all[t+1,DET.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],DET.zone)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    [DET_volume,DET_elevation] = inner.UpdateVolume_elev (DET, DET5A.iloc[t+2,1], DET_outflow,volumes_all[t+1,DET.ID-1])
    
    outflows_all[t+2,DET.ID-1] = DET_outflow 
    volumes_all[t+2,DET.ID-1] =  DET_volume
    elevations_all[t+2,DET.ID-1]=  DET_elevation
    hydropower_all[t+2,6] = DET_power_output   
        
    #BIG CLIFF ID=13 count=12
    BCL = RES[12]
    BCL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,BCL.ID-1],BCL)
    [BCL_outflow, powerFlow,RO_flow,spillwayFlow, BCL.zone] = inner.GetResOutflow(BCL,volumes_all[t+1,BCL.ID-1],DET_outflow,outflows_all[t+1,BCL.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],BCL.zone)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    [BCL_volume,BCL_elevation] = inner.UpdateVolume_elev (BCL, DET_outflow, BCL_outflow,volumes_all[t+1,BCL.ID-1])
    
    outflows_all[t+2,BCL.ID-1] = BCL_outflow 
    volumes_all[t+2,BCL.ID-1] =  BCL_volume
    elevations_all[t+2,BCL.ID-1]=  BCL_elevation
    hydropower_all[t+2,7] = BCL_power_output


    #UPDATE CONTROL POINTS DISCHARGE
    #add in balance eqns
    #cp_list =['SAL', 'ALB', 'JEF', 'MEH', 'HAR', 'VID', 'JAS', 'GOS', 'WAT', 'MON', 'FOS']
    
#SALin(t) = ALBin(t) + JEFin(t) + SALloc(t)
#JEFin(t) = WATin(t) +MEHin(t) + JEFloc(t)
#MEHin(t) = BCL5H(t) +MEHloc(t)
#WATin = FOS5H(t) +WATloc(t)
#ALBin(t) = MONin(t 􀀀 1) + HARin(t 􀀀 1) + ALBloc(t)
#MONin(t) = FER5H(t) +MONloc(t)
#HARin(t) = VIDin(t) + GOSin(t) + JASin(t) + HARloc(t)
#V IDin(t) = CGR5H(t) + BLU5H(t) + V IDloc(t)
#JASin(t) = DEX5H(t) + FAL5H(t) + JASloc(t)
#GOSin(t) = COT5H(t) + DOR5H(t) + GOSloc(t)

    #in order of upstream to down
    
    #GOSHEN
    cp_discharge_all[t,7] = COT_outflow + DOR_outflow + GOS_loc.iloc[t,1]    
    #JASPER
    cp_discharge_all[t,6] = DEX_outflow + FAL_outflow + JAS_loc.iloc[t,1]    
    #VIDA
    cp_discharge_all[t,5] = CGR_outflow + BLU_outflow + VID_loc.iloc[t,1]    
    #HARRISBURG
    cp_discharge_all[t,4] = cp_discharge_all[t,7] + cp_discharge_all[t,6] + cp_discharge_all[t,5] + HAR_loc.iloc[t,1]    
    #MONROE
    cp_discharge_all[t,9] = FRN_outflow + MON_loc.iloc[t,1]    
    #ALBANY
    cp_discharge_all[t,1] = cp_discharge_all[t-1,9] + cp_discharge_all[t-1,4]+ ALB_loc.iloc[t,1]    
    #WATERLOO
    cp_discharge_all[t,8] = FOS_outflow + WAT_loc.iloc[t,1]    
    #MEHAMA
    cp_discharge_all[t,3] = BCL_outflow + MEH_loc.iloc[t,1]    
    #JEFFERSON
    cp_discharge_all[t,2] = cp_discharge_all[t,8] + cp_discharge_all[t,3] + JEF_loc.iloc[t,1]    
    #SALEM
    cp_discharge_all[t,0] = cp_discharge_all[t,1] + cp_discharge_all[t,2] + SAL_loc.iloc[t,1]   
    #FOSTER ID=11 count=10
    cp_discharge_all[t,10] = FOS_outflow
    
#    for j in range(0,n_cp):
#        if cp_discharge_all[t,j]<0:
#            print("cp discharge is negative for CP=",CP[j].name," t=", t)
            
#%%    
#historical validation -- HYDRO AND OUTFLOWS
Willamette_gen_2005 = pd.read_excel('Data/Williamette_historical_hydropower_gen.xlsx',sheetname='2005',skiprows=4) #this is hourly
Willamette_gen_2005.columns = ['CGR','DET','DEX','FOS','GPR','HCR','LOP','BCL']




#CGR hydro
CGR_2005 = Willamette_gen_2005['CGR']
CGR_2005_daily = np.mean(np.array(CGR_2005).reshape(-1,24),axis=1)

x = (CGR_2005_daily[1:365].astype('float'))
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
DET_2005 = Willamette_gen_2005['DET']
DET_2005_daily = np.mean(np.array(DET_2005).reshape(-1,24),axis=1)

x = (DET_2005_daily[3:365].astype('float'))
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
DEX_2005 = Willamette_gen_2005['DEX']
DEX_2005_daily = np.mean(np.array(DEX_2005).reshape(-1,24),axis=1)  

x = (DEX_2005_daily[1:365].astype('float'))
y = hydropower_all[1:365,2]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DEX.hydro_R2 = r_val**2
print(DEX.hydro_R2)

plt.figure()
plt.plot(DEX_2005_daily[1:365])
plt.plot(hydropower_all[1:365,2])
plt.title('Dexter Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(150,15,'$R^2$={}'.format(round(DEX.hydro_R2,4)),fontsize=12)



FOS_2005 = Willamette_gen_2005['FOS']
FOS_2005_daily = np.mean(np.array(FOS_2005).reshape(-1,24),axis=1)  

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
x = (FOS_2005_daily[3:365].astype('float'))
y = hydropower_all[3:365,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FOS.hydro_R2 = r_val**2
print(FOS.hydro_R2)

plt.figure()
plt.plot(FOS_2005_daily[3:365])
plt.plot(hydropower_all[3:365,5])
plt.title('Foster Hydropower Generation',fontsize=20)
plt.legend(['historical generation','simulated generation'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Generation (MWh)')
plt.text(200,25,'$R^2$={}'.format(round(FOS.hydro_R2,4)),fontsize=12)





GPR_2005 = Willamette_gen_2005['GPR']
GPR_2005_daily = np.mean(np.array(GPR_2005).reshape(-1,24),axis=1)  

x = (GPR_2005_daily[3:365].astype('float'))
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
HCR_2005 = Willamette_gen_2005['HCR']
HCR_2005_daily = np.mean(np.array(HCR_2005).reshape(-1,24),axis=1)  
x = (HCR_2005_daily[1:365].astype('float'))
y = hydropower_all[1:365,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.hydro_R2 = r_val**2
print(HCR.hydro_R2)

plt.figure()
plt.plot(HCR_2005_daily[1:365])
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
LOP_2005 = Willamette_gen_2005['LOP']
LOP_2005_daily = np.mean(np.array(LOP_2005).reshape(-1,24),axis=1)
x = (LOP_2005_daily[1:365].astype('float'))
y = hydropower_all[1:365,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.hydro_R2 = r_val**2
print(LOP.hydro_R2)

plt.figure()
plt.plot(LOP_2005_daily[1:365])
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
BCL_2005 = Willamette_gen_2005['BCL']
BCL_2005_daily = np.mean(np.array(BCL_2005).reshape(-1,24),axis=1)  

x = (BCL_2005_daily[3:365].astype('float'))
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

#HCR
x = HCR2005vol_d
y = volumes_all[0:365,0]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
HCR.vol_R2 = r_val**2
print(HCR.vol_R2)

plt.figure()
plt.plot(HCR2005vol_d)
plt.plot(volumes_all[0:365,0])
plt.title('Hills Creek Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(20,3.8e8,'$R^2$={}'.format(round(HCR.vol_R2,4)),fontsize=12)




#LOP
x = LOP2005vol_d
y = volumes_all[0:365,1]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
LOP.vol_R2 = r_val**2
print(LOP.vol_R2)

plt.figure()
plt.plot(LOP2005vol_d)
plt.plot(volumes_all[0:365,1])
plt.title('Lookout Point Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(20,4.5e8,'$R^2$={}'.format(round(LOP.vol_R2,4)),fontsize=12)

#FAL
x = FAL2005vol_d
y = volumes_all[0:365,3]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
FAL.vol_R2 = r_val**2
print(FAL.vol_R2)

plt.figure()
plt.plot(FAL2005vol_d)
plt.plot(volumes_all[0:365,3])
plt.title('Fall Creek Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,1.4e8,'$R^2$={}'.format(round(FAL.vol_R2,4)),fontsize=12)


#DOR
x = DOR2005vol_d
y = volumes_all[0:365,4]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
DOR.vol_R2 = r_val**2
print(DOR.vol_R2)

plt.figure()
plt.plot(DOR2005vol_d)
plt.plot(volumes_all[0:365,4])
plt.title('Dorena Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,0.8e8,'$R^2$={}'.format(round(DOR.vol_R2,4)),fontsize=12)

#COT
x = COT2005vol_d
y = volumes_all[0:365,5]
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
COT.vol_R2 = r_val**2
print(COT.vol_R2)

plt.figure()
plt.plot(COT2005vol_d)
plt.plot(volumes_all[0:365,5])
plt.title('Cottage Grove Reservoir Storage',fontsize=20)
plt.legend(['historical storage levels','simulated storage levels'],fontsize=12)
plt.xlabel('doy')
plt.ylabel('Storage ($m^3$)')
plt.text(25,3.5e7,'$R^2$={}'.format(round(COT.vol_R2,4)),fontsize=12)

#FRN
x = FRN2005vol_d
y = volumes_all[0:365,6]
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

#after looking at the query, I have discovered that 2005 was quite an unusual year for FRN. Volume curve does not follow normal pattern.

#CGR
x = CGR2005vol_d
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
x = BLU2005vol_d
y = volumes_all[0:365,8]
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
x = GPR2005vol_d[0:365]
y = volumes_all[0:365,9]
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
x = FOS2005vol_d[0:365]
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
x = DET2005vol_d[0:365]
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

outflows_aggr_2005 = np.sum(outflows_2005_all[1:365],axis=1).astype(float)

x = outflows_aggr_2005
y = outflows_aggr
slope,intercept,r_val,p_val,std_err = stats.linregress(x,y)
outflows_R2 = r_val**2
print(outflows_R2)

plt.figure()
plt.plot(outflows_aggr_2005)
plt.plot(outflows_aggr)
plt.title('Aggregate Outflows 2005')
plt.legend(['Historical aggregate outflows','Simulated aggregate outflows'])
plt.xlabel('doy')
plt.ylabel('Outflows ($m^3$/s)')
plt.text(150,1200,'$R^2$={}'.format(round(outflows_R2,4)),fontsize=12)


hydro = np.sum(np.array(Willamette_gen_2005),axis=1) #this is hourly
hydro_aggr_2005 = np.mean(hydro.reshape(-1,24),axis=1) 

hydro_aggr = np.sum(hydropower_all[3:365],axis=1) #this is only 362 days

plt.figure()
plt.plot(hydro_aggr_2005)
plt.plot(hydro_aggr)
plt.legend(['Historical aggregate hydro 2005','Simulated aggregate hydro 2005'])

#%%















  
    
    
    
    
    
