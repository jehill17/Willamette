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
import Willamette_model as inner #reading in the inner function
import os
import sys

#%%  load reservoirs and control point infos
#with open('Flow_2005.xml') as fd:
with open(str(sys.argv[1])) as fd:
    flow = xmld.parse(fd.read())

flow_model = flow["flow_model"]

controlPoints = flow_model["controlPoints"]['controlPoint']

reservoirs = flow_model["reservoirs"]['reservoir']

horizon=flow_model['simulation_horizon']

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
        self.histVol=[]
        self.init_outflow =[]
        self.minOutflow=[]
        self.maxVolume=[]
        self.Td_elev=[]
        self.inactive_elev=[]
        self.Fc1_elev=[]
        self.Fc2_elev=[]
        self.Fc3_elev=[]
        self.GateMaxPowerFlow=[]
        self.maxHydro=[]
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
    res.initVol_filename=str(reservoirs[id-1]["@initVol_filename"])
    res.initVol=[]
    res.minOutflow=float(reservoirs[id-1]["@minOutflow"])
    res.inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
    res.GateMaxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"])
    res.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
    res.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
    res.maxHydro = float(reservoirs[id-1]["@max_hydro_production"])
    res.filename_dataIN = str(reservoirs[id-1]['@filename_dataIN'])
    res.filename_dataEV = str(reservoirs[id-1]['@filename_dataEV'])
    res.filename_dataOUT = str(reservoirs[id-1]['@filename_dataOUT'])
    res.dataIN = []
    res.dataEV = []
    res.dataOUT=[]
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

HCR=RES[0]; LOP=RES[1]; DEX=RES[2]; FAL=RES[3]; DOR=RES[4]; COT=RES[5]; FRN=RES[6];
CGR=RES[7]; BLU=RES[8]; GPR=RES[9]; FOS=RES[10]; DET=RES[11]; BCL=RES[12]
  
cp_list =['SAL', 'ALB', 'JEF', 'MEH', 'HAR', 'VID', 'JAS', 'GOS', 'WAT', 'MON', 'FOS']
del id
CP = [ControlPoint(id) for id in range(1, len(cp_list)+1)]

for cp in CP:
    id = cp.ID
    cp.name=str(controlPoints[id-1]["@name"])
    cp.influencedReservoirs =np.array((str(controlPoints[id-1]["@reservoirs"]).split(',')))
    cp.COMID=str(controlPoints[id-1]["@location"])
    cp.dis=[]
    cp.loc=[]

SAL=CP[0]; ALB=CP[1]; JEF=CP[2]; MEH=CP[3]; HAR=CP[4]; VID=CP[5]; JAS=CP[6];
GOS=CP[7]; WAT=CP[8]; MON=CP[9]; FOS_cp=CP[10];

#%% LOAD DATA 

#converters

#reservoirs:
#read in historical reservoir inflows -- this will contain the array of 'dates' to use
top=int(horizon["res_data"]["@skiprows_number"])
bottom=int(horizon["res_data"]["@skip_footer_number"])

BLU5Ad = pd.read_excel('Data/BLU5A_daily.xls',skiprows=top, skip_footer=bottom) #only using data from 2005
BLU5Ad.columns = ['Date','Inflow']
BLU5A = pd.read_excel('Data/BLU5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
CGR5A = pd.read_excel('Data/CGR5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
DET5A = pd.read_excel('Data/DET5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
DOR5A = pd.read_excel('Data/DOR5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
FAL5A = pd.read_excel('Data/FAL5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
FOS5A = pd.read_excel('Data/FOS5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
FRN5M = pd.read_excel('Data/FRN5M_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
GPR5A = pd.read_excel('Data/GPR5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
HCR5A = pd.read_excel('Data/HCR5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
LOP5A = pd.read_excel('Data/LOP5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
LOP5E = pd.read_excel('Data/LOP5E_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
COT5A = pd.read_excel('Data/COT5A_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms
FOS_loc = pd.read_excel('Data/FOS_loc.xls',usecols = [0,3],skiprows=top, skip_footer=bottom)*cfs_to_cms
LOP_loc = pd.read_excel('Data/LOP_loc.xls',usecols = [0,3],skiprows=top, skip_footer=bottom)*cfs_to_cms


#historical outflows 
BLU5H = np.array(pd.read_excel('Data/BLU5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms) #only using data from 2005
BCL5H = np.array(pd.read_excel('Data/BCL5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms) #only using data from 2005
CGR5H = np.array(pd.read_excel('Data/CGR5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
DET5H = np.array(pd.read_excel('Data/DET5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
DEX5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
DOR5H = np.array(pd.read_excel('Data/DOR5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
FAL5H = np.array(pd.read_excel('Data/FAL5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
FRN5H = np.array(pd.read_excel('Data/FRN5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
GPR5H = np.array(pd.read_excel('Data/GPR5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
HCR5H = np.array(pd.read_excel('Data/HCR5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
COT5H = np.array(pd.read_excel('Data/COT5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=top, skip_footer=bottom)*cfs_to_cms)

outflows_all_hist = np.stack((HCR5H[:,1],LOP5H[:,1],DEX5H[:,1],FAL5H[:,1],DOR5H[:,1],COT5H[:,1],FRN5H[:,1],CGR5H[:,1],BLU5H[:,1],GPR5H[:,1],FOS5H[:,1],DET5H[:,1],BCL5H[:,1]),axis=1)
outflows_wo_FRN_hist = np.stack((HCR5H[:,1],LOP5H[:,1],DEX5H[:,1],FAL5H[:,1],DOR5H[:,1],COT5H[:,1],CGR5H[:,1],BLU5H[:,1],GPR5H[:,1],FOS5H[:,1],DET5H[:,1],BCL5H[:,1]),axis=1)


#reading in historical volumes
M3_PER_ACREFT = 1233.4

#HCR
HCR_vol = pd.read_excel(str(volumes['@HCR_vol']))
HCR_vol.columns = ['Date','Time','Storage(AF)']
RES[0].histVol = np.array(HCR_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#LOP
LOP_vol = pd.read_excel(str(volumes['@LOP_vol']))
LOP_vol.columns = ['Date','Time','Storage(AF)']
RES[1].histVol = np.array(LOP_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#DEX
DEX_vol = pd.read_excel(str(volumes['@DEX_vol']))
DEX_vol.columns = ['Date','Time','Storage(AF)']
RES[2].histVol = np.array(LOP_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#FAL
FAL_vol = pd.read_excel(str(volumes['@FAL_vol']))
FAL_vol.columns = ['Date','Time','Storage(AF)']
RES[3].histVol = np.array(FAL_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#DOR
DOR_vol = pd.read_excel(str(volumes['@DOR_vol']))
DOR_vol.columns = ['Date','Time','Storage(AF)']
RES[4].histVol = np.array(DOR_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#COT
COT_vol = pd.read_excel(str(volumes['@COT_vol']))
COT_vol.columns = ['Date','Time','Storage(AF)']
RES[5].histVol = np.array(COT_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#FRN
FRN_vol = pd.read_excel(str(volumes['@FRN_vol']))
FRN_vol.columns = ['Date','Time','Storage(AF)']
RES[6].histVol = np.array(FRN_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#CGR
CGR_vol = pd.read_excel(str(volumes['@CGR_vol']))
CGR_vol.columns = ['Date','Time','Storage(AF)']
RES[7].histVol = np.array(CGR_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#BLU
BLU_vol = pd.read_excel(str(volumes['@BLU_vol']))
BLU_vol.columns = ['Date','Time','Storage(AF)']
RES[8].histVol = np.array(BLU_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#GPR
GPR_vol = pd.read_excel(str(volumes['@GPR_vol']))
GPR_vol.columns = ['Date','Time','Storage(AF)']
RES[9].histVol = np.array(GPR_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#FOS
FOS_vol = pd.read_excel(str(volumes['@FOS_vol']))
FOS_vol.columns = ['Date','Time','Storage(AF)']
RES[10].histVol = np.array(FOS_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#DET
DET_vol = pd.read_excel(str(volumes['@DET_vol']))
DET_vol.columns = ['Date','Time','Storage(AF)']
RES[11].histVol = np.array(DET_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)
#BCL
BCL_vol = pd.read_excel(str(volumes['@BCL_vol']))
BCL_vol.columns = ['Date','Time','Storage(AF)']
RES[12].histVol = np.array(BCL_vol.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)




#control points
#cp_hist discharge: start this at 12/31/2004
filename=str(horizon["cp_data"]["@filename"])

SAL = pd.read_excel(filename,sheetname='Salem')
SAL_dis = np.array(SAL['Discharge'])*cfs_to_cms
ALB = pd.read_excel(filename,sheetname='Albany')
ALB_dis = np.array(ALB['Discharge'])*cfs_to_cms
JEF = pd.read_excel(filename,sheetname='Jefferson')
JEF_dis = np.array(JEF['Discharge'])*cfs_to_cms
MEH = pd.read_excel(filename,sheetname='Mehama')
MEH_dis = np.array(MEH['Discharge'])*cfs_to_cms
HAR = pd.read_excel(filename,sheetname='Harrisburg')
HAR_dis = np.array(HAR['Discharge'])*cfs_to_cms
VID = pd.read_excel(filename,sheetname='Vida')
VID_dis = np.array(VID['Discharge'])*cfs_to_cms
JAS = pd.read_excel(filename,sheetname='Jasper')
JAS_dis = np.array(JAS['Discharge'])*cfs_to_cms
GOS = pd.read_excel(filename,sheetname='Goshen')
GOS_dis = np.array(GOS['Discharge'])*cfs_to_cms
WAT = pd.read_excel(filename,sheetname='Waterloo')
WAT_dis = np.array(WAT['Discharge'])*cfs_to_cms
MON = pd.read_excel(filename,sheetname='Monroe')
MON_dis = np.array(MON['Discharge'])*cfs_to_cms
FOS = pd.read_excel(filename,sheetname='Foster')
FOS_dis = np.array(FOS['Discharge'])*cfs_to_cms
cp_discharge_all_hist = np.stack((SAL_dis,ALB_dis,JEF_dis,MEH_dis,HAR_dis,VID_dis,JAS_dis,GOS_dis,WAT_dis,MON_dis,FOS_dis),axis=1)    


#cp local flows, starts at 1/1
    res.dataOUT = np.array(pd.read_excel(res.filename_dataOUT, skiprows=top, skip_footer=bottom)*cfs_to_cms)
    temp=pd.read_excel(res.initVol_filename)
    temp.columns = ['Date','Time','Storage(AF)']
    res.initVol = np.array(temp.groupby('Date')['Storage(AF)'].mean()*M3_PER_ACREFT)

#load cp data: 
hist_dis_filename=str(horizon["cp_data"]["@hist_dis_filename"])
locflow_filename=str(horizon["cp_data"]["@locflow_filename"])
top=int(horizon["cp_data"]["@skiprows_number"])
bottom=int(horizon["cp_data"]["@skip_footer_number"])

for cp in CP:
    temp= pd.read_excel(hist_dis_filename,sheetname=cp.name)
    cp.dis= np.array(temp['Discharge'])*cfs_to_cms
    if cp.name!="Foster":
        cp.loc=pd.read_excel(locflow_filename,sheetname=cp.name,skiprows=top, skip_footer=bottom)
        cp.loc.columns = ['Date','Local Flow']

#%% Allocate and initialize
T =int(horizon["length"]["@T"]) #Set the simulation horizon

n_res=len(res_list)
n_HPres=8
n_cp = len(cp_list)

#set dates
dates = np.array(CP[0].loc['Date'])
#=======
# allocate output 
outflows_all = np.full((T+2,n_res),np.nan) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.full((T+2,n_HPres), np.nan)
volumes_all = np.full((T+2,n_res),np.nan)
elevations_all = np.full((T+2,n_res),np.nan)
cp_discharge_all = np.full((T+2,(n_cp)),np.nan)

#initialize values
for  i in range(0,n_res):
    outflows_all[0:3,i] = RES[i].dataOUT[0:3,1] 
    volumes_all[0:3,i] = RES[i].initVol[0:3] 
    elevations_all[0:3,i]=inner.GetPoolElevationFromVolume(volumes_all[0:3,i],RES[i])


for  i in range(0,n_cp):
     cp_discharge_all[0,i] = CP[i].dis[0]  

InitwaterYear = 1.2
waterYear = InitwaterYear

#%% SIMULATION  

#Daily loop
for t in range(1,T-1):    
    doy = inner.DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')
    
    if doy==120:
        waterYear = inner.UpdateReservoirWaterYear(doy,t,volumes_all) #function missing
    
    #COTTAGE GROVE ID=6 count=5 NO HYDROPOWER
    COT_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,COT.ID-1],COT)
    [COT_outflow, _,_,_, COT.zone] = inner.GetResOutflow(COT,volumes_all[t-1,COT.ID-1],COT.dataIN.iloc[t,1],COT.dataIN.iloc[t-1,1],outflows_all[t-1,COT.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],COT.zone)
    [COT_volume,COT_elevation] = inner.UpdateVolume_elev (COT, COT.dataIN.iloc[t,1], COT_outflow, volumes_all[t-1,COT.ID-1])
    
    outflows_all[t,COT.ID-1] = COT_outflow 
    volumes_all[t,COT.ID-1] =  COT_volume
    elevations_all[t,COT.ID-1]=  COT_elevation


    #DORENA ID=5 count=4 NO HYDROPOWER
    DOR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,DOR.ID-1],DOR)
    [DOR_outflow, _,_,_, DOR.zone] = inner.GetResOutflow(DOR,volumes_all[t-1,DOR.ID-1],DOR.dataIN.iloc[t,1],DOR.dataIN.iloc[t-1,1],outflows_all[t-1,DOR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],DOR.zone)
    [DOR_volume,DOR_elevation] = inner.UpdateVolume_elev (DOR, DOR.dataIN.iloc[t,1], DOR_outflow,volumes_all[t-1,DOR.ID-1])
    
    outflows_all[t,DOR.ID-1] = DOR_outflow 
    volumes_all[t,DOR.ID-1] =  DOR_volume
    elevations_all[t,DOR.ID-1]=  DOR_elevation
    
    #FERN RIDGE ID=7 count=6 NO HYDROPOWER
    FRN_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,FRN.ID-1],FRN)
    [FRN_outflow,_,_,_, FRN.zone] = inner.GetResOutflow(FRN,volumes_all[t-1,FRN.ID-1],FRN.dataIN.iloc[t,1],FRN.dataIN.iloc[t-1,1],outflows_all[t-1,FRN.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],FRN.zone)
    [FRN_volume,FRN_elevation] = inner.UpdateVolume_elev (FRN, FRN.dataIN.iloc[t,1], FRN_outflow,volumes_all[t-1,FRN.ID-1])
    
    outflows_all[t,FRN.ID-1] = FRN_outflow 
    volumes_all[t,FRN.ID-1] =  FRN_volume
    elevations_all[t,FRN.ID-1]=  FRN_elevation
    
    #HILLS CREEK ID=1 count =0
    HCR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,HCR.ID-1],HCR)
    [HCR_outflow,powerFlow,RO_flow,spillwayFlow, HCR.zone] = inner.GetResOutflow(HCR,volumes_all[t-1,HCR.ID-1],HCR.dataIN.iloc[t,1],HCR.dataIN.iloc[t-1,1],outflows_all[t-1,HCR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],HCR.zone)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    [HCR_volume,HCR_elevation] = inner.UpdateVolume_elev (HCR, HCR.dataIN.iloc[t,1], HCR_outflow,volumes_all[t-1,HCR.ID-1])
    
    outflows_all[t,HCR.ID-1] = HCR_outflow 
    volumes_all[t,HCR.ID-1] =  HCR_volume
    elevations_all[t,HCR.ID-1]=  HCR_elevation
    hydropower_all[t,0] = HCR_power_output
    
    #LOOKOUT POINT ID=2 count=1
    LOP_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,LOP.ID-1],LOP)
    LOP_inflow =  HCR_outflow + LOP.dataIN.iloc[t,1] + LOP.dataEV.iloc[t,1] #balance equation
    LOP_inflow_lag =  outflows_all[t-1,HCR.ID-1] + LOP.dataIN.iloc[t-1,1] + LOP.dataEV.iloc[t-1,1] #balance equation
    [LOP_outflow,powerFlow,RO_flow,spillwayFlow, LOP.zone]  = inner.GetResOutflow(LOP,volumes_all[t-1,LOP.ID-1],LOP_inflow,LOP_inflow_lag,outflows_all[t-1,LOP.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],LOP.zone)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    [LOP_volume,LOP_elevation] = inner.UpdateVolume_elev (LOP, LOP_inflow, LOP_outflow,volumes_all[t-1,LOP.ID-1])
    
    outflows_all[t,LOP.ID-1] = LOP_outflow 
    volumes_all[t,LOP.ID-1] =  LOP_volume
    elevations_all[t,LOP.ID-1]=  LOP_elevation
    hydropower_all[t,1] = LOP_power_output
    
    #DEXTER ID=3 count=2
    DEX_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,DEX.ID-1],DEX)
    [DEX_outflow,powerFlow,RO_flow,spillwayFlow, DEX.zone] = inner.GetResOutflow(DEX,volumes_all[t-1,DEX.ID-1],LOP_outflow,outflows_all[t-1,LOP.ID-1],outflows_all[t-1,DEX.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],DEX.zone)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    [DEX_volume,DEX_elevation] = inner.UpdateVolume_elev (DEX, LOP_outflow, DEX_outflow,volumes_all[t-1,DEX.ID-1])
    
    outflows_all[t,DEX.ID-1] = DEX_outflow 
    volumes_all[t,DEX.ID-1] =  DEX_volume
    elevations_all[t,DEX.ID-1]=  DEX_elevation
    hydropower_all[t,2] = DEX_power_output   
    
    #FALL CREEK ID=4 count=3 NO HYDROPOWER
    FAL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,FAL.ID-1],FAL)
    [FAL_outflow, _,_,_, FAL.zone] = inner.GetResOutflow(FAL,volumes_all[t-1,FAL.ID-1],FAL.dataIN.iloc[t,1],FAL.dataIN.iloc[t-1,1],outflows_all[t-1,FAL.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],FAL.zone)
    [FAL_volume,FAL_elevation] = inner.UpdateVolume_elev (FAL, FAL.dataIN.iloc[t,1], FAL_outflow,volumes_all[t-1,FAL.ID-1])
    
    outflows_all[t,FAL.ID-1] = FAL_outflow 
    volumes_all[t,FAL.ID-1] =  FAL_volume
    elevations_all[t,FAL.ID-1]=  FAL_elevation
    
    #COUGAR ID=8 count=7
    CGR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,CGR.ID-1],CGR)
    [CGR_outflow,powerFlow,RO_flow,spillwayFlow, CGR.zone] = inner.GetResOutflow(CGR,volumes_all[t-1,CGR.ID-1],CGR.dataIN.iloc[t,1],CGR.dataIN.iloc[t-1,1],outflows_all[t-1,CGR.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],CGR.zone)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    [CGR_volume,CGR_elevation] = inner.UpdateVolume_elev (CGR, CGR.dataIN.iloc[t,1], CGR_outflow,volumes_all[t-1,CGR.ID-1])
    
    outflows_all[t,CGR.ID-1] = CGR_outflow 
    volumes_all[t,CGR.ID-1] =  CGR_volume
    elevations_all[t,CGR.ID-1]=  CGR_elevation
    hydropower_all[t,3] = CGR_power_output  

    
    #BLUE RIVER ID=9 count= 8 NO HYDROPOWER
    BLU_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t-1,BLU.ID-1],BLU)
    [BLU_outflow,_,_,_, BLU.zone]  = inner.GetResOutflow(BLU,volumes_all[t-1,BLU.ID-1],BLU.dataIN.iloc[t,1],BLU.dataIN.iloc[t-1,1],outflows_all[t-1,BLU.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],BLU.zone)
    [BLU_volume,BLU_elevation] = inner.UpdateVolume_elev (BLU, BLU.dataIN.iloc[t,1], BLU_outflow,volumes_all[t-1,BLU.ID-1])
    
    outflows_all[t,BLU.ID-1] = BLU_outflow 
    volumes_all[t,BLU.ID-1] =  BLU_volume
    elevations_all[t,BLU.ID-1]=  BLU_elevation    
       
    #the next reservoirs are at time "t+2" 
    #GREEN PETER ID=10 count=9
    GPR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,GPR.ID-1],GPR)
    [GPR_outflow, powerFlow,RO_flow,spillwayFlow,GPR.zone] = inner.GetResOutflow(GPR,volumes_all[t+1,GPR.ID-1],GPR.dataIN.iloc[t+2,1],GPR.dataIN.iloc[t+1,1],outflows_all[t+1,GPR.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],GPR.zone)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    [GPR_volume,GPR_elevation] = inner.UpdateVolume_elev (GPR, GPR.dataIN.iloc[t+2,1], GPR_outflow,volumes_all[t+1,GPR.ID-1])
    
    outflows_all[t+2,GPR.ID-1] = GPR_outflow 
    volumes_all[t+2,GPR.ID-1] =  GPR_volume
    elevations_all[t+2,GPR.ID-1]=  GPR_elevation
    hydropower_all[t+2,4] = GPR_power_output      
    
    
    #FOSTER ID=11 count=10
    FOS_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,FOS.ID-1],FOS)
    FOS_inflow =GPR_outflow + FOS.dataIN.iloc[t+2,1] #balance equation
    FOS_inflow_lag =outflows_all[t+1,GPR.ID-1] + FOS.dataIN.iloc[t+1,1] #balance equation
    [FOS_outflow, powerFlow,RO_flow,spillwayFlow, FOS.zone] = inner.GetResOutflow(FOS,volumes_all[t+1,FOS.ID-1],FOS_inflow,FOS_inflow_lag,outflows_all[t+1,FOS.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],FOS.zone)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    [FOS_volume,FOS_elevation] = inner.UpdateVolume_elev (FOS, FOS_inflow, FOS_outflow,volumes_all[t+1,FOS.ID-1])
    
    outflows_all[t+2,FOS.ID-1] = FOS_outflow 
    volumes_all[t+2,FOS.ID-1] =  FOS_volume
    elevations_all[t+2,FOS.ID-1]=  FOS_elevation
    hydropower_all[t+2,5] = FOS_power_output   
    
    
    #DETROIT ID=12 count=11
    DET_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,DET.ID-1],DET)
    [DET_outflow, powerFlow,RO_flow,spillwayFlow, DET.zone] = inner.GetResOutflow(DET,volumes_all[t+1,DET.ID-1],DET.dataIN.iloc[t+2,1],DET.dataIN.iloc[t+1,1],outflows_all[t+1,DET.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],DET.zone)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    [DET_volume,DET_elevation] = inner.UpdateVolume_elev (DET, DET.dataIN.iloc[t+2,1], DET_outflow,volumes_all[t+1,DET.ID-1])
    
    outflows_all[t+2,DET.ID-1] = DET_outflow 
    volumes_all[t+2,DET.ID-1] =  DET_volume
    elevations_all[t+2,DET.ID-1]=  DET_elevation
    hydropower_all[t+2,6] = DET_power_output   
        
    #BIG CLIFF ID=13 count=12
    BCL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+1,BCL.ID-1],BCL)
    [BCL_outflow, powerFlow,RO_flow,spillwayFlow, BCL.zone] = inner.GetResOutflow(BCL,volumes_all[t+1,BCL.ID-1],DET_outflow,outflows_all[t+1,DET.ID-1] ,outflows_all[t+1,BCL.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],BCL.zone)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    [BCL_volume,BCL_elevation] = inner.UpdateVolume_elev (BCL, DET_outflow, BCL_outflow,volumes_all[t+1,BCL.ID-1])
    
    outflows_all[t+2,BCL.ID-1] = BCL_outflow 
    volumes_all[t+2,BCL.ID-1] =  BCL_volume
    elevations_all[t+2,BCL.ID-1]=  BCL_elevation
    hydropower_all[t+2,7] = BCL_power_output

    #Update cp discharge
    #in order of upstream to down
    #GOSHEN ID=7
    cp_discharge_all[t,7] = COT_outflow + DOR_outflow + GOS.loc.iloc[t,1]    
    #JASPER ID=6
    cp_discharge_all[t,6] = DEX_outflow + FAL_outflow + JAS.loc.iloc[t,1]    
    #VIDA ID=5
    cp_discharge_all[t,5] = CGR_outflow + BLU_outflow + VID.loc.iloc[t,1]    
    #HARRISBURG ID=4
    cp_discharge_all[t,4] = cp_discharge_all[t,7] + cp_discharge_all[t,6] + cp_discharge_all[t,5] + HAR.loc.iloc[t,1]    
    #MONROE ID=9
    cp_discharge_all[t,9] = FRN_outflow + MON.loc.iloc[t,1]    
    #ALBANY ID=1
    cp_discharge_all[t,1] = cp_discharge_all[t-1,9] + cp_discharge_all[t-1,4]+ ALB.loc.iloc[t,1]    
    #WATERLOO ID=8
    cp_discharge_all[t,8] = FOS_outflow + WAT.loc.iloc[t,1]    
    #MEHAMA ID=3
    cp_discharge_all[t,3] = BCL_outflow + MEH.loc.iloc[t,1]    
    #JEFFERSON ID=2
    cp_discharge_all[t,2] = cp_discharge_all[t,8] + cp_discharge_all[t,3] + JEF.loc.iloc[t,1]    
    #SALEM ID=0
    cp_discharge_all[t,0] = cp_discharge_all[t,1] + cp_discharge_all[t,2] + SAL.loc.iloc[t,1]   
    #FOSTER-cp ID=10 (reservoir ID=11)
    cp_discharge_all[t,10] = FOS_outflow

#%%















  
    
    
    
    
    
