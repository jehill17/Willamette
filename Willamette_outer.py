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
#with open('Flow_2001.xml') as fd:
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
  
cp_list =['SAL', 'ALB', 'JEF', 'MEH', 'HAR', 'VID', 'JAS', 'GOS', 'WAT', 'MON', 'FOS_out', 'FOS_in']
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
GOS=CP[7]; WAT=CP[8]; MON=CP[9]; FOS_out=CP[10]; FOS_in=CP[11]

#%% LOAD DATA 

#converters
cfs_to_cms = 0.0283168  
M3_PER_ACREFT = 1233.4  

#load reservoirs inflow and volume data:   
top=int(horizon["res_data"]["@skiprows_number"])
bottom=int(horizon["res_data"]["@skip_footer_number"])

for res in RES:
    res.dataIN = pd.read_excel(res.filename_dataIN, skiprows=top, skip_footer=bottom, usecols = [0, 1])
    res.dataIN.columns=['Date','Inflow']
    res.dataIN['Inflow']=res.dataIN['Inflow']*cfs_to_cms
    if res.name=="LOP":
        res.dataEV = pd.read_excel(res.filename_dataEV, skiprows=top, skip_footer=bottom)*cfs_to_cms
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
    temp= pd.read_excel(hist_dis_filename,sheetname=cp.name, skiprows=top, skip_footer=bottom)
    temp.columns=['Date','Discharge']
    cp.dis= np.array(temp['Discharge'])*cfs_to_cms
    if cp.name not in ("Foster_in","Foster_out") :
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
     cp_discharge_all[0:3,i] = CP[i].dis[0:3]  

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
    DEX_inflow =  LOP_outflow + DEX.dataIN.iloc[t,1] #balance equation
    DEX_inflow_lag =  outflows_all[t-1,LOP.ID-1] + DEX.dataIN.iloc[t-1,1] #balance equation
    [DEX_outflow,powerFlow,RO_flow,spillwayFlow, DEX.zone] = inner.GetResOutflow(DEX,volumes_all[t-1,DEX.ID-1],DEX_inflow,DEX_inflow_lag,outflows_all[t-1,DEX.ID-1],doy,waterYear,CP,cp_discharge_all[t-1,:],DEX.zone)
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
    
    #Update cp discharge
    #in order of upstream to down    
    #GOSHEN ID=7
    cp_discharge_all[t,7] = COT_outflow + DOR_outflow + GOS.loc.iloc[t,1]    
    #MONROE ID=9
    cp_discharge_all[t,9] = FRN_outflow + MON.loc.iloc[t,1] 
    #JASPER ID=6
    cp_discharge_all[t,6] = DEX_outflow + FAL_outflow + JAS.loc.iloc[t,1]    
    #VIDA ID=5
    cp_discharge_all[t,5] = CGR_outflow + BLU_outflow + VID.loc.iloc[t,1] 
    #HARRISBURG ID=4
    cp_discharge_all[t,4] = cp_discharge_all[t,7] + cp_discharge_all[t,6] + cp_discharge_all[t,5] + HAR.loc.iloc[t,1]   
    #____t+1
    #ALBANY ID=1
    cp_discharge_all[t+1,1] = cp_discharge_all[t,9] + cp_discharge_all[t,4]+ ALB.loc.iloc[t+1,1]    
    #WATERLOO ID=8
    cp_discharge_all[t+1,8] = outflows_all[t+1,FOS.ID-1] + WAT.loc.iloc[t+1,1] 
    #MEHAMA ID=3
    cp_discharge_all[t+1,3] = outflows_all[t+1,BCL.ID-1] + MEH.loc.iloc[t+1,1]    
    #JEFFERSON ID=2
    cp_discharge_all[t+1,2] = cp_discharge_all[t+1,8] + cp_discharge_all[t+1,3] + JEF.loc.iloc[t+1,1]    
    #SALEM ID=0
    cp_discharge_all[t+1,0] = cp_discharge_all[t+1,1] + cp_discharge_all[t+1,2] + SAL.loc.iloc[t+1,1]   
    #FOSTER-out ID=10 (reservoir ID=11)
    cp_discharge_all[t+1,10] = outflows_all[t+1,FOS.ID-1]
    #FOSTER-in ID=11 (reservoir ID=11)
    cp_discharge_all[t+1,11] = outflows_all[t+1,GPR.ID-1] + FOS.dataIN.iloc[t+1,1]
    
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
    BCL_inflow =DET_outflow + BCL.dataIN.iloc[t+2,1] #balance equation
    BCL_inflow_lag =outflows_all[t+1,DET.ID-1] + BCL.dataIN.iloc[t+1,1] #balance equation   
    [BCL_outflow, powerFlow,RO_flow,spillwayFlow, BCL.zone] = inner.GetResOutflow(BCL,volumes_all[t+1,BCL.ID-1],BCL_inflow,BCL_inflow_lag ,outflows_all[t+1,BCL.ID-1],doy,waterYear,CP,cp_discharge_all[t+1,:],BCL.zone)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    [BCL_volume,BCL_elevation] = inner.UpdateVolume_elev (BCL, DET_outflow, BCL_outflow,volumes_all[t+1,BCL.ID-1])
    
    outflows_all[t+2,BCL.ID-1] = BCL_outflow 
    volumes_all[t+2,BCL.ID-1] =  BCL_volume
    elevations_all[t+2,BCL.ID-1]=  BCL_elevation
    hydropower_all[t+2,7] = BCL_power_output

 
   
   


#%%















  
    
    
    
    
    
