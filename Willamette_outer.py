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

# load reservoirs and control point infos
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
    res.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
    res.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
    res.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
    if res.Restype == "Storage":
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
    cp.influencedReservoirs =np.asarray((controlPoints[id-1]["@reservoirs"]).split(','))
    cp.COMID=str(controlPoints[id-1]["@location"])


#import control point historical data-- shifted one day before

#convert data
cfs_to_cms = 0.0283168
 
cp_local = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname=[0,1,2,3,4,5,6,7,8,9]) 

#reservoirs:
#read in historical reservoir inflows -- this will contain the array of 'dates' to use
BLU5Ad = pd.read_excel('Data/BLU5A_daily.xls',skiprows=27942,skip_footer =1004) #only using data from 2005
BLU5Ad.columns = ['Date','Inflow']
BLU5A = BLU5Ad['Inflow']*cfs_to_cms
CGR5A = pd.read_excel('Data/CGR5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
DET5A = pd.read_excel('Data/DET5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
DEX5M = pd.read_excel('Data/DEX5M_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
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


#historical inflows 
BLU5H = np.array(pd.read_excel('Data/BLU5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms) #only using data from 2005
BCL5H = np.array(pd.read_excel('Data/BCL5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms) #only using data from 2005
CGR5H = np.array(pd.read_excel('Data/CGR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DET5H = np.array(pd.read_excel('Data/DET5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DEX5H = np.array(pd.read_excel('Data/DEX5M_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
DOR5H = np.array(pd.read_excel('Data/DOR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FAL5H = np.array(pd.read_excel('Data/FAL5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FRN5H = np.array(pd.read_excel('Data/FRN5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
GPR5H = np.array(pd.read_excel('Data/GPR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
HCR5H = np.array(pd.read_excel('Data/HCR5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
COT5H = np.array(pd.read_excel('Data/COT5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
FOS5H = np.array(pd.read_excel('Data/FOS5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)
LOP5H = np.array(pd.read_excel('Data/LOP5H_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms)

outflows_all_2005 = np.stack((BLU5H[:,1],BCL5H[:,1],CGR5H[:,1],DET5H[:,1],DEX5H[:,1],DOR5H[:,1],FAL5H[:,1],FOS5H[:,1],FRN5H[:,1],GPR5H[:,1],HCR5H[:,1],LOP5H[:,1],COT5H[:,1],FOS5H[:,1]),axis=1)


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

#=======
T = 7 # Set the simulation horizon

n_res=13
n_HPres=8
n_cp = 11


#=======
# allocate output 
outflows_all = np.full((T+1,n_res),np.nan) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.full((T+1,n_HPres), np.nan)
volumes_all = np.full((T+1,n_res),np.nan)
elevations_all = np.full((T+1,n_res),np.nan)
cp_discharge_all = np.full((T+1,(n_cp-1)),np.nan)

#initialize values
for  i in range(0,n_res):
    #outflows_all[0,i] = outflows_2005_all[0,i] #remember to stack outflows historical values
    volumes_all[0,i] = RES[i].InitVol 
    elevations_all[0,i]=inner.GetPoolElevationFromVolume(volumes_all[0,i],RES[i])

for  i in range(0,n_cp-1):
     cp_discharge_all[0,i] = cp_discharge_2005_all[0,i]

#define an outer fnt here that takes date, name, vol as inputs?

InitwaterYear = 1.2
waterYear = InitwaterYear




for t in range(1,T+2):    
    doy = inner.DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')
    
    waterYear = inner.UpdateReservoirWaterYear(doy,t,volumes_all) #function missing
    
    #COTTAGE GROVE ID=6 count=5 NO HYDROPOWER
    COT = RES[5]
    COT_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,COT.ID],COT)
    COT_outflow = inner.GetResOutflow(COT,volumes_all[t,COT.ID],COT5A.iloc[t,1],outflows_all[t-1,COT.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [COT_volume,COT_elevation] = inner.UpdateVolume_elev (COT, COT5A.iloc[t,1], COT_outflow, volumes_all[t,COT.ID])
    
    outflows_all[t+1,COT.ID] = COT_outflow 
    volumes_all[t+1,COT.ID] =  COT_volume
    elevations_all[t+1,COT.ID]=  COT_elevation


    #DORENA ID=5 count=4 NO HYDROPOWER
    DOR = RES[4]
    DOR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,DOR.ID],DOR)
    DOR_outflow = inner.GetResOutflow(DOR,volumes_all[t,DOR.ID],DOR5A.iloc[t,1],outflows_all[t-1,DOR.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [DOR_volume,DOR_elevation] = inner.UpdateVolume_elev (DOR, DOR5A.iloc[t,1], DOR_outflow,volumes_all[t,DOR.ID])
    
    outflows_all[t+1,DOR.ID] = DOR_outflow 
    volumes_all[t+1,DOR.ID] =  DOR_volume
    elevations_all[t+1,DOR.ID]=  DOR_elevation
    
    #FERN RIDGE ID=7 count=6 NO HYDROPOWER
    FRN = RES[6]
    FRN_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,FRN.ID],FRN)
    FRN_outflow = inner.GetResOutflow(FRN,volumes_all[t,FRN.ID],FRN5M.iloc[t,1],outflows_all[t-1,FRN.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [FRN_volume,FRN_elevation] = inner.UpdateVolume_elev (FRN, FRN5M.iloc[t,1], FRN_outflow,volumes_all[t,FRN.ID])
    
    outflows_all[t+1,FRN.ID] = FRN_outflow 
    volumes_all[t+1,FRN.ID] =  FRN_volume
    elevations_all[t+1,FRN.ID]=  FRN_elevation
    
    #HILLS CREEK ID=1 count =0
    HCR = RES[0]
    HCR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,HCR.ID],HCR)
    HCR_outflow = inner.GetResOutflow(HCR,volumes_all[t,HCR.ID],HCR5A.iloc[t,1],outflows_all[t-1,HCR.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(HCR,HCR_outflow)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    [HCR_volume,HCR_elevation] = inner.UpdateVolume_elev (HCR, HCR5A.iloc[t,1], HCR_outflow,volumes_all[t,HCR.ID])
    
    outflows_all[t+1,HCR.ID] = HCR_outflow 
    volumes_all[t+1,HCR.ID] =  HCR_volume
    elevations_all[t+1,HCR.ID]=  HCR_elevation
    hydropower_all[t,1] = HCR_power_output
    
    #LOOKOUT POINT ID=2 count=1
    LOP = RES[1]
    LOP_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,LOP.ID],LOP)
    LOP_outflow = inner.GetResOutflow(LOP,volumes_all[t,LOP.ID],LOP5A.iloc[t,1],outflows_all[t-1,LOP.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(LOP,LOP_outflow)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    [LOP_volume,LOP_elevation] = inner.UpdateVolume_elev (LOP, LOP5A.iloc[t,1], LOP_outflow,volumes_all[t,LOP.ID])
    
    outflows_all[t+1,LOP.ID] = LOP_outflow 
    volumes_all[t+1,LOP.ID] =  LOP_volume
    elevations_all[t+1,LOP.ID]=  LOP_elevation
    hydropower_all[t,2] = LOP_power_output
    
    #DEXTER ID=3 count=2
    DEX = RES[2]
    DEX_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,DEX.ID],DEX)
    DEX_outflow = inner.GetResOutflow(DEX,volumes_all[t,DEX.ID],LOP_outflow,outflows_all[t-1,DEX.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(DEX,DEX_outflow)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    [DEX_volume,DEX_elevation] = inner.UpdateVolume_elev (DEX, LOP_outflow, DEX_outflow,volumes_all[t,DEX.ID])
    
    outflows_all[t+1,DEX.ID] = DEX_outflow 
    volumes_all[t+1,DEX.ID] =  DEX_volume
    elevations_all[t+1,DEX.ID]=  DEX_elevation
    hydropower_all[t,3] = DEX_power_output   
    
    #FALL CREEK ID=4 count=3 NO HYDROPOWER
    FAL = RES[3]
    FAL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,FAL.ID],FAL)
    FAL_outflow = inner.GetResOutflow(FAL,volumes_all[t,FAL.ID],FAL5A.iloc[t,1],outflows_all[t-1,FAL.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [FAL_volume,FAL_elevation] = inner.UpdateVolume_elev (FAL, FAL5A.iloc[t,1], FAL_outflow,volumes_all[t,FAL.ID])
    
    outflows_all[t+1,FAL.ID] = FAL_outflow 
    volumes_all[t+1,FAL.ID] =  FAL_volume
    elevations_all[t+1,FAL.ID]=  FAL_elevation
    
    #COUGAR ID=8 count=7
    CGR = RES[7]
    CGR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,CGR.ID],CGR)
    CGR_outflow = inner.GetResOutflow(CGR,volumes_all[t,CGR.ID],CGR5A.iloc[t,1],outflows_all[t-1,CGR.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(CGR,CGR_outflow)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    [CGR_volume,CGR_elevation] = inner.UpdateVolume_elev (CGR, CGR5A.iloc[t,1], CGR_outflow,volumes_all[t,CGR.ID])
    
    outflows_all[t+1,CGR.ID] = CGR_outflow 
    volumes_all[t+1,CGR.ID] =  CGR_volume
    elevations_all[t+1,CGR.ID]=  CGR_elevation
    hydropower_all[t,4] = CGR_power_output  

    
    #BLUE RIVER ID=9 count= 8 NO HYDROPOWER
    BLU = RES[8]
    BLU_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t,BLU.ID],BLU)
    BLU_outflow = inner.GetResOutflow(BLU,volumes_all[t,BLU.ID],BLU5A.iloc[t,1],outflows_all[t-1,BLU.ID],doy,waterYear,CP,cp_discharge_all[t,:])
    [BLU_volume,BLU_elevation] = inner.UpdateVolume_elev (BLU, BLU5A.iloc[t,1], BLU_outflow,volumes_all[t,BLU.ID])
    
    outflows_all[t+1,BLU.ID] = BLU_outflow 
    volumes_all[t+1,BLU.ID] =  BLU_volume
    elevations_all[t+1,BLU.ID]=  BLU_elevation    
    
    
    #the next reservoirs are at time "t+2" #check!!!
    #GREEN PETER ID=10 count=9
    GPR = RES[9]
    GPR_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+2,GPR.ID],GPR)
    GPR_outflow = inner.GetResOutflow(GPR,volumes_all[t+2,GPR.ID],GPR5A.iloc[t+2,1],outflows_all[t+1,GPR.ID],doy,waterYear,CP,cp_discharge_all[t+2,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(GPR,GPR_outflow)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    [GPR_volume,GPR_elevation] = inner.UpdateVolume_elev (GPR, GPR5A.iloc[t+2,1], GPR_outflow,volumes_all[t+2,GPR.ID])
    
    outflows_all[t+2,GPR.ID] = GPR_outflow 
    volumes_all[t+2,GPR.ID] =  GPR_volume
    elevations_all[t+2,GPR.ID]=  GPR_elevation
    hydropower_all[t+2,5] = GPR_power_output      
    
    
    #FOSTER ID=11 count=10
    FOS = RES[10]
    FOS_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+2,FOS.ID],FOS)
    FOS_outflow = inner.GetResOutflow(FOS,volumes_all[t+2,FOS.ID],FOS5A.iloc[t+2,1],outflows_all[t+1,FOS.ID],doy,waterYear,CP,cp_discharge_all[t+2,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(FOS,FOS_outflow)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    [FOS_volume,FOS_elevation] = inner.UpdateVolume_elev (FOS, FOS5A.iloc[t+2,1], FOS_outflow,volumes_all[t+2,FOS.ID])
    
    outflows_all[t+2,FOS.ID] = FOS_outflow 
    volumes_all[t+2,FOS.ID] =  FOS_volume
    elevations_all[t+2,FOS.ID]=  FOS_elevation
    hydropower_all[t+2,6] = FOS_power_output   
    
    
    #DETROIT ID=12 count=11
    DET = RES[11]
    DET_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+2,DET.ID],DET)
    DET_outflow = inner.GetResOutflow(DET,volumes_all[t+2,DET.ID],DET5A.iloc[t+2,1],outflows_all[t+1,DET.ID],doy,waterYear,CP,cp_discharge_all[t+2,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(DET,DET_outflow)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    [DET_volume,DET_elevation] = inner.UpdateVolume_elev (DET, DET5A.iloc[t+2,1], DET_outflow,volumes_all[t+2,DET.ID])
    
    outflows_all[t+2,DET.ID] = DET_outflow 
    volumes_all[t+2,DET.ID] =  DET_volume
    elevations_all[t+2,DET.ID]=  DET_elevation
    hydropower_all[t+2,7] = DET_power_output   
        
    #BIG CLIFF ID=13 count=12
    BCL = RES[12]
    BCL_poolElevation = inner.GetPoolElevationFromVolume(volumes_all[t+2,BCL.ID],BCL)
    BCL_outflow = inner.GetResOutflow(BCL,volumes_all[t+2,BCL.ID],DET_outflow,outflows_all[t+1,BCL.ID],doy,waterYear,CP,cp_discharge_all[t+2,:])
    [powerFlow,RO_flow,spillwayFlow, massbalancecheck] = inner.AssignReservoirOutletFlows(BCL,BCL_outflow)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    [BCL_volume,BCL_elevation] = inner.UpdateVolume_elev (BCL, DET_outflow, BCL_outflow,volumes_all[t+2,BCL.ID])
    
    outflows_all[t+2,BCL.ID] = BCL_outflow 
    volumes_all[t+2,BCL.ID] =  BCL_volume
    elevations_all[t+2,BCL.ID]=  BCL_elevation
    hydropower_all[t+2,8] = BCL_power_output      


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
    
    
    
    
    
    
    
