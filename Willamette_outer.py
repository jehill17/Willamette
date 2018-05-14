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
        self.InitVol=[]
        self.init_outflow =[]
        self.minOutflow=[]
        self.maxVolume=[]
        self.Td_elev=[]
        self.Inactive_elev=[]
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
        self.init_discharge = []



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
    res.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
    res.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
    res.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
    res.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
    if res.Restype == "Storage":
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
<<<<<<< HEAD
    cp.influencedReservoirs = pd.read_csv(os.path.join('ControlPoints/',str(controlPoints[id-1]["@reservoirs"])))
=======
>>>>>>> 37cc07be000107ab68229cf439f102162c71c043
    cp.COMID=str(controlPoints[id-1]["@location"])
    #cp.init_discharge=float(controlPoints[id-1]["@initDischarge"]) TO BE ADDED in the xml file


#import control point historical data-- shifted one day before

#convert data
cfs_to_cms = 0.0283168

#cp_hist: start this at 12/31/2004
<<<<<<< HEAD
SAL_2005 = pd.read_excel(os.path.join('Data/'Control point historical discharge 2005.xlsx',sheetname='Salem')
SAL_2005_dis = np.array(SAL_2005['Discharge'])*cfs_to_cms
ALB_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Albany')
ALB_2005_dis = np.array(ALB_2005['Discharge'])*cfs_to_cms
JEF_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Jefferson')
JEF_2005_dis = np.array(JEF_2005['Discharge'])*cfs_to_cms
MEH_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Mehama')
MEH_2005_dis = np.array(MEH_2005['Discharge'])*cfs_to_cms
HAR_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Harrisburg')
HAR_2005_dis = np.array(HAR_2005['Discharge'])*cfs_to_cms

VID_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Vida')
VID_2005_dis = np.array(VID_2005['Discharge'])*cfs_to_cms

JAS_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Jasper')
JAS_2005_dis = np.array(JAS_2005['Discharge'])*cfs_to_cms

GOS_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Goshen')
GOS_2005_dis = np.array(GOS_2005['Discharge'])*cfs_to_cms

WAT_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Waterloo')
WAT_2005_dis = np.array(WAT_2005['Discharge'])*cfs_to_cms

MON_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Monroe')
MON_2005_dis = np.array(MON_2005['Discharge'])*cfs_to_cms

FOS_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Foster')
FOS_2005_dis = np.array(FOS_2005['Discharge'])*cfs_to_cms


 

cp_local = pd.read_excel('Controlpoints_local_flows.xls',sheetname=[0,1,2,3,4,5,6,7,8,9]) #when does this data come into play?
#add in fnt that updates control pt discharge after every timestep

=======
>>>>>>> 37cc07be000107ab68229cf439f102162c71c043

#reservoirs:
#read in historical reservoir inflows -- this will contain the array of 'dates' to use
BLU5A = pd.read_excel('Data/BLU5A_daily.xls',skiprows=27943,skip_footer =1004) #only using data from 2005
BLU5A.columns = ['Date','Inflow']
CGR5A = pd.read_excel('Data/CGR5A_daily.xls',skiprows=27943,skip_footer =1004)
DET5A = pd.read_excel('Data/DET5A_daily.xls',skiprows=27943,skip_footer =1004)
DEX5M = pd.read_excel('Data/DEX5M_daily.xls',skiprows=27943,skip_footer =1004)
DOR5A = pd.read_excel('Data/DOR5A_daily.xls',skiprows=27943,skip_footer =1004)
FAL5A = pd.read_excel('Data/FAL5A_daily.xls',skiprows=27943,skip_footer =1004)
FOS5A = pd.read_excel('Data/FOS5A_daily.xls',skiprows=27943,skip_footer =1004)
FRN5M = pd.read_excel('Data/FRN5M_daily.xls',skiprows=27943,skip_footer =1004)
GPR5A = pd.read_excel('Data/GPR5A_daily.xls',skiprows=27943,skip_footer =1004)
HCR5A = pd.read_excel('Data/HCR5A_daily.xls',skiprows=27943,skip_footer =1004)
LOP5A = pd.read_excel('Data/LOP5A_daily.xls',skiprows=27943,skip_footer =1004)
LOP5E = pd.read_excel('Data/LOP5E_daily.xls',skiprows=27943,skip_footer =1004)
COT5A = pd.read_excel('Data/COT5A_daily.xls',skiprows=27943,skip_footer =1004)
FOS_loc = pd.read_excel('Data/FOS_loc.xls',usecols = [0,3],skiprows=27943,skip_footer =1004)
LOP_loc = pd.read_excel('Data/LOP_loc.xls',usecols = [0,3],skiprows=27943,skip_footer =1004)



#control points
#filename='Data/Control point historical discharge 2005.xlsx'
#SAL_2005 = pd.read_excel(filename,sheetname='Salem')
#SAL_2005_dis = np.array(SAL_2005['Discharge'])*cfs_to_cms
#ALB_2005 = pd.read_excel(filename,sheetname='Albany')
#ALB_2005_dis = np.array(ALB_2005['Discharge'])*cfs_to_cms
#JEF_2005 = pd.read_excel(filename,sheetname='Jefferson')
#JEF_2005_dis = np.array(JEF_2005['Discharge'])*cfs_to_cms
#MEH_2005 = pd.read_excel(filename,sheetname='Mehama')
#MEH_2005_dis = np.array(MEH_2005['Discharge'])*cfs_to_cms
#HAR_2005 = pd.read_excel(filename,sheetname='Harrisburg')
#HAR_2005_dis = np.array(HAR_2005['Discharge'])*cfs_to_cms
#VID_2005 = pd.read_excel(filename,sheetname='Vida')
#VID_2005_dis = np.array(VID_2005['Discharge'])*cfs_to_cms
#JAS_2005 = pd.read_excel(filename,sheetname='Jasper')
#JAS_2005_dis = np.array(JAS_2005['Discharge'])*cfs_to_cms
#GOS_2005 = pd.read_excel(filename,sheetname='Goshen')
#GOS_2005_dis = np.array(GOS_2005['Discharge'])*cfs_to_cms
#WAT_2005 = pd.read_excel(filename,sheetname='Waterloo')
#WAT_2005_dis = np.array(WAT_2005['Discharge'])*cfs_to_cms
#MON_2005 = pd.read_excel(filename,sheetname='Monroe')
#MON_2005_dis = np.array(MON_2005['Discharge'])*cfs_to_cms
#FOS_2005 = pd.read_excel(filename,sheetname='Foster')
#FOS_2005_dis = np.array(FOS_2005['Discharge'])*cfs_to_cms

cp_local = pd.read_excel('Data/Controlpoints_local_flows.xls',sheetname=[0,1,2,3,4,5,6,7,8,9]) #when does this data come into play?

dates = np.array(BLU5A['Date'])

#=======
T = 365 # Set the simulation horizon

n_res=13
n_HPres=8
n_cp = 11


#=======
# allocate output 
outflows_all = np.nan(T+1,n_res) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.nan((T+1,n_HPres))
volumes_all = np.nan((T+1,n_res))
elevations_all = np.nan((T+1,n_res))
cp_discharge_all = np.nan((T+1,n_cp))

#initialize values
for  i in range(0,n_res):
    #outflows_all[0,i] = RES[i].init_outflow
    volumes_all[0,i] = RES[i].InitVol 
    elevations_all[0,i]=inner.GetPoolElevationFromVolume(volumes_all[0,i])

for  i in range(0,n_cp):
    cp_discharge_all[0,i] = CP[i].init_discharge

#define an outer fnt here that takes date, name, vol as inputs?

InitwaterYear = 1.2

for t in range(0,T+1):
    
    doy = inner.DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')
    
    waterYear = inner.UpdateReservoirWaterYear(doy) #function missing
    #calculate waterYear
    #conditional based on doy 
    #calculate at doy = 140
    
    #COTTAGE GROVE ID =6 
    COT_poolElevation = inner.GetPoolElevationfromVolume(RES[6],volumes_all[t,RES[6].ID]) 
    COT_outflow = inner.GetResOutflow(RES[6],volumes_all[t,RES[6].ID],COT5A.iloc[t,1],outflows_all[t,RES[6].ID],doy,waterYear,cp_list,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(RES[6],COT_outflow)
    COT_power_output = inner.CalculateHydropowerOutput(RES[6],COT_poolElevation,powerFlow) #do I need to give a unique name to poweroutflow?
    
    outflows_all[t+1,RES[6].ID] = COT_outflow
#    hydropower_all[t,0] = COT_power_output
    
    
    #DORENA ID=5
    DOR_poolElevation = inner.GetPoolElevationFromVolume(DOR,DOR.volume)
    DOR_outflow = inner.GetResOutflow(DOR,DOR.InitVol,DOR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DOR,DOR_outflow)
    DOR_power_output = inner.CalculateHydropowerOutput(DOR,DOR_poolElevation,powerFlow)
    
    outflows_all[t,DOR.ID] = DOR_outflow
#    hydropower_all[t,1] = DOR_power_output
    
    #FERN RIDGE ID=7
    FRN_poolElevation = inner.GetPoolElevationFromVolume(FRN,FRN.volume)
    FRN_outflow = inner.GetResOutflow(FRN,FRN.InitVol,FRN5M.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FRN,FRN_outflow)
    FRN_power_output = inner.CalculateHydropowerOutput(FRN,FRN_poolElevation,powerFlow)
    
    outflows_all[t,FRN.ID] = FRN_outflow
#    hydropower_all[t,2] = FRN_power_output
    
    #HILLS CREEK ID=1
    HCR_poolElevation = inner.GetPoolElevationFromVolume(HCR,HCR.volume)
    HCR_outflow = inner.GetResOutflow(HCR,HCR.InitVol,HCR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(HCR,HCR_outflow)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    
    outflows_all[t,HCR.ID] = HCR_outflow
    hydropower_all[t,1] = HCR_power_output
    
    #LOOKOUT POINT ID=2
    LOP_poolElevation = inner.GetPoolElevationFromVolume(LOP,LOP.volume)
    LOP_in = HCR_outflow + LOP_loc[t] + LOP5E[t]
    LOP_outflow = inner.GetResOutflow(LOP,LOP.InitVol,LOP_in,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(LOP,LOP_outflow)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    
    outflows_all[t,LOP.ID] = LOP_outflow
    hydropower_all[t,2] = LOP_power_output
    
    #DEXTER ID=3
    DEX_poolElevation = inner.GetPoolElevationFromVolume(DEX,DEX.volume)
    DEX_outflow = inner.GetResOutflow(DEX,DEX.InitVol,LOP_outflow,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DEX,DEX_outflow)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    
    outflows_all[t,DEX.ID] = DEX_outflow
    hydropower_all[t,3] = DEX_power_output
    
    #FALL CREEK ID=4
    FAL_poolElevation = inner.GetPoolElevationFromVolume(FAL,FAL.volume)
    FAL_outflow = inner.GetResOutflow(FAL,FAL.InitVol,FAL5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FAL,FAL_outflow)
    FAL_power_output = inner.CalculateHydropowerOutput(FAL,FAL_poolElevation,powerFlow)
    
    #COUGAR ID=8
    CGR_poolElevation = inner.GetPoolElevationFromVolume(CGR,CGR.volume)
    CGR_outflow = inner.GetResOutflow(CGR,CGR.InitVol,CGR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(CGR,CGR_outflow)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    
    outflows_all[t,CGR.ID] = CGR_outflow
    hydropower_all[t,4] = CGR_power_output
    
    #BLUE RIVER ID=9
    BLU_poolElevation = inner.GetPoolElevationFromVolume(BLU,BLU.volume)
    BLU_outflow = inner.GetResOutflow(BLU,BLU.InitVol,BLU5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BLU,BLU_outflow)
    BLU_power_output = inner.CalculateHydropowerOutput(BLU,BLU_poolElevation,powerFlow)
    
    
    #the above reservoirs are at time "t-2"
    
    #GREEN PETER ID=10
    GPR_poolElevation = inner.GetPoolElevationFromVolume(GPR,GPR.volume)
    GPR_outflow = inner.GetResOutflow(GPR,GPR.InitVol,GPR5A.iloc[t+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(GPR,GPR_outflow)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    
    outflows_all[t+2,GPR.ID] = GPR_outflow
    hydropower_all[t+2,5] = GPR_power_output
    
    #FOSTER ID=11
    FOS_poolElevation = inner.GetPoolElevationFromVolume(FOS,FOS.volume)
    FOS_in = GPR_outflow + FOS_loc[t+2]
    FOS_outflow = inner.GetResOutflow(FOS,FOS.InitVol,FOS_in,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FOS,FOS_outflow)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    
    outflows_all[t+2,FOS.ID] = FOS_outflow
    hydropower_all[t+2,6] = FOS_power_output
    
    
    #DETROIT ID=12
    DET_poolElevation = inner.GetPoolElevationFromVolume(DET,DET.volume)
    DET_outflow = inner.GetResOutflow(DET,DET.InitVol,DET5A.iloc[t+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DET,DET_outflow)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    
    outflows_all[t+2,DET.ID] = DET_outflow
    hydropower_all[t+2,7] = DET_power_output
    
    #BIG CLIFF ID=13
    BCL_poolElevation = inner.GetPoolElevationFromVolume(BCL,BCL.volume)
    BCL_outflow = inner.GetResOutflow(BCL,BCL.InitVol,DET_outflow,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BCL,BCL_outflow)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    
    outflows_all[t+2,BCL.ID] = BCL_outflow
    hydropower_all[t+2,8] = BCL_power_output
    
    #the above reservoirs are at time "t"
    
    
    
    
    
    
    
    
