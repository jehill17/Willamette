# -*- coding: utf-8 -*-
"""
Created on Tue May  8 09:09:37 2018

@author: Joy Hill
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
import scipy.stats as stats
#import scipy.interpolate.interp2d as interp2d
from sklearn import linear_model
from sklearn.metrics import r2_score
import datetime as dt
import Willamette_model_updated as inner #reading in the inner function

#I think we need to add the class definition here so we have each reservoir's attributes stored

#import control point historical data

cp_local = pd.read_excel('Controlpoints_local_flows.xls',sheetname=[0,1,2,3,4,5,6,7,8,9]) #when does this data come into play?

#read in historical reservoir inflows -- this will contain the array of 'dates' to use
BCL5M = pd.read_excel('BCL5M_daily.xls') #this data starts 7/01/1928
BLU5A = pd.read_excel('BLU5A_daily.xls')
CGR5A = pd.read_excel('CGR5A_daily.xls')
DET5A = pd.read_excel('DET5A_daily.xls')
DEX5M = pd.read_excel('DEX5M_daily.xls')
DOR5A = pd.read_excel('DOR5A_daily.xls')
FAL5A = pd.read_excel('FAL5A_daily.xls')
FOS5A = pd.read_excel('FOS5A_daily.xls')
FRN5M = pd.read_excel('FRN5M_daily.xls')
GPR5A = pd.read_excel('GPR5A_daily.xls')
HCR5A = pd.read_excel('HCR5A_daily.xls')
LOP5A = pd.read_excel('LOP5A_daily.xls')
LOP5E = pd.read_excel('LOP5E_daily.xls')
COT5A = pd.read_excel('COT5A_daily.xls')
FOS_loc = pd.read_excel('FOS_loc.xls',usecols = [0,3])
LOP_loc = pd.read_excel('LOP_loc.xls',usecols = [0,3])



k = 365 #of days we will run the simulation

outflows_all = np.zeros((k,13)) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.zeros((k,8))

#define an outer fnt here that takes date, name, vol as inputs?

# list with inflow / elev / outflow / spill / RO / power
# Reservoirs network modeling
    # start a daily for loop
    # set inflow /  elev / vol (from elev)
    
    #COT/DOR:
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update goshen
    #update Harrisburg
    
    #COT/DOR:
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update goshen
    #update Harrisburg
    
    #HCR:
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update Jasper
    # update inflow to LOP (use balance equation)
    
    #LOP and FAL which is farthest from targetPoolelev and start from that
    
    # if it's LOP:
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    # update inflow to DEX (= outflow from LOP)
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update Jasper
    
    #if it's FAL :
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update Jasper
    
    #CGR and BLU which is farthest from targetPoolelev and start from that
    #CGR/BLU:
    # GetResOutflow
    # AssignReservoirOutletFlows
    # CalculateHydropowerOutput
    
    # update Vida

for i in range(0,k+1):
    
    #date = inflow_data_dates[k]
    doy = inner.DatetoDayOfYear(date,'%Y/%m/%d')
    
    #COTTAGE GROVE
    COT_outflow = inner.GetResOutflow(COT,COT.InitVol,COT5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(COT,COT_outflow)
    COT_power_output = inner.CalculateHydropowerOutput(COT,COT_poolElevation,powerFlow) #do I need to give a unique name to poweroutflow?
    
    outflows_all[k,COT.ID] = COT_outflow
    hydropower_all[k,0] = COT_power_output
    
    #DORENA
    DOR_outflow = inner.GetResOutflow(DOR,DOR.InitVol,DOR5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DOR,DOR_outflow)
    DOR_power_output = inner.CalculateHydropowerOutput(DOR,DOR_poolElevation,powerFlow)
    
    outflows_all[k,DOR.ID] = DOR_outflow
    hydropower_all[k,1] = DOR_power_output
    
    #FERN RIDGE
    FRN_outflow = inner.GetResOutflow(FRN,FRN.InitVol,FRN5M.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FRN,FRN_outflow)
    FRN_power_output = inner.CalculateHydropowerOutput(FRN,FRN_poolElevation,powerFlow)
    
    outflows_all[k,FRN.ID] = FRN_outflow
    hydropower_all[k,2] = FRN_power_output
    
    #HILLS CREEK
    HCR_outflow = inner.GetResOutflow(HCR,HCR.InitVol,HCR5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(HCR,HCR_outflow)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    
    outflows_all[k,HCR.ID] = HCR_outflow
    hydropower_all[k,3] = HCR_power_output
    
    #LOOKOUT POINT
    LOP_in = HCR_outflow + LOP_loc[k] + LOP5E[k]
    LOP_outflow = inner.GetResOutflow(LOP,LOP.InitVol,LOP_in,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(LOP,LOP_outflow)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    
    outflows_all[k,LOP.ID] = LOP_outflow
    hydropower_all[k,4] = LOP_power_output
    
    #DEXTER
    DEX_outflow = inner.GetResOutflow(DEX,DEX.InitVol,LOP_outflow,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DEX,DEX_outflow)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    
    outflows_all[k,DEX.ID] = DEX_outflow
    hydropower_all[k,5] = DEX_power_output
    
    #FALL CREEK
    FAL_outflow = inner.GetResOutflow(FAL,FAL.InitVol,FAL5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FAL,FAL_outflow)
    FAL_power_output = inner.CalculateHydropowerOutput(FAL,FAL_poolElevation,powerFlow)
    
    #COUGAR
    CGR_outflow = inner.GetResOutflow(CGR,CGR.InitVol,CGR5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(CGR,CGR_outflow)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    
    #BLUE RIVER
    BLU_outflow = inner.GetResOutflow(BLU,BLU.InitVol,BLU5A.iloc[k,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BLU,BLU_outflow)
    BLU_power_output = inner.CalculateHydropowerOutput(BLU,BLU_poolElevation,powerFlow)
    
    
    #the above reservoirs are at time "t-2"
    
    #GREEN PETER
    GPR_outflow = inner.GetResOutflow(GPR,GPR.InitVol,GPR5A.iloc[k+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(GPR,GPR_outflow)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    
    outflows_all[k+2,GPR.ID] = GPR_outflow
    hydropower_all[k+2,6] = GPR_power_output
    
    #FOSTER
    FOS_in = GPR_outflow + FOS_loc[k+2]
    FOS_outflow = inner.GetResOutflow(FOS,FOS.InitVol,FOS_in,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FOS,FOS_outflow)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    
    outflows_all[k+2,FOS.ID] = FOS_outflow
    hydropower_all[k+2,7] = FOS_power_output
    
    
    #DETROIT
    DET_outflow = inner.GetResOutflow(DET,DET.InitVol,DET5A.iloc[k+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DET,DET_outflow)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    
    outflows_all[k+2,DET.ID] = DET_outflow
    hydropower_all[k+2,8] = DET_power_output
    
    #BIG CLIFF
    BCL_outflow = inner.GetResOutflow(BCL,BCL.InitVol,DET_outflow,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BCL,BCL_outflow)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    
    #the above reservoirs are at time "t"
    
    
    
    
    
    
    
    
