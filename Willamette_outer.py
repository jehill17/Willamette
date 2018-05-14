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
    cp.COMID=str(controlPoints[id-1]["@location"])



#
##HILLS CREEK
#id=1
#HCR=Reservoir(1)
#HCR.Restype='Storage'
#HCR.AreaVolCurve=pd.read_csv('Area_Capacity_Curves/Hills_creek_area_capacity.csv')
#HCR.RuleCurve=pd.read_csv('Rule_Curves/Hills_creek_rule_curve.csv')
#HCR.Composite=pd.read_csv('HC_composite_rc.csv')
#HCR.RO=pd.read_csv('HC_RO_capacity.csv')
#HCR.RulePriorityTable=pd.read_csv('hills_creek_rule_priorities.csv')
#HCR.Buffer=pd.read_csv('HC_buffer.csv')
#HCR.Spillway=pd.read_csv('HC_spillway_capacity.csv')
#HCR.InitVol=float(reservoirs[id-1]["@initVolume"])
#HCR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#HCR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#HCR.Td_elev=float(reservoirs[id-1]["@td_elev"])
#HCR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#HCR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
##HCR.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
#HCR.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#HCR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#HCR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##LOOKOUT POINT
#id=2
#LOP=Reservoir(2)
#LOP.Restype='Storage'
#LOP.AreaVolCurve=pd.read_csv('lookout_point_area_capacity.csv')
#LOP.RuleCurve=pd.read_csv('lookout_point_rule_curve.csv')
#LOP.Composite=pd.read_csv('LO_composite_rc.csv')
#LOP.RO=pd.read_csv('LO_RO_capacity.csv')
#LOP.RulePriorityTable=pd.read_csv('lookout_rule_priorities.csv')
#LOP.Buffer=pd.read_csv('LO_buffer.csv')
#LOP.Spillway=pd.read_csv('LO_spillway_capacity.csv')
#LOP.InitVol=float(reservoirs[id-1]["@initVolume"])
#LOP.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#LOP.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#LOP.Td_elev=float(reservoirs[id-1]["@td_elev"])
#LOP.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#LOP.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#LOP.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
#LOP.Fc3_elev=float(reservoirs[id-1]["@fc3_elev"])
#LOP.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#LOP.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#LOP.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##DEXTER re-regulating
#id=3
#DEX=Reservoir(3)
#DEX.Restype='RunOfRiver'
#DEX.AreaVolCurve=pd.read_csv('dexter_area_capacity.csv')
#DEX.Composite=pd.read_csv('Dexter_composite_rc.csv')
#DEX.RO=pd.read_csv('Dexter_RO_capacity.csv')
#DEX.Spillway=pd.read_csv('Dexter_Spillway_capacity.csv')
#DEX.InitVol=float(reservoirs[id-1]["@initVolume"])
#DEX.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#DEX.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#DEX.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#DEX.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#DEX.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##FALL CREEK
#id=4
#FAL=Reservoir(4)
#FAL.Restype='Storage'
#FAL.AreaVolCurve=pd.read_csv('fall_creek_area_capacity.csv')
#FAL.RuleCurve=pd.read_csv('fall_creek_rule_curve.csv')
#FAL.Composite=pd.read_csv('FC_composite_rc.csv')
#FAL.RO=pd.read_csv('FC_RO_capacity.csv')
#FAL.RulePriorityTable=pd.read_csv('fall_creek_rule_priorities.csv')
#FAL.Buffer=pd.read_csv('FC_buffer.csv')
#FAL.Spillway=pd.read_csv('FC_spillway_capacity.csv')
#FAL.InitVol=float(reservoirs[id-1]["@initVolume"])
#FAL.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#FAL.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#FAL.Td_elev=float(reservoirs[id-1]["@td_elev"])
#FAL.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#FAL.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#FAL.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
#FAL.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#
#
##DORENA
#id=5
#DOR=Reservoir(5)
#DOR.Restype='Storage'
#DOR.AreaVolCurve=pd.read_csv('dorena_area_capacity.csv')
#DOR.RuleCurve=pd.read_csv('dorena_rule_curve.csv')
#DOR.Composite=pd.read_csv('Dorena_composite_rc.csv')
#DOR.RO=pd.read_csv('Dorena_RO_capacity.csv')
#DOR.RulePriorityTable=pd.read_csv('dorena_rule_priorities.csv')
#DOR.Buffer=pd.read_csv('Dorena_buffer.csv')
#DOR.Spillway=pd.read_csv('Dorena_spillway_capacity.csv')
#DOR.InitVol=float(reservoirs[id-1]["@initVolume"])
#DOR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#DOR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#DOR.Td_elev=float(reservoirs[id-1]["@td_elev"])
#DOR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#DOR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#DOR.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
#DOR.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#
#
##COTTAGE GROVE
#id=6
#COT=Reservoir(6)
#COT.Restype='Storage'
#COT.AreaVolCurve=pd.read_csv('cottage_grove_area_capacity.csv')
#COT.RuleCurve=pd.read_csv('cottage_grove_rule_curve.csv')
#COT.Composite=pd.read_csv('CG_composite_rc.csv')
#COT.RO=pd.read_csv('CG_RO_capacity.csv')
#COT.RulePriorityTable=pd.read_csv('cottage_grove_rule_priorities.csv')
#COT.Buffer=pd.read_csv('CG_buffer.csv')
#COT.Spillway=pd.read_csv('CG_spillway_capacity.csv')
#COT.InitVol=float(reservoirs[id-1]["@initVolume"])
#COT.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#COT.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#COT.Td_elev=float(reservoirs[id-1]["@td_elev"])
#COT.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#COT.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#COT.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"])
#COT.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#
#
##FERN RIDGE
#id=7
#FRN=Reservoir(7)
#FRN.Restype='Storage'
#FRN.AreaVolCurve=pd.read_csv('fern_ridge_area_capacity.csv')
#FRN.RuleCurve=pd.read_csv('fern_ridge_rule_curve.csv')
#FRN.Composite=pd.read_csv('FR_composite_rc.csv')
#FRN.RO=pd.read_csv('FR_RO_capacity.csv')
#FRN.RulePriorityTable=pd.read_csv('fern_ridge_rule_priorities.csv')
#FRN.Buffer=pd.read_csv('FR_buffer.csv')
#FRN.Spillway=pd.read_csv('FR_spillway_capacity.csv')
#FRN.InitVol=float(reservoirs[id-1]["@initVolume"])
#FRN.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#FRN.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#FRN.Td_elev=float(reservoirs[id-1]["@td_elev"])
#FRN.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#FRN.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#FRN.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#
#
##COUGAR
#id=8
#CGR=Reservoir(8)
#CGR.Restype='Storage'
#CGR.AreaVolCurve=pd.read_csv('cougar_area_capacity.csv')
#CGR.RuleCurve=pd.read_csv('cougar_rule_curve.csv')
#CGR.Composite=pd.read_csv('Cougar_composite_rc.csv')
#CGR.RO=pd.read_csv('Cougar_RO_capacity.csv')
#CGR.RulePriorityTable=pd.read_csv('cougar_rule_priorities.csv')
#CGR.Buffer=pd.read_csv('Cougar_buffer.csv')
#CGR.Spillway=pd.read_csv('Cougar_spillway_capacity.csv')
#CGR.InitVol=float(reservoirs[id-1]["@initVolume"])
#CGR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#CGR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#CGR.Td_elev=float(reservoirs[id-1]["@td_elev"])
#CGR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#CGR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#CGR.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#CGR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#CGR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
##BLUE RIVER
#id=9
#BLU=Reservoir(9)
#BLU.Restype='Storage'
#BLU.AreaVolCurve=pd.read_csv('blue_river_area_capacity.csv')
#BLU.RuleCurve=pd.read_csv('blue_river_rule_curve.csv')
#BLU.Composite=pd.read_csv('BR_composite_rc.csv')
#BLU.RO=pd.read_csv('BR_RO_capacity.csv')
#BLU.RulePriorityTable=pd.read_csv('blue_river_rule_priorities.csv')
#BLU.Buffer=pd.read_csv('BR_buffer.csv')
#BLU.Spillway=pd.read_csv('BR_spillway_capacity.csv')
#BLU.InitVol=float(reservoirs[id-1]["@initVolume"])
#BLU.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#BLU.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#BLU.Td_elev=float(reservoirs[id-1]["@td_elev"])
#BLU.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#BLU.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#BLU.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#
#
##GREEN PETER
#id=10
#GPR=Reservoir(10)
#GPR.Restype='Storage'
#GPR.AreaVolCurve=pd.read_csv('green_peter_area_capacity.csv')
#GPR.RuleCurve=pd.read_csv('green_peter_rule_curve.csv')
#GPR.Composite=pd.read_csv('GP_composite_rc.csv')
#GPR.RO=pd.read_csv('GP_RO_capacity.csv')
#GPR.RulePriorityTable=pd.read_csv('green_peter_rule_priorities.csv')
#GPR.Buffer=pd.read_csv('GP_buffer.csv')
#GPR.Spillway=pd.read_csv('GP_spillway_capacity.csv')
#GPR.InitVol=float(reservoirs[id-1]["@initVolume"])
#GPR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#GPR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#GPR.Td_elev=float(reservoirs[id-1]["@td_elev"])
#GPR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#GPR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#GPR.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#GPR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#GPR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##FOSTER
#id=11
#FOS=Reservoir(11)
#FOS.Restype='Storage'
#FOS.AreaVolCurve=pd.read_csv('foster_area_capacity.csv')
#FOS.RuleCurve=pd.read_csv('foster_rule_curve.csv')
#FOS.Composite=pd.read_csv('Foster_composite_rc.csv')
#FOS.RO=pd.read_csv('Foster_RO_capacity.csv')
#FOS.RulePriorityTable=pd.read_csv('foster_rule_priorities.csv')
#FOS.Buffer=pd.read_csv('Foster_buffer.csv')
#FOS.Spillway=pd.read_csv('Foster_spillway_capacity.csv')
#FOS.InitVol=float(reservoirs[id-1]["@initVolume"])
#FOS.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#FOS.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#FOS.Td_elev=float(reservoirs[id-1]["@td_elev"])
#FOS.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#FOS.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#FOS.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#FOS.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#FOS.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##DETROIT
#id=12
#DET=Reservoir(12)
#DET.Restype='Storage'
#DET.AreaVolCurve=pd.read_csv('Detroit_area_capacity.csv')
#DET.Composite=pd.read_csv('Detroit_composite_rc.csv')
#DET.RO=pd.read_csv('Detroit_RO_capacity.csv')
#DET.RulePriorityTable=pd.read_csv('detroit_rule_priorities.csv')
#DET.Buffer=pd.read_csv('Detroit_buffer.csv')
#DET.Spillway=pd.read_csv('Detroit_spillway_capacity.csv')
#DET.InitVol=float(reservoirs[id-1]["@initVolume"])
#DET.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#DET.maxVolume=float(reservoirs[id-1]["@maxVolume"])
#DET.Td_elev=float(reservoirs[id-1]["@td_elev"])
#DET.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#DET.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])
#DET.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#DET.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#DET.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])
#
#
##BIG CLIFF re-regulating
#id=13
#BCL=Reservoir(13)
#BCL.Restype='RunOfRiver'
#BCL.AreaVolCurve=pd.read_csv('Big_Cliff_area_capacity.csv')
#BCL.Composite=pd.read_csv('BC_composite_rc.csv')
#BCL.RO=pd.read_csv('BC_RO_capacity.csv')
#BCL.Spillway=pd.read_csv('BC_Spillway_capacity.csv')
#BCL.InitVol=float(reservoirs[id-1]["@initVolume"])
#BCL.minOutflow=float(reservoirs[id-1]["@minOutflow"])
#BCL.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])
#BCL.GateMaxPowerFlow==float(reservoirs[id-1]["@maxPowerFlow"])
#BCL.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"])
#BCL.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])


#CONTROL POINTS
#in order of ID

#SALEM
SAL=ControlPoint(1)
SAL.influencedReservoirs =float(controlPoints[id-1]["@reservoirs"])
SAL.COMID=str(controlPoints[id-1]["@location"])
#SAL.COMID = str(23791083)


#ALBANY
ALB=ControlPoint(2)
ALB.influencedReservoirs = []
ALB.COMID = str(23762845)

#JEFFERSON
JEF=ControlPoint(3)
JEF.influencedReservoirs = []
JEF.COMID = str(23780423)

#MEHAMA
MEH=ControlPoint(4)
MEH.influencedReservoirs = []
MEH.COMID = str(23780481)

#HARRISBURG
HAR=ControlPoint(5)
HAR.influencedReservoirs = []
HAR.COMID =str(23763337)

#VIDA
VID=ControlPoint(6)
VID.influencedReservoirs = []
VID.COMID =str(23772903)

#JASPER
JAS=ControlPoint(7)
JAS.influencedReservoirs = [HCR, LOP, FAL]
JAS.COMID =str(23751778)

#GOSHEN
GOS=ControlPoint(8)
GOS.influencedReservoirs = []
GOS.COMID =str(23759228)

#WATERLOO
WAT=ControlPoint(9)
WAT.influencedReservoirs = []
WAT.COMID=str(23785687)

#MONROE
MON=ControlPoint(10)
MON.influencedReservoirs = []
MON.COMID =str(23763073)

#FOSTER
FOS=ControlPoint(11)
FOS.influencedReservoirs = []
FOS.COMID  =str(23787257)




#import control point historical data-- shifted one day before

#convert data
cfs_to_cms = 0.0283168

#cp_hist: start this at 12/31/2004
SAL_2005 = pd.read_excel('Control point historical discharge 2005.xlsx',sheetname='Salem')
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


#read in historical reservoir inflows -- this will contain the array of 'dates' to use
BLU5A = pd.read_excel('BLU5A_daily.xls',skiprows=27943,skip_footer =1004) #only using data from 2005
BLU5A.columns = ['Date','Inflow']
CGR5A = pd.read_excel('CGR5A_daily.xls',skiprows=27943,skip_footer =1004)
DET5A = pd.read_excel('DET5A_daily.xls',skiprows=27943,skip_footer =1004)
DEX5M = pd.read_excel('DEX5M_daily.xls',skiprows=27943,skip_footer =1004)
DOR5A = pd.read_excel('DOR5A_daily.xls',skiprows=27943,skip_footer =1004)
FAL5A = pd.read_excel('FAL5A_daily.xls',skiprows=27943,skip_footer =1004)
FOS5A = pd.read_excel('FOS5A_daily.xls',skiprows=27943,skip_footer =1004)
FRN5M = pd.read_excel('FRN5M_daily.xls',skiprows=27943,skip_footer =1004)
GPR5A = pd.read_excel('GPR5A_daily.xls',skiprows=27943,skip_footer =1004)
HCR5A = pd.read_excel('HCR5A_daily.xls',skiprows=27943,skip_footer =1004)
LOP5A = pd.read_excel('LOP5A_daily.xls',skiprows=27943,skip_footer =1004)
LOP5E = pd.read_excel('LOP5E_daily.xls',skiprows=27943,skip_footer =1004)
COT5A = pd.read_excel('COT5A_daily.xls',skiprows=27943,skip_footer =1004)
FOS_loc = pd.read_excel('FOS_loc.xls',usecols = [0,3],skiprows=27943,skip_footer =1004)
LOP_loc = pd.read_excel('LOP_loc.xls',usecols = [0,3],skiprows=27943,skip_footer =1004)

dates = np.array(BLU5A['Date'])


T = 365 # Set the simulation horizon

n_res=13
n_HPres=8
n_cp = 11


#=======
# allocate output 
outflows_all = np.nan((T+1,n_res)) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.nan((T+1,n_HPres))
volumes_all = np.nan((T+1,n_res))
elevations_all = np.nan((T+1,n_res))
cp_discharge_all = np.nan((T+1,n_cp))

#initialize values
for  i in range(0,n_res):
    outflows_all[0,i] = res_list[i].init_outflow
    volumes_all[0,i] = res_list[i].InitVol 
    elevations_all[0,i]=inner.GetPoolElevationFromVolume(volumes_all[0,i])

for  i in range(0,n_cp):
    cp_discharge_all[0,i] = cp_list[i].init_discharge

#define an outer fnt here that takes date, name, vol as inputs?

InitwaterYear = 1.2

for t in range(0,T+1):
    
    doy = inner.DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')
    
    waterYear = inner.UpdateReservoirWaterYear(doy)
    #calculate waterYear
    #conditional based on doy 
    #calculate at doy = 140
    
    #COTTAGE GROVE 
    COT_poolElevation = inner.GetPoolElevationfromVolume(COT,volumes_all[t,COT.ID]) 
    COT_outflow = inner.GetResOutflow(COT,volumes_all[t,COT.ID],COT5A.iloc[t,1],outflows_all[t,COT.ID],doy,waterYear,cp_list,cp_discharge_all[t,:])
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(COT,COT_outflow)
    COT_power_output = inner.CalculateHydropowerOutput(COT,COT_poolElevation,powerFlow) #do I need to give a unique name to poweroutflow?
    
    outflows_all[t+1,COT.ID] = COT_outflow
#    hydropower_all[t,0] = COT_power_output
    
    
    #DORENA
    DOR_poolElevation = inner.GetPoolElevationFromVolume(DOR,DOR.volume)
    DOR_outflow = inner.GetResOutflow(DOR,DOR.InitVol,DOR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DOR,DOR_outflow)
    DOR_power_output = inner.CalculateHydropowerOutput(DOR,DOR_poolElevation,powerFlow)
    
    outflows_all[t,DOR.ID] = DOR_outflow
#    hydropower_all[t,1] = DOR_power_output
    
    #FERN RIDGE
    FRN_poolElevation = inner.GetPoolElevationFromVolume(FRN,FRN.volume)
    FRN_outflow = inner.GetResOutflow(FRN,FRN.InitVol,FRN5M.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FRN,FRN_outflow)
    FRN_power_output = inner.CalculateHydropowerOutput(FRN,FRN_poolElevation,powerFlow)
    
    outflows_all[t,FRN.ID] = FRN_outflow
#    hydropower_all[t,2] = FRN_power_output
    
    #HILLS CREEK
    HCR_poolElevation = inner.GetPoolElevationFromVolume(HCR,HCR.volume)
    HCR_outflow = inner.GetResOutflow(HCR,HCR.InitVol,HCR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(HCR,HCR_outflow)
    HCR_power_output = inner.CalculateHydropowerOutput(HCR,HCR_poolElevation,powerFlow)
    
    outflows_all[t,HCR.ID] = HCR_outflow
    hydropower_all[t,1] = HCR_power_output
    
    #LOOKOUT POINT
    LOP_poolElevation = inner.GetPoolElevationFromVolume(LOP,LOP.volume)
    LOP_in = HCR_outflow + LOP_loc[t] + LOP5E[t]
    LOP_outflow = inner.GetResOutflow(LOP,LOP.InitVol,LOP_in,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(LOP,LOP_outflow)
    LOP_power_output = inner.CalculateHydropowerOutput(LOP,LOP_poolElevation,powerFlow)
    
    outflows_all[t,LOP.ID] = LOP_outflow
    hydropower_all[t,2] = LOP_power_output
    
    #DEXTER
    DEX_poolElevation = inner.GetPoolElevationFromVolume(DEX,DEX.volume)
    DEX_outflow = inner.GetResOutflow(DEX,DEX.InitVol,LOP_outflow,doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DEX,DEX_outflow)
    DEX_power_output = inner.CalculateHydropowerOutput(DEX,DEX_poolElevation,powerFlow)
    
    outflows_all[t,DEX.ID] = DEX_outflow
    hydropower_all[t,3] = DEX_power_output
    
    #FALL CREEK
    FAL_poolElevation = inner.GetPoolElevationFromVolume(FAL,FAL.volume)
    FAL_outflow = inner.GetResOutflow(FAL,FAL.InitVol,FAL5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FAL,FAL_outflow)
    FAL_power_output = inner.CalculateHydropowerOutput(FAL,FAL_poolElevation,powerFlow)
    
    #COUGAR
    CGR_poolElevation = inner.GetPoolElevationFromVolume(CGR,CGR.volume)
    CGR_outflow = inner.GetResOutflow(CGR,CGR.InitVol,CGR5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(CGR,CGR_outflow)
    CGR_power_output = inner.CalculateHydropowerOutput(CGR,CGR_poolElevation,powerFlow)
    
    outflows_all[t,CGR.ID] = CGR_outflow
    hydropower_all[t,4] = CGR_power_output
    
    #BLUE RIVER
    BLU_poolElevation = inner.GetPoolElevationFromVolume(BLU,BLU.volume)
    BLU_outflow = inner.GetResOutflow(BLU,BLU.InitVol,BLU5A.iloc[t,1],doy,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BLU,BLU_outflow)
    BLU_power_output = inner.CalculateHydropowerOutput(BLU,BLU_poolElevation,powerFlow)
    
    
    #the above reservoirs are at time "t-2"
    
    #GREEN PETER
    GPR_poolElevation = inner.GetPoolElevationFromVolume(GPR,GPR.volume)
    GPR_outflow = inner.GetResOutflow(GPR,GPR.InitVol,GPR5A.iloc[t+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(GPR,GPR_outflow)
    GPR_power_output = inner.CalculateHydropowerOutput(GPR,GPR_poolElevation,powerFlow)
    
    outflows_all[t+2,GPR.ID] = GPR_outflow
    hydropower_all[t+2,5] = GPR_power_output
    
    #FOSTER
    FOS_poolElevation = inner.GetPoolElevationFromVolume(FOS,FOS.volume)
    FOS_in = GPR_outflow + FOS_loc[t+2]
    FOS_outflow = inner.GetResOutflow(FOS,FOS.InitVol,FOS_in,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(FOS,FOS_outflow)
    FOS_power_output = inner.CalculateHydropowerOutput(FOS,FOS_poolElevation,powerFlow)
    
    outflows_all[t+2,FOS.ID] = FOS_outflow
    hydropower_all[t+2,6] = FOS_power_output
    
    
    #DETROIT
    DET_poolElevation = inner.GetPoolElevationFromVolume(DET,DET.volume)
    DET_outflow = inner.GetResOutflow(DET,DET.InitVol,DET5A.iloc[t+2,1],doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(DET,DET_outflow)
    DET_power_output = inner.CalculateHydropowerOutput(DET,DET_poolElevation,powerFlow)
    
    outflows_all[t+2,DET.ID] = DET_outflow
    hydropower_all[t+2,7] = DET_power_output
    
    #BIG CLIFF
    BCL_poolElevation = inner.GetPoolElevationFromVolume(BCL,BCL.volume)
    BCL_outflow = inner.GetResOutflow(BCL,BCL.InitVol,DET_outflow,doy+2,waterYear)
    [powerFlow,RO_flow,spillwayFlow] = inner.AssignReservoirOutletFlows(BCL,BCL_outflow)
    BCL_power_output = inner.CalculateHydropowerOutput(BCL,BCL_poolElevation,powerFlow)
    
    outflows_all[t+2,BCL.ID] = BCL_outflow
    hydropower_all[t+2,8] = BCL_power_output
    
    #the above reservoirs are at time "t"
    
    
    
    
    
    
    
    
