# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:11:23 2018

@authors: Joy Hill, SDenaro
"""

#this code will read in ResSim Lite rule curves, control points, etc. for the 13 dams in the Williamette Basin

import matplotlib.pyplot as plt
import pandas as pd 
from pandas.plotting import autocorrelation_plot
from pandas import ExcelWriter
import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from xml.dom import minidom
import xml.etree.ElementTree as ET
import untangle as unt
import xmltodict as xmld
from collections import OrderedDict
import dict_digger as dd


#first need to read in XML file containg reservoir and control point identifying info
with open('Flow.xml') as fd:
    flow = xmld.parse(fd.read())
    
flow_model = flow["flow_model"]

controlPoints = flow_model["controlPoints"]

reservoirs = flow_model["reservoirs"]["reservoir"]


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
        self.maxPowerFlow=[]
        self.Tailwater_elev=[]
        self.Turbine_eff=[]
        

#RESERVOIR rules
#in order of RES ID


#HILLS CREEK
id=1
HCR=Reservoir(1)        
HCR.Restype='Storage'
HCR.AreaVolCurve=pd.read_csv('hills_creek_area_capacity.csv')
HCR.RuleCurve=pd.read_csv('hills_creek_rule_curve.csv')
HCR.Composite=pd.read_csv('HC_composite_rc.csv')
HCR.RO=pd.read_csv('HC_RO_capacity.csv')
HCR.RulePriorityTable=pd.read_csv('hills_creek_rule_priorities.csv')
HCR.Buffer=pd.read_csv('HC_buffer.csv')
HCR.Spillway=pd.read_csv('HC_spillway_capacity.csv')
HCR.InitVol=float(reservoirs[id-1]["@initVolume"])
HCR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
HCR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
HCR.Td_elev=float(reservoirs[id-1]["@td_elev"])
HCR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
HCR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
#HCR.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"]) 
HCR.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
HCR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
HCR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])     
          

#LOOKOUT POINT
id=2
LOP=Reservoir(2)        
LOP.Restype='Storage'
LOP.AreaVolCurve=pd.read_csv('lookout_point_area_capacity.csv')
LOP.RuleCurve=pd.read_csv('lookout_point_rule_curve.csv')
LOP.Composite=pd.read_csv('LO_composite_rc.csv')
LOP.RO=pd.read_csv('LO_RO_capacity.csv')
LOP.RulePriorityTable=pd.read_csv('lookout_rule_priorities.csv')
LOP.Buffer=pd.read_csv('LO_buffer.csv')
LOP.Spillway=pd.read_csv('LO_spillway_capacity.csv')
LOP.InitVol=float(reservoirs[id-1]["@initVolume"])
LOP.minOutflow=float(reservoirs[id-1]["@minOutflow"])
LOP.maxVolume=float(reservoirs[id-1]["@maxVolume"])
LOP.Td_elev=float(reservoirs[id-1]["@td_elev"])
LOP.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
LOP.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
LOP.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"]) 
LOP.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
LOP.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
LOP.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])     


#DEXTER re-regulating
id=3
DEX=Reservoir(3)        
DEX.Restype='RunOfRiver'
DEX.AreaVolCurve=pd.read_csv('dexter_area_capacity.csv')
DEX.Composite=pd.read_csv('Dexter_composite_rc.csv')
DEX.RO=pd.read_csv('Dexter_RO_capacity.csv')
DEX.Spillway=pd.read_csv('Dexter_Spillway_capacity.csv')
DEX.InitVol=float(reservoirs[id-1]["@initVolume"])
DEX.minOutflow=float(reservoirs[id-1]["@minOutflow"])
DEX.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
DEX.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"])   
DEX.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
DEX.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])   


#FALL CREEK
id=4
FAL=Reservoir(4)        
FAL.Restype='Storage'
FAL.AreaVolCurve=pd.read_csv('fall_creek_area_capacity.csv')
FAL.RuleCurve=pd.read_csv('fall_creek_rule_curve.csv')
FAL.Composite=pd.read_csv('FC_composite_rc.csv')
FAL.RO=pd.read_csv('FC_RO_capacity.csv')
FAL.RulePriorityTable=pd.read_csv('fall_creek_rule_priorities.csv')
FAL.Buffer=pd.read_csv('FC_buffer.csv')
FAL.Spillway=pd.read_csv('FC_spillway_capacity.csv')
FAL.InitVol=float(reservoirs[id-1]["@initVolume"])
FAL.minOutflow=float(reservoirs[id-1]["@minOutflow"])
FAL.maxVolume=float(reservoirs[id-1]["@maxVolume"])
FAL.Td_elev=float(reservoirs[id-1]["@td_elev"])
FAL.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
FAL.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
FAL.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"]) 
FAL.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"])     


#DORENA
id=5
DOR=Reservoir(5)        
DOR.Restype='Storage'
DOR.AreaVolCurve=pd.read_csv('dorena_area_capacity.csv')
DOR.RuleCurve=pd.read_csv('dorena_rule_curve.csv')
DOR.Composite=pd.read_csv('Dorena_composite_rc.csv')
DOR.RO=pd.read_csv('Dorena_RO_capacity.csv')
DOR.RulePriorityTable=pd.read_csv('dorena_rule_priorities.csv')
DOR.Buffer=pd.read_csv('Dorena_buffer.csv')
DOR.Spillway=pd.read_csv('Dorena_spillway_capacity.csv')
DOR.InitVol=float(reservoirs[id-1]["@initVolume"])
DOR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
DOR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
DOR.Td_elev=float(reservoirs[id-1]["@td_elev"])
DOR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
DOR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
DOR.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"]) 
DOR.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 


#COTTAGE GROVE
id=6
COT=Reservoir(6)        
COT.Restype='Storage'
COT.AreaVolCurve=pd.read_csv('cottage_grove_area_capacity.csv')
COT.RuleCurve=pd.read_csv('cottage_grove_rule_curve.csv')
COT.Composite=pd.read_csv('CG_composite_rc.csv')
COT.RO=pd.read_csv('CG_RO_capacity.csv')
COT.RulePriorityTable=pd.read_csv('cottage_grove_rule_priorities.csv')
COT.Buffer=pd.read_csv('CG_buffer.csv')
COT.Spillway=pd.read_csv('CG_spillway_capacity.csv')
COT.InitVol=float(reservoirs[id-1]["@initVolume"])
COT.minOutflow=float(reservoirs[id-1]["@minOutflow"])
COT.maxVolume=float(reservoirs[id-1]["@maxVolume"])
COT.Td_elev=float(reservoirs[id-1]["@td_elev"])
COT.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
COT.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
COT.Fc2_elev=float(reservoirs[id-1]["@fc2_elev"]) 
COT.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 


#FERN RIDGE
id=7
FRN=Reservoir(7)        
FRN.Restype='Storage'
FRN.AreaVolCurve=pd.read_csv('fern_ridge_area_capacity.csv')
FRN.RuleCurve=pd.read_csv('fern_ridge_rule_curve.csv')
FRN.Composite=pd.read_csv('FR_composite_rc.csv')
FRN.RO=pd.read_csv('FR_RO_capacity.csv')
FRN.RulePriorityTable=pd.read_csv('fern_ridge_rule_priorities.csv')
FRN.Buffer=pd.read_csv('FR_buffer.csv')
FRN.Spillway=pd.read_csv('FR_spillway_capacity.csv')
FRN.InitVol=float(reservoirs[id-1]["@initVolume"])
FRN.minOutflow=float(reservoirs[id-1]["@minOutflow"])
FRN.maxVolume=float(reservoirs[id-1]["@maxVolume"])
FRN.Td_elev=float(reservoirs[id-1]["@td_elev"])
FRN.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
FRN.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
FRN.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 


#COUGAR
id=8
CGR=Reservoir(8)        
CGR.Restype='Storage'
CGR.AreaVolCurve=pd.read_csv('cougar_area_capacity.csv')
CGR.RuleCurve=pd.read_csv('cougar_rule_curve.csv')
CGR.Composite=pd.read_csv('Cougar_composite_rc.csv')
CGR.RO=pd.read_csv('Cougar_RO_capacity.csv')
CGR.RulePriorityTable=pd.read_csv('cougar_rule_priorities.csv')
CGR.Buffer=pd.read_csv('Cougar_buffer.csv')
CGR.Spillway=pd.read_csv('Cougar_spillway_capacity.csv')
CGR.InitVol=float(reservoirs[id-1]["@initVolume"])
CGR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
CGR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
CGR.Td_elev=float(reservoirs[id-1]["@td_elev"])
CGR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
CGR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
CGR.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
CGR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
CGR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])  

#BLUE RIVER
id=9
BLU=Reservoir(9)        
BLU.Restype='Storage'
BLU.AreaVolCurve=pd.read_csv('blue_river_area_capacity.csv')
BLU.RuleCurve=pd.read_csv('blue_river_rule_curve.csv')
BLU.Composite=pd.read_csv('BR_composite_rc.csv')
BLU.RO=pd.read_csv('BR_RO_capacity.csv')
BLU.RulePriorityTable=pd.read_csv('blue_river_rule_priorities.csv')
BLU.Buffer=pd.read_csv('BR_buffer.csv')
BLU.Spillway=pd.read_csv('BR_spillway_capacity.csv')
BLU.InitVol=float(reservoirs[id-1]["@initVolume"])
BLU.minOutflow=float(reservoirs[id-1]["@minOutflow"])
BLU.maxVolume=float(reservoirs[id-1]["@maxVolume"])
BLU.Td_elev=float(reservoirs[id-1]["@td_elev"])
BLU.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
BLU.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
BLU.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 


#GREEN PETER
id=10
GPR=Reservoir(10)        
GPR.Restype='Storage'
GPR.AreaVolCurve=pd.read_csv('green_peter_area_capacity.csv')
GPR.RuleCurve=pd.read_csv('green_peter_rule_curve.csv')
GPR.Composite=pd.read_csv('GP_composite_rc.csv')
GPR.RO=pd.read_csv('GP_RO_capacity.csv')
GPR.RulePriorityTable=pd.read_csv('green_peter_rule_priorities.csv')
GPR.Buffer=pd.read_csv('GP_buffer.csv')
GPR.Spillway=pd.read_csv('GP_spillway_capacity.csv')
GPR.InitVol=float(reservoirs[id-1]["@initVolume"])
GPR.minOutflow=float(reservoirs[id-1]["@minOutflow"])
GPR.maxVolume=float(reservoirs[id-1]["@maxVolume"])
GPR.Td_elev=float(reservoirs[id-1]["@td_elev"])
GPR.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
GPR.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
GPR.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
GPR.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
GPR.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])     


#FOSTER
id=11
FOS=Reservoir(11)        
FOS.Restype='Storage'
FOS.AreaVolCurve=pd.read_csv('foster_area_capacity.csv')
FOS.RuleCurve=pd.read_csv('foster_rule_curve.csv')
FOS.Composite=pd.read_csv('Foster_composite_rc.csv')
FOS.RO=pd.read_csv('Foster_RO_capacity.csv')
FOS.RulePriorityTable=pd.read_csv('foster_rule_priorities.csv')
FOS.Buffer=pd.read_csv('Foster_buffer.csv')
FOS.Spillway=pd.read_csv('Foster_spillway_capacity.csv')
FOS.InitVol=float(reservoirs[id-1]["@initVolume"])
FOS.minOutflow=float(reservoirs[id-1]["@minOutflow"])
FOS.maxVolume=float(reservoirs[id-1]["@maxVolume"])
FOS.Td_elev=float(reservoirs[id-1]["@td_elev"])
FOS.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
FOS.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
FOS.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
FOS.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
FOS.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])     
          

#DETROIT
id=12
DET=Reservoir(12)        
DET.Restype='Storage'
DET.AreaVolCurve=pd.read_csv('detroit_area_capacity.csv')
DET.RuleCurve=pd.read_csv('detroit_rule_curve.csv')
DET.Composite=pd.read_csv('Detroit_composite_rc.csv')
DET.RO=pd.read_csv('Detroit_RO_capacity.csv')
DET.RulePriorityTable=pd.read_csv('detroit_rule_priorities.csv')
DET.Buffer=pd.read_csv('Detroit_buffer.csv')
DET.Spillway=pd.read_csv('Detroit_spillway_capacity.csv')
DET.InitVol=float(reservoirs[id-1]["@initVolume"])
DET.minOutflow=float(reservoirs[id-1]["@minOutflow"])
DET.maxVolume=float(reservoirs[id-1]["@maxVolume"])
DET.Td_elev=float(reservoirs[id-1]["@td_elev"])
DET.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
DET.Fc1_elev=float(reservoirs[id-1]["@fc1_elev"])      
DET.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"]) 
DET.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
DET.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])     


#BIG CLIFF re-regulating
id=13
BCL=Reservoir(13)        
BCL.Restype='RunOfRiver'
BCL.AreaVolCurve=pd.read_csv('Big_Cliff_area_capacity.csv')
BCL.Composite=pd.read_csv('BC_composite_rc.csv')
BCL.RO=pd.read_csv('BC_RO_capacity.csv')
BCL.Spillway=pd.read_csv('BC_Spillway_capacity.csv')
BCL.InitVol=float(reservoirs[id-1]["@initVolume"])
BCL.minOutflow=float(reservoirs[id-1]["@minOutflow"])
BCL.Inactive_elev=float(reservoirs[id-1]["@inactive_elev"])     
BCL.maxPowerFlow=float(reservoirs[id-1]["@maxPowerFlow"])   
BCL.Tailwater_elev=float(reservoirs[id-1]["@tailwater_elev"]) 
BCL.Turbine_eff=float(reservoirs[id-1]["@turbine_efficiency"])   





#CONTROL POINTS
#in order of ID

#SALEM
Salem_min_flow = pd.read_csv('cp_Min_flow_at_Salem_23791083.csv')
Salem_max_bank = pd.read_csv('cp_Max_bf_Salem_23791083.csv')

#ALBANY
Albany_min_flow = pd.read_csv('cp_Min_Flow_at_Albany_23762845.csv')

#JEFFERSON
Jefferson_max_bank = pd.read_csv('cp_Max_bf_Jefferson_23780423.csv')

#MEHAMA
Mehama_max_bank = pd.read_csv('cp_Max_bf_Mehama_23780481.csv')

#HARRISBURG
Harrisburg_max_42500 = pd.read_csv('cp_Max_HB_42500_cfs_23763337.csv')
Harrisburg_max_51000 = pd.read_csv('cp_Max_HB_51000_cfs_23763337.csv')
Harrisburg_max_70500 = pd.read_csv('cp_Max_HB_70500_cfs_23763337.csv')

#VIDA
Vida_max_flood = pd.read_csv('cp_Max_Vida_Flood_23772903.csv')

#JASPER
Jasper_max_bank = pd.read_csv('cp_Max_Jasper_Bankfull_23751778.csv')

#GOSHEN
Goshen_max_bank = pd.read_csv('cp_Max_bf_Goshen_23759228.csv')

#WATERLOO
Waterloo_max_bank = pd.read_csv('cp_Max_bf_Waterloo_23785687.csv')

#MONROE
Monroe_max_reg = pd.read_csv('cp_Max_Reg_Monroe_23763073.csv')
Monroe_min_flow = pd.read_csv('cp_Min_Flow_at_Monroe_23763073.csv')

#MEHAMA winter
Mehama_max_flow_winter = pd.read_csv('cp_Max_Winter_Ops_Mehama_23780481.csv')

#SALEM winter
Salem_max_flow_winter = pd.read_csv('cp_Max_Winter_Ops_Salem_23791083.csv')

#FOSTER
Foster_max_10K = pd.read_csv('cp_Max_10K_into_Foster_23787257.csv')
Foster_max_12K = pd.read_csv('cp_Max_12K_into_Foster_23787257.csv')
Foster_min_BiOp = pd.read_csv('cp_Min_BiOp_MaxD_at_FOS_23785773.csv') #what is this cp?
Foster_min_con_flow = pd.read_csv('cp_Min_Con_Flow_from_Foster_23785773.csv')
Foster_min_buffer = pd.read_csv('cp_Min_Buffer_Flow_from_Foster_23785773.csv')


#extract each control pt from the ordered dict, then extract the reservoirs influenced information
#use this to create an array of reservoirs influenced for each cp














    
