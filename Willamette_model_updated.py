# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:11:23 2018

@authors: Joy Hill, Simona Denaro
"""

#this code will read in ResSim Lite rule curves, control points, etc. for the 13 dams in the Williamette Basin

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


#first need to read in XML file containg reservoir and control point identifying info
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
        self.maxPowerFlow=[]
        self.Tailwater_elev=[]
        self.Turbine_eff=[]
        self.inflow=[]
        self.outflow=[]
        self.powerflow=[]
        self.RO_flow =[]
        self.spillwayflow =[]
        self.zone=[]


#Create ControlPoint class
class ControlPoint:
    def __init__(self, ID):
        self.ID=ID
        self.influencedReservoirs=[]
        self.discharge = []
        self.localFlow = []


#RESERVOIR rules
#in order of RES ID


#HILLS CREEK
id=1
HCR=Reservoir(1)
HCR.Restype='Storage'
HCR.AreaVolCurve=pd.read_csv('Hills_creek_area_capacity.csv')
HCR.RuleCurve=pd.read_csv('Hills_creek_rule_curve.csv')
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
GPR.Fc2_elev = 289.6 #these values are from the release notes
#GPR.Fc3_elev = 301.8 #these values are from the release notes
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
DET.AreaVolCurve=pd.read_csv('Detroit_area_capacity.csv')
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
#DET.Fc2_elev = 478 this value in release notes??
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
SAL=ControlPoint(1)
SAL.influencedReservoirs = []  #are we filling this in?

#ALBANY
ALB=ControlPoint(2)
ALB.influencedReservoirs = []

#JEFFERSON
JEF=ControlPoint(3)
JEF.influencedReservoirs = []

#MEHAMA
MEH=ControlPoint(4)
MEH.influencedReservoirs = []

#HARRISBURG
HAR=ControlPoint(5)
HAR.influencedReservoirs = []

#VIDA
VID=ControlPoint(6)
VID.influencedReservoirs = []

#JASPER
JAS=ControlPoint(7)
JAS.influencedReservoirs = []

#GOSHEN
GOS=ControlPoint(8)
GOS.influencedReservoirs = []

#WATERLOO
WAT=ControlPoint(9)
WAT.influencedReservoirs = []

#MONROE
MON=ControlPoint(10)
MON.influencedReservoirs = []

#FOSTER
FOS=ControlPoint(11)
FOS.influencedReservoirs = []



####################

def DatetoDayOfYear(val, fmt): #this might need to be added to outer function
#val = '2012/11/07'
#fmt = '%Y/%m/%d'
    date_d = dt.datetime.strptime(val,fmt)
    tt = date_d.timetuple()
    doy = tt.tm_yday

    return doy


def GetPoolElevationFromVolume(volume,name):
    if name.AreaVolCurve is None:
        return 0
    else:
        poolElevation = np.interp(volume,name.AreaVolCurve['Storage_m3'],name.AreaVolCurve['Elevation_m'])

        return poolElevation #returns pool elevation (m)


def GetPoolVolumeFromElevation(pool_elev,name): #do we need to use this funt if our initial condition is volume?
    if name.AreaVolCurve is None:
        return 0
    else:
        poolVolume = np.interp(pool_elev,name.AreaVolCurve['Elevation_m'],name.AreaVolCurve['Storage_m3'])

        return poolVolume #returns pool vol(m^3)


def GetBufferZoneElevation(doy,name):
    if name.Buffer is None:
        return 0
    else:
        bufferZoneElevation = np.interp(doy,name.Buffer['Date'],name.Buffer['Pool_elevation_m'])

        return bufferZoneElevation #returns what the buffer zone elevation level is for this time of year (in m)


def GetTargetElevationFromRuleCurve(doy,name): #target_table is same as rule_curve
    if name.RuleCurve is None:
        return 0
    else:
        target = np.interp(doy,name.RuleCurve['Day'],name.RuleCurve['Cons_Pool_elev_m'])

        return target #target pool elevation in m


def UpdateMaxGateOutflows(name,poolElevation): #not sure if these are the right inputs
    if name.Restype == 'Storage':
        if name.RO is not None:
            name.maxRO_flow = np.interp(poolElevation,name.RO['pool_elev_m'],name.RO['release_cap_cms'])
#         return maxRO_flow
        name.maxSpillway_flow = np.interp(poolElevation,name.Spillway['pool_elev_m'],name.Spillway['release_cap_cms'])
        return (name.maxRO_flow,name.maxSpillway_flow)

#def GetResOutflow(name, volume, inflow, doy, waterYear):
#    currentPoolElevation = GetPoolElevationFromVolume(volume,name)
#    UpdateMaxGateOutflows( name, currentPoolElevation )
#    if name.Restype=='RunOfRiver':
#      outflow = inflow / 86400   # convert outflow to m3/day
#    else:
#      targetPoolElevation = GetTargetElevationFromRuleCurve( doy, name );
#      targetPoolVolume    = GetPoolVolumeFromElevation(targetPoolElevation, name);
#      currentVolume = GetPoolVolumeFromElevation(currentPoolElevation, name);
#      bufferZoneElevation = GetBufferZoneElevation( doy, name );
#
#      if currentVolume > name.maxVolume:
#         currentVolume = name.maxVolume;   #Don't allow res volumes greater than max volume.This code may be removed once hydro model is calibrated.
#         currentPoolElevation = GetPoolElevationFromVolume(currentVolume, name);  #Adjust pool elevation
#         print ("Reservoir volume at ", name, " on day of year ", doy, " exceeds maximum. Volume set to maxVolume. Mass Balance not closed.")
#        
#
#      desiredRelease = (currentVolume - targetPoolVolume)/86400     #This would bring pool elevation back to the rule curve in one timestep.  Converted from m3/day to m3/s  (cms).
#                                                                          #This would be ideal if possible within the given constraints.
#      if currentVolume < targetPoolVolume: #if we want to fill the reservoir
#         desiredRelease=name.minOutflow;
#
#      actualRelease = desiredRelease   #Before any constraints are applied
#      
#      # ASSIGN ZONE:  zone 0 = top of dam, zone 1 = flood control high, zone 2 = conservation operations, zone 3 = buffer, zone 4 = alternate flood control 1, zone 5 = alternate flood control 2.  
#
#      if currentPoolElevation > name.Td_elev:
#         print ("Reservoir elevation at ", name, "on day of year ", doy," exceeds dam top elevation.")
#         currentPoolElevation = name.Td_elev - 0.1    #Set just below top elev to keep from exceeding values in lookup tables
#         zone = 0    
#      elif currentPoolElevation > name.Fc1_elev:
#         zone = 0
#      elif currentPoolElevation > targetPoolElevation:
#         if name.Fc2_elev!=[] and currentPoolElevation <= name.Fc2_elev:
#            zone = 4
#         elif name.Fc3_elev!=[] and currentPoolElevation > name.Fc3_elev:
#            zone = 5
#         else:
#            zone = 1
#      elif currentPoolElevation <= targetPoolElevation:
#         if currentPoolElevation <= bufferZoneElevation:   #in the buffer zone (HERE WE REMOVED THE PART THAT COUNTED THE DAYS IN THE BUFFER ZONE)
#            zone = 3     
#         else:                                           #in the conservation zone
#            zone = 2
#      else:
#         print ("*** GetResOutflow(): We should never get here. doy = ", doy ,"reservoir = ", name)
#      
#        # Once we know what zone we are in, we can access the array of appropriate constraints for the particular reservoir and zone.       
#
#      constraint_array=name.RulePriorityTable.iloc[:,zone]
#      constraint_array=constraint_array[(constraint_array !='Missing')] #eliminate 'Missing' rows
#      #Loop through constraints and modify actual release.  Apply the flood control rules in order here.
#      for i in range (0, len(constraint_array)-1):
#          #open the csv file and get first column label and appropriate data to use for lookup
#          constraintRules = pd.read_csv(constraint_array[i])
#          xlabel = list(constraintRules)[0]   
#          xvalue = []
#          yvalue = []
#          if xlabel=="date":                           # Date based rule?  xvalue = current date.
#             xvalue = doy
#          elif xlabel=="release_cms":                  # Release based rule?  xvalue = release last timestep
#               xvalue = name.outflow/86400              # SEC_PER_DAY = cubic meters per second to cubic meters per day
#          elif xlabel=="pool_elev_m" :                 # Pool elevation based rule?  xvalue = pool elevation (meters)
#               xvalue = currentPoolElevation
#          elif xlabel=="inflow_cms":                    # Inflow based rule?   xvalue = inflow to reservoir
#               xvalue = name.inflow/86400              
#          elif xlabel=="Outflow_lagged_24h":            #24h lagged outflow based rule?   xvalue = outflow from reservoir at last timestep
#               xvalue = name.outflow/8640               #placeholder (assumes that timestep=24 hours)
#          elif xlabel=="Date_pool_elev_m":             # Lookup based on two values...date and pool elevation.  x value is date.  y value is pool elevation
#               xvalue = doy
#               yvalue = currentPoolElevation
#          elif xlabel=="Date_Water_year_type":          #Lookup based on two values...date and wateryeartype (storage in 13 USACE reservoirs on May 20th).
#               xvalue = doy
#               yvalue = waterYear
#          elif xlabel == "Date_release_cms": 
#               xvalue = doy
#               yvalue = name.outflow/86400
#          else:                                            #Unrecognized xvalue for constraint lookup table
#              "Unrecognized x value for reservoir constraint lookup label = ", xlabel 
#   
#          if constraint_array[i].startswith('Max_'):  #case RCT_MAX  maximum
#             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
#                 cols=constraintRules.iloc[0,1::]
#                 rows=constraintRules.iloc[1::,0]
#                 vals=constraintRules.iloc[1::,1::]
#                 interp_table = interp2d(cols, rows, vals, kind='linear')
#                 constraintValue = interp_table(xvalue, yvalue)    
#             else:             #//If not, just use xvalue
#                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#             if actualRelease >= constraintValue:
#                actualRelease = constraintValue;
#
#
#          elif constraint_array[i].startswith('Min_'):  # case RCT_MIN:  //minimum
#             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
#                 cols=constraintRules.iloc[0,1::]
#                 rows=constraintRules.iloc[1::,0]
#                 vals=constraintRules.iloc[1::,1::]
#                 interp_table = interp2d(cols, rows, vals, kind='linear')
#                 constraintValue = interp_table(xvalue, yvalue)                 
#             else:             #//If not, just use xvalue
#                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#             if actualRelease <= constraintValue:
#                actualRelease = constraintValue;
#
#
#          elif constraint_array[i].startswith('MaxI_'):     #case RCT_INCREASINGRATE:  //Increasing Rate
#             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
#                 cols=constraintRules.iloc[0,1::]
#                 rows=constraintRules.iloc[1::,0]
#                 vals=constraintRules.iloc[1::,1::]
#                 interp_table = interp2d(cols, rows, vals, kind='linear')
#                 constraintValue = interp_table(xvalue, yvalue)
#                 constraintValue = constraintValue*24   #Covert hourly to daily                  
#             else:             #//If not, just use xvalue
#                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#                 constraintValue = constraintValue*24   #Covert hourly to daily
#             if actualRelease >= name.outflow/86400 + constraintValue:  #Is planned release more than current release + contstraint? 
#                actualRelease = (name.outflow/86400) + constraintValue  #If so, planned release can be no more than current release + constraint.
#                 
#
#          elif constraint_array[i].startswith('MaxD_'):    #case RCT_DECREASINGRATE:  //Decreasing Rate
#             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
#                 cols=constraintRules.iloc[0,1::]
#                 rows=constraintRules.iloc[1::,0]
#                 vals=constraintRules.iloc[1::,1::]
#                 interp_table = interp2d(cols, rows, vals, kind='linear')
#                 constraintValue = interp_table(xvalue, yvalue)
#                 constraintValue = constraintValue*24   #Covert hourly to daily                  
#             else:             #//If not, just use xvalue
#                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#                 constraintValue = constraintValue*24   #Covert hourly to daily
#             if actualRelease >= name.outflow/86400 - constraintValue:  #Is planned release less than current release - contstraint? 
#                actualRelease = (name.outflow/86400) - constraintValue  #If so, planned release can be no less than current release - constraint.
#
#
#          elif constraint_array[i].startswith('Pow_Max'): # case RCT_POWERPLANT:  //maximum Power plant rule  Assign m_maxPowerFlow attribute.
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               maxPowerFlow = constraintValue  #Just for this timestep.  name.MaxPowerFlow is the physical limitation for the reservoir.
#          elif constraint_array[i].startswith('Pow_Min'): # case RCT_POWERPLANT:  //minimum Power plant rule  Assign m_minPowerFlow attribute.
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               minPowerFlow = constraintValue 
#           
#            
#          elif constraint_array[i].startswith('RO_Max'): #case RCT_REGULATINGOUTLET:  Max Regulating outlet rule, Assign m_maxRO_Flow attribute.
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               maxRO_Flow  = constraintValue  
#          elif constraint_array[i].startswith('RO_Min'): #Min Regulating outlet rule, Assign m_maxRO_Flow attribute.
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               minRO_Flow  = constraintValue 
#
#
#          elif constraint_array[i].startswith('Spill_Max'): #  case RCT_SPILLWAY:   //Max Spillway rule
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               maxSpillwayFlow  = constraintValue  
#          elif constraint_array[i].startswith('Spill_Min'): #Min Spillway rule
#               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
#               minSpillwayFlow  = constraintValue 
#
#
#          elif constraint_array[i].startswith('cp_'):  #case RCT_CONTROLPOINT:  #Downstream control point 
#                  {
#                   CString filename = pConstraint->m_constraintFileName;
#                  # get control point location.  Assumes last characters of filename contain a COMID
#                  LPTSTR p = (LPTSTR) _tcsrchr( filename, '.' );
#   
#                  if ( p != NULL )
#                     {
#                     p--;
#                     while ( isdigit( *p ) )
#                        p--;
#   
#                     int comID = atoi( p+1 );
#                     pConstraint->m_comID = comID;
#                     }
#                        
#                  #Determine which control point this is.....use COMID to identify
#                  for (int k=0;  k < gpModel->m_controlPointArray.GetSize(); k++) 
#                     {
#                     ControlPoint *pControl = gpModel->m_controlPointArray.GetAt(k);
#                     ASSERT( pControl != NULL );
#
#                     if ( pControl->InUse() )
#                        {                     
#                        int location = pControl->m_location;     # Get COMID of this control point
#                        #if (pControl->m_location == pConstraint->m_comID)  #Do they match?
#                        if (_stricmp(pControl->m_controlPointFileName,pConstraint->m_constraintFileName) == 0)  #compare names
#                           {
#                           ASSERT( pControl->m_pResAllocation != NULL );
#                           constraintValue = 0.0f;
#                           int releaseFreq = 1;
#
#                           for ( int l=0; l < pControl->m_influencedReservoirsArray.GetSize(); l++ )
#                              {
#                              if ( pControl->m_influencedReservoirsArray[ l ] == pRes )
#                              {
#                                 if (pRes->m_reservoirType == ResType_CtrlPointControl)
#                                 {
#                                    int rowCount = pControl->m_pResAllocation->GetRowCount();
#                                    releaseFreq = pControl->m_influencedReservoirsArray[l]->m_releaseFreq;
#
#                                    if (releaseFreq > 1 && doy >= releaseFreq - 1)
#                                    {
#                                       for (int k = 1; k <= releaseFreq; k++)
#                                          {
#                                          float tmp = pControl->m_pResAllocation->Get(l, rowCount - k);
#                                          constraintValue += pControl->m_pResAllocation->Get(l, rowCount - k);
#                                          }
#                                       constraintValue = constraintValue / releaseFreq;
#                                    }
#                  #                  constraintValue = pControl->m_pResAllocation->IGet(0, l);   #Flow allocated from this control point
#                                 }
#                                 else
#                                    constraintValue = pControl->m_pResAllocation->IGet(0, l );   #Flow allocated from this control point
#                                 
#                                 actualRelease += constraintValue;    #constraint value will be negative if flow needs to be withheld, positive if flow needs to be augmented
#                                 
#                                 if ( constraintValue > 0.0 )         #Did this constraint affect release?  If so, save as active rule.
#                                    {
#                                    pRes->m_activeRule = pConstraint->m_constraintFileName; pRes->m_constraintValue = actualRelease;
#                                    }
#                                 break;
#                                 }
#                              }
#                           }
#                        }
#                     }  
#                  }
#                  break;
#
#            
#               }  # end of: switch( pConstraint->m_type )
#
#               /*
#            float minVolum = pRes->GetPoolVolumeFromElevation(m_inactive_elev);
#            //float targetPoolElevation = pRes->GetTargetElevationFromRuleCurve(doy);
#            float currentVolume = pRes->GetPoolVolumeFromElevation(currentPoolElevation);
#            float rc_outflowVolum = actualRelease*SEC_PER_DAY;//volume that would be drained (m3 over the day)
#         
#            
#            if (rc_outflowVolum > currentVolume - minVolum)      //In the inactive zone, water is not accessible for release from any of the gates.
#               {
#               actualRelease = (currentVolume - minVolum)/SEC_PER_DAY*0.5f;
#               CString resMsg;
#
#               resMsg.Format("Pool is only %8.1f m above inactive zone. Outflow set to drain %8.1fm above inactive zone.  RC outflow of %8.1f m3/s (from %s) would have resulted in %8.0f m3 of discharged water (over a day) but there is only %8.0f m3 above the inactive zone", pRes->m_elevation - pRes->m_inactive_elev, (pRes->m_elevation - pRes->m_inactive_elev) / 2, rc_outflowVolum / SEC_PER_DAY, pRes->m_activeRule, rc_outflowVolum, currentVolume - minVolum);
#                  pRes->m_activeRule = resMsg;
#              
#               }
#               */
#            if (actualRelease < 0.0f)
#               actualRelease = 0.0f;
#            pRes->m_zone = (int)zone;
#
#          // if (actualRelease < pRes->m_minOutflow)              // No release values less than the minimum
#          //     actualRelease = pRes->m_minOutflow;
#
#            if (pRes->m_elevation < pRes->m_inactive_elev)      //In the inactive zone, water is not accessible for release from any of the gates.
#               actualRelease = pRes->m_outflow/SEC_PER_DAY*0.5f;
#
#            outflow = actualRelease;
#            }//end of for ( int i=0; i < pZone->m_resConstraintArray.GetSize(); i++ )
#         }
#      }  // end of: else (not run of river gate flow
# 
#   //Code here to assign total outflow to powerplant, RO and spillway (including run of river projects) 
#   if (pRes->m_reservoirType == ResType_FloodControl || pRes->m_reservoirType == ResType_RiverRun )
#      AssignReservoirOutletFlows(pRes, outflow);
#
#   pRes->m_power = (pRes->m_powerFlow > 0.) ? CalculateHydropowerOutput(pRes) : 0.f; // MW
#
#   /* output activ rule to log window
#   if (true || pRes->m_id==5)
#      { // Dorena is reservoir 5
#      CString msg;
#      msg.Format("Flow: Day %i: Active Rule for Reservoir %s is %s", doy, (LPCTSTR)pRes->m_name, (LPCTSTR)pRes->m_activeRule);
#      Report::LogMsg(msg);
#      }
#*/
#   return outflow;
#   }






def AssignReservoirOutletFlows(name,outflow):
    #flow file has a condition on reservoir not being null...dk if we need that here

    #reset values to 0.0
    powerFlow = 0.0
    RO_flow = 0.0
    spillwayFlow = 0.0

    if outflow < name.maxPowerFlow: #this value is stored
        powerFlow = outflow
    else:
        powerFlow = name.maxPowerFlow
        excessFlow = outflow - name.maxPowerFlow
        if excessFlow <= name.maxRO_flow:
            RO_flow = excessFlow
            if RO_flow < name.minRO_flow: #why is this condition less than where as the previous are <=
                RO_flow = name.minRO_flow
                powerFlow = outflow - name.minRO_flow
        else:
            RO_flow = name.maxRO_flow
            excessFlow = RO_flow

            spillwayFlow = excessFlow

            if spillwayFlow < name.minSpillwayFlow:
                spillwayFlow = name.minSpillwayFlow
                RO_flow =- name.minSpillwayFlow - excessFlow
            if spillwayFlow > name.maxSpillwayflow:
                print('Flow: Maximum spillway volume exceed')

    #reset max outflows to gate maximums for next timestep
    maxPowerFlow = gateMaxPowerFlow #where are these gate values?
    maxROFlow = gateMaxRO_flow
    maxSpillwayFlow = gateMaxSpillwayFlow

    massbalancecheck = outflow - (powerFlow + RO_flow + spillwayFlow)
    #does this equal 0?

    return(powerFlow,RO_flow,spillwayFlow, massbalancecheck)

def CalculateHydropowerOutput(name,poolElevation,powerFlow):
    head = poolElevation - name.Tailwater_elev #should this input be poolElevation? or target?
    powerOut = (1000*powerFlow*9.81*head*0.9)/1000000  #assume a 0.9 turbine efficiency

    return powerOut

# list with inflow / elev / outflow / spill / RO / power
# Reservoirs network modeling
    # start a daily for loop
    # determne doy
    # set inflow /  elev / vol (from elev)
    
    # COT and DOR check which is farthest from targetPoolelev and start from that
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