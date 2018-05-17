# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:07:48 2018

@author: Joy Hill
"""

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
from xml.dom import minidom
import xml.etree.ElementTree as ET
import xmltodict as xmld
from collections import OrderedDict
import datetime as dt
import os



####################

def DatetoDayOfYear(val, fmt):
#val = '2012/11/07'
#fmt = '%Y/%m/%d'
    date_d = dt.datetime.strptime(val,fmt)
    tt = date_d.timetuple()
    doy = tt.tm_yday

    return doy

def UpdateReservoirWaterYear(doy,t, volumes_all):
    waterYear=np.nan
    M3_PER_ACREFT = 1233.4
    resVolumeBasin = np.sum(volumes_all[t-1,:])
    resVolumeBasin = resVolumeBasin*M3_PER_ACREFT*1000000
    if resVolumeBasin > float(1.48):
        waterYear = float(1.48) #Abundant
    elif resVolumeBasin < float(1.48) and resVolumeBasin >  float(1.2):
        waterYear = float(1.2) #Adequate
    elif resVolumeBasin < float(1.2) and resVolumeBasin > float(0.9):
        waterYear = float(0.9) #Insufficient
    elif resVolumeBasin < float(0.9):
        waterYear = 0 #Deficit
            
    return waterYear
        
        


def GetPoolElevationFromVolume(volume,name):
    if name.AreaVolCurve is None:
        return 0
    else:
        poolElevation = np.interp(volume,name.AreaVolCurve['Storage_m3'],name.AreaVolCurve['Elevation_m'])

        return poolElevation #returns pool elevation (m)


def GetPoolVolumeFromElevation(pool_elev,name):
    if name.AreaVolCurve is None:
        return 0
    else:
        poolVolume = np.interp(pool_elev,name.AreaVolCurve['Elevation_m'],name.AreaVolCurve['Storage_m3'])

        return poolVolume #returns pool vol(m^3)


def GetBufferZoneElevation(doy,name):
    if name.Buffer is None:
        return 0
    else:
        bufferZoneElevation = np.interp(doy,name.Buffer['Day'],name.Buffer['Pool_elevation_m'])

        return bufferZoneElevation #returns what the buffer zone elevation level is for this time of year (in m)


def GetTargetElevationFromRuleCurve(doy,name): #target_table is same as rule_curve
    if name.RuleCurve is None:
        return 0
    else:
        target = np.interp(doy,name.RuleCurve['Day'],name.RuleCurve['Cons_Pool_elev_m'])

        return target #target pool elevation in m

def UpdateMaxGateOutflows(name,poolElevation): 
    name.maxPowerFlow=name.GateMaxPowerFlow    #does not depend on elevation but can change due to constraints
    name.maxRO_Flow = np.interp(poolElevation,name.RO['pool_elev_m'],name.RO['release_cap_cms'])
    name.maxSpillwayFlow = np.interp(poolElevation,name.Spillway['pool_elev_m'],name.Spillway['release_cap_cms'])
        
    return (name.maxPowerFlow, name.maxRO_Flow, name.maxSpillwayFlow)


def AssignReservoirOutletFlows(name,outflow):
    #flow file has a condition on reservoir not being null...dk if we need that here
    #outflow = outflow * 86400  # convert to daily volume    m3 per day 
    #initialize values to 0.0
    powerFlow = 0.0
    RO_flow = 0.0
    spillwayFlow = 0.0

    if outflow < name.maxPowerFlow: #this value is stored
        powerFlow = outflow
    else:
        powerFlow = name.maxPowerFlow
        excessFlow = outflow - name.maxPowerFlow
        if excessFlow <= name.maxRO_Flow:
            RO_flow = excessFlow
            if RO_flow < name.minRO_Flow: #why is this condition less than where as the previous are <=
                RO_flow = name.minRO_Flow
                powerFlow = outflow - name.minRO_Flow
        else:
            RO_flow = name.maxRO_Flow
            excessFlow = RO_flow

            spillwayFlow = excessFlow

            if spillwayFlow < name.minSpillwayFlow:
                spillwayFlow = name.minSpillwayFlow
                RO_flow =- name.minSpillwayFlow - excessFlow
            if spillwayFlow > name.maxSpillwayFlow:
                print('Maximum spillway volume exceed')

            
    massbalancecheck = outflow - (powerFlow + RO_flow + spillwayFlow)
    #does this equal 0?
    if massbalancecheck != 0:
        print ("Mass balance didn't close, massbalancecheck = ", massbalancecheck )

    return(powerFlow,RO_flow,spillwayFlow)
    
    
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
BLU5A = pd.read_excel('Data/BLU5A_daily.xls',skiprows=27942,skip_footer =1004)*cfs_to_cms
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

outflows_2005_all = np.stack((BLU5H[:,1],BCL5H[:,1],CGR5H[:,1],DET5H[:,1],DEX5H[:,1],DOR5H[:,1],FAL5H[:,1],FOS5H[:,1],FRN5H[:,1],GPR5H[:,1],HCR5H[:,1],LOP5H[:,1],COT5H[:,1],FOS5H[:,1]),axis=1)


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
T = 364 # Set the simulation horizon

n_res=13
n_HPres=8
n_cp = 11


#=======
# allocate output 
outflows_all = np.full((T+2,n_res),np.nan) #we can fill these in later, or make them empty and 'append' the values
hydropower_all = np.full((T+2,n_HPres), np.nan)
volumes_all = np.full((T+2,n_res),np.nan)
elevations_all = np.full((T+2,n_res),np.nan)
cp_discharge_all = np.full((T+2,(n_cp-1)),np.nan)

#initialize values
for  i in range(0,n_res):
    outflows_all[0:3,i] = outflows_2005_all[0:3,i] #remember to stack outflows historical values
    volumes_all[0:3,i] = np.tile(RES[i].InitVol,(3)) #TO BE CHANGED!
    elevations_all[0:3,i]=GetPoolElevationFromVolume(volumes_all[0:3,i],RES[i])


for  i in range(0,n_cp-1):
     cp_discharge_all[0,i] = cp_discharge_2005_all[0,i]

#define an outer fnt here that takes date, name, vol as inputs?

InitwaterYear = 1.2
waterYear = InitwaterYear
#%%
#%%
CGR = RES[7]

name = CGR

t=1
doy = DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')

volume = volumes_all[t,7]
inflow = CGR5A.iloc[t,1]
lag_outflow = outflows_all[t-1,7]

CP_list = CP

cp_discharge = cp_discharge_all
#%%
 #GetResOutflow(name, volume, inflow, lag_outflow, doy, waterYear, CP_list, cp_discharge):
    currentPoolElevation = GetPoolElevationFromVolume(volume,name)    
    if name.Restype!='Storage_flood': #if it produces hydropower
        #reset gate specific flows
        [name.maxPowerFlow, name.maxRO_Flow, name.maxSpillwayFlow]=UpdateMaxGateOutflows( name, currentPoolElevation )
        #Reset min outflows to 0 for next timestep (max outflows are reset by the function UpdateMaxGateOutflows)
        name.minPowerFlow=0
        name.minRO_Flow =0
        name.minSpillwayFlow=0 
    
    if name.Restype=='RunOfRiver':
      outflow = inflow
    else:
      targetPoolElevation = GetTargetElevationFromRuleCurve( doy, name )
      targetPoolVolume    = GetPoolVolumeFromElevation(targetPoolElevation, name)
      currentVolume = GetPoolVolumeFromElevation(currentPoolElevation, name)
      bufferZoneElevation = GetBufferZoneElevation( doy, name )

      if currentVolume > name.maxVolume:
         currentVolume = name.maxVolume;   #Don't allow res volumes greater than max volume.This code may be removed once hydro model is calibrated.
         currentPoolElevation = GetPoolElevationFromVolume(currentVolume, name);  #Adjust pool elevation
         print ("Reservoir volume at ", name.name , " on day of year ", doy, " exceeds maximum. Volume set to maxVolume. Mass Balance not closed.")
        

      desiredRelease = (currentVolume - targetPoolVolume)/86400     #This would bring pool elevation back to the rule curve in one timestep.  Converted from m3/day to m3/s  (cms).
                                                                          #This would be ideal if possible within the given constraints.
      if currentVolume < targetPoolVolume: #if we want to fill the reservoir
         desiredRelease=name.minOutflow

      actualRelease = desiredRelease   #Before any constraints are applied
      
      # ASSIGN ZONE:  zone 0 = top of dam, zone 1 = flood control high, zone 2 = conservation operations, zone 3 = buffer, zone 4 = alternate flood control 1, zone 5 = alternate flood control 2.  
      zone=[]
      if currentPoolElevation > name.Td_elev:
         print ("Reservoir elevation at ", name.name, "on day of year ", doy," exceeds dam top elevation.")
         currentPoolElevation = name.Td_elev - 0.1    #Set just below top elev to keep from exceeding values in lookup tables
         zone = 0    
      elif currentPoolElevation > name.Fc1_elev:
         zone = 0
      elif currentPoolElevation > targetPoolElevation:
         if name.Fc2_elev!=[] and currentPoolElevation <= name.Fc2_elev:
            zone = 4
         elif name.Fc3_elev!=[] and currentPoolElevation > name.Fc3_elev:
            zone = 5
         else:
            zone = 1
      elif currentPoolElevation <= targetPoolElevation:
         if currentPoolElevation <= bufferZoneElevation:   #in the buffer zone (HERE WE REMOVED THE PART THAT COUNTED THE DAYS IN THE BUFFER ZONE)
            zone = 3     
         else:                                           #in the conservation zone
            zone = 2
      else:
         print ("*** GetResOutflow(): We should never get here. doy = ", doy ,"reservoir = ", name.name)
      print('zone is', zone)   
      
        # Once we know what zone we are in, we can access the array of appropriate constraints for the particular reservoir and zone.       

      constraint_array=name.RulePriorityTable.iloc[:,zone]
      constraint_array=constraint_array[(constraint_array !='Missing')] #eliminate 'Missing' rows
      #Loop through constraints and modify actual release.  Apply the flood control rules in order here.
      for i in range (0, (len(constraint_array)-1)):
          #open the csv file and get first column label and appropriate data to use for lookup
          constraintRules = pd.read_csv(os.path.join(name.ruleDir, constraint_array[i])) 
          xlabel = list(constraintRules)[0]   
          xvalue = []
          yvalue = []
          if xlabel=="Date":                           # Date based rule?  xvalue = current date.
             xvalue = doy
          elif xlabel=="Release_cms":                  # Release based rule?  xvalue = release last timestep
               xvalue = lag_outflow       
          elif xlabel=="Pool_elev_m" :                 # Pool elevation based rule?  xvalue = pool elevation (meters)
               xvalue = currentPoolElevation
          elif xlabel=="Inflow_cms":                    # Inflow based rule?   xvalue = inflow to reservoir
               xvalue = inflow            
          elif xlabel=="Outflow_lagged_24h":            #24h lagged outflow based rule?   xvalue = outflow from reservoir at last timestep
               xvalue = lag_outflow               #placeholder (assumes that timestep=24 hours)
          elif xlabel=="Date_pool_elev_m":             # Lookup based on two values...date and pool elevation.  x value is date.  y value is pool elevation
               xvalue = doy
               yvalue = currentPoolElevation
          elif xlabel=="Date_Water_year_type":          #Lookup based on two values...date and wateryeartype (storage in 13 USACE reservoirs on May 20th).
               xvalue = doy
               yvalue = waterYear
          elif xlabel == "Date_release_cms": 
               xvalue = doy
               yvalue = lag_outflow 
          else:                                            #Unrecognized xvalue for constraint lookup table
             print ("Unrecognized x value for reservoir constraint lookup label = ", xlabel) 
          print('The constraint array of i is',constraint_array[i])   
          if constraint_array[i].startswith('Max_'):  #case RCT_MAX  maximum
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(xvalue, yvalue))   
                 #constraintValue =interp_table(xvalue, yvalue)  
             else:             #//If not, just use xvalue
                constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
             if actualRelease >= constraintValue:
                actualRelease = constraintValue;
             print('The constraint value is',constraintValue)   

          elif constraint_array[i].startswith('Min_'):  # case RCT_MIN:  //minimum
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(xvalue, yvalue))                
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
             if actualRelease <= constraintValue:
                actualRelease = constraintValue;
             print('The constraint value is',constraintValue)   



          elif constraint_array[i].startswith('MaxI_'):     #case RCT_INCREASINGRATE:  //Increasing Rate
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(xvalue, yvalue))
                 constraintValue = constraintValue*24   #Covert hourly to daily                  
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                 constraintValue = constraintValue*24   #Covert hourly to daily
             if actualRelease >= lag_outflow  + constraintValue:  #Is planned release more than current release + contstraint? 
                actualRelease = lag_outflow  + constraintValue
             print('The constraint value is',constraintValue)   

                #If so, planned release can be no more than current release + constraint.
                 

          elif constraint_array[i].startswith('MaxD_'):    #case RCT_DECREASINGRATE:  //Decreasing Rate
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(xvalue, yvalue))
                 constraintValue = constraintValue*24   #Covert hourly to daily                  
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                 constraintValue = constraintValue*24   #Covert hourly to daily
             if actualRelease <= lag_outflow  - constraintValue:  #Is planned release less than current release - contstraint? 
                actualRelease = lag_outflow  - constraintValue  #If so, planned release can be no less than current release - constraint.
             print('The constraint value is',constraintValue)   


          elif constraint_array[i].startswith('cp_'):  #case RCT_CONTROLPOINT:  #Downstream control point  
              #Determine which control point this is.....use COMID to identify
              for j in range(len(CP_list)):
                  if CP_list[j].COMID in constraint_array[i]:
                      cp_name = CP_list[j]
                      cp_id = CP_list[j].ID        
                      if name in cp_name.influencedReservoirs:  #Make sure that the reservoir is on the influenced reservoir list
                          resallocation=np.nan 
                          if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                              cols=constraintRules.iloc[0,1::]
                              rows=constraintRules.iloc[1::,0]
                              vals=constraintRules.iloc[1::,1::]
                              interp_table = interp2d(cols, rows, vals, kind='linear')
                              constraintValue =float(interp_table(xvalue, yvalue))
                          else:             #//If not, just use xvalue
                              constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                              #Compare to current discharge and allocate flow increases or decreases
                              #Currently allocated evenly......need to update based on storage balance curves in ResSIM 
                          if constraint_array[i].startswith('cp_Max'):  #maximum    
                              if cp_discharge[cp_id] > constraintValue:   #Are we above the maximum flow?   
                                  resallocation = constraintValue - cp_discharge[cp_id]/len(cp_name.influencedReservoirs) #Allocate decrease in releases (should be negative) over "controlled" reservoirs if maximum, evenly for now
                              else:  
                                  resallocation = 0
                          elif constraint_array[i].startswith('cp_Min'):  #minimum
                              if cp_discharge[cp_id] < constraintValue:   #Are we below the minimum flow?  
                                 resallocation = constraintValue - cp_discharge[cp_id]/len(cp_name.influencedReservoirs) 
                              else:  
                                  resallocation = 0
                          actualRelease += resallocation #add/subract cp allocation
                          print('The resallocation is',resallocation)               
          #GATE SPECIFIC RULES:   
          elif constraint_array[i].startswith('Pow_Max'): # case RCT_POWERPLANT:  //maximum Power plant rule  Assign m_maxPowerFlow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxPowerFlow = constraintValue  #Just for this timestep.  name.MaxPowerFlow is the physical limitation for the reservoir.
          elif constraint_array[i].startswith('Pow_Min'): # case RCT_POWERPLANT:  //minimum Power plant rule  Assign m_minPowerFlow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minPowerFlow = constraintValue 
           
            
          elif constraint_array[i].startswith('RO_Max'): #case RCT_REGULATINGOUTLET:  Max Regulating outlet rule, Assign m_maxRO_Flow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxRO_Flow  = constraintValue  
          elif constraint_array[i].startswith('RO_Min'): #Min Regulating outlet rule, Assign m_maxRO_Flow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minRO_Flow  = constraintValue 


          elif constraint_array[i].startswith('Spill_Max'): #  case RCT_SPILLWAY:   //Max Spillway rule
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxSpillwayFlow  = constraintValue  
          elif constraint_array[i].startswith('Spill_Min'): #Min Spillway rule
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minSpillwayFlow  = constraintValue 
                
               
          if actualRelease < 0:
             actualRelease = 0
          if actualRelease < name.minOutflow:         # No release values less than the minimum
             actualRelease = name.minOutflow
          if currentPoolElevation < name.inactive_elev:     #In the inactive zone, water is not accessible for release from any of the gates.
             actualRelease = 0 #lag_outflow *0.5

          outflow = actualRelease
    if name.Restype!='Storage_flood':
        [powerFlow,RO_flow,spillwayFlow]=AssignReservoirOutletFlows(name,outflow)
    else:
        [powerFlow,RO_flow,spillwayFlow]=[np.nan, np.nan, np.nan]
    return outflow, powerFlow,RO_flow,spillwayFlow
    #return outflow

    