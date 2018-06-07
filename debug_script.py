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

runfile('C:/Users/sdenaro/OneDrive - University of North Carolina at Chapel Hill/UNC_2017/PNW/INFEWSgroup_Willamette/Willamette_model/Williamette/Williamette_model.py', wdir='C:/Users/sdenaro/OneDrive - University of North Carolina at Chapel Hill/UNC_2017/PNW/INFEWSgroup_Willamette/Willamette_model/Williamette')
runfile('C:/Users/sdenaro/OneDrive - University of North Carolina at Chapel Hill/UNC_2017/PNW/INFEWSgroup_Willamette/Willamette_model/Williamette/Williamette_outer.py', wdir='C:/Users/sdenaro/OneDrive - University of North Carolina at Chapel Hill/UNC_2017/PNW/INFEWSgroup_Willamette/Willamette_model/Williamette')

#%% Allocate and initialize

InitwaterYear = 1.2
waterYear = InitwaterYear
GPR = RES[9]
name = GPR

CP_list = CP
cp_discharge = cp_discharge_all
#%%
 #GetResOutflow(name, volume, inflow, lag_outflow, doy, waterYear, CP_list, cp_discharge):
for t in range (0, 364):
    doy = DatetoDayOfYear(str(dates[t])[:10],'%Y-%m-%d')

    volume = volumes_all[t,9]
    inflow = GPR5A.iloc[t,1]
    lag_outflow = outflows_all[t-1,9]
    
    
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
        
      #print('for t=', t, 'target elev is', targetPoolElevation,'and current elev is',currentPoolElevation)
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
      print('for t=', t, 'zone is', zone)   
      
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
          #print('The constraint array is',constraint_array[i])   
          if constraint_array[i].startswith('Max_'):  #case RCT_MAX  maximum
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(yvalue, xvalue))   
                 #constraintValue =interp_table(xvalue, yvalue)  
             else:             #//If not, just use xvalue
                constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
             if actualRelease >= constraintValue:
                actualRelease = constraintValue;
                print('The constraint is',constraint_array[i], 'value is',constraintValue)   

          elif constraint_array[i].startswith('Min_'):  # case RCT_MIN:  //minimum
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(yvalue, xvalue))                
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
             if actualRelease <= constraintValue:
                actualRelease = constraintValue;
                print('The constraint is',constraint_array[i], 'value is',constraintValue)   



          elif constraint_array[i].startswith('MaxI_'):     #case RCT_INCREASINGRATE:  //Increasing Rate
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(yvalue, xvalue)) 
                 constraintValue = constraintValue*24   #Covert hourly to daily                  
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                 constraintValue = constraintValue*24   #Covert hourly to daily
             if actualRelease >= lag_outflow  + constraintValue:  #Is planned release more than current release + contstraint? 
                actualRelease = lag_outflow  + constraintValue
                print('The constraint is',constraint_array[i], 'value is',constraintValue) 

                #If so, planned release can be no more than current release + constraint.
                 

          elif constraint_array[i].startswith('MaxD_'):    #case RCT_DECREASINGRATE:  //Decreasing Rate
             if yvalue != [] :    # Does the constraint depend on two values?  If so, use both xvalue and yvalue
                 cols=constraintRules.iloc[0,1::]
                 rows=constraintRules.iloc[1::,0]
                 vals=constraintRules.iloc[1::,1::]
                 interp_table = interp2d(cols, rows, vals, kind='linear')
                 constraintValue =float(interp_table(yvalue, xvalue)) 
                 constraintValue = constraintValue*24   #Covert hourly to daily                  
             else:             #//If not, just use xvalue
                 constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                 constraintValue = constraintValue*24   #Covert hourly to daily
             if actualRelease <= lag_outflow  - constraintValue:  #Is planned release less than current release - contstraint? 
                actualRelease = lag_outflow  - constraintValue  #If so, planned release can be no less than current release - constraint.
                print('The constraint is',constraint_array[i], 'value is',constraintValue)  


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
                              constraintValue =float(interp_table(yvalue, xvalue)) 
                          else:             #//If not, just use xvalue
                              constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
                              #Compare to current discharge and allocate flow increases or decreases
                              #Currently allocated evenly......need to update based on storage balance curves in ResSIM 
                          if constraint_array[i].startswith('cp_Max'):  #maximum    
                              if cp_discharge[t-1,cp_id-1] > constraintValue:   #Are we above the maximum flow?   
                                  resallocation = constraintValue - cp_discharge[cp_id]/len(cp_name.influencedReservoirs) #Allocate decrease in releases (should be negative) over "controlled" reservoirs if maximum, evenly for now
                              else:  
                                  resallocation = 0
                          elif constraint_array[i].startswith('cp_Min'):  #minimum
                              if cp_discharge[t-1,cp_id-1] < constraintValue:   #Are we below the minimum flow?  
                                 resallocation = constraintValue - cp_discharge[cp_id]/len(cp_name.influencedReservoirs) 
                              else:  
                                  resallocation = 0
                          actualRelease += resallocation #add/subract cp allocation
                          print('The constraint is',constraint_array[i], 'resallocation is',resallocation)               
          #GATE SPECIFIC RULES:   
          elif constraint_array[i].startswith('Pow_Max'): # case RCT_POWERPLANT:  //maximum Power plant rule  Assign m_maxPowerFlow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxPowerFlow = constraintValue  #Just for this timestep.  name.MaxPowerFlow is the physical limitation for the reservoir.
          elif constraint_array[i].startswith('Pow_Min'): # case RCT_POWERPLANT:  //minimum Power plant rule  Assign m_minPowerFlow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minPowerFlow = constraintValue 
               print('The constraint is',constraint_array[i], 'value is',constraintValue)  

            
          elif constraint_array[i].startswith('RO_Max'): #case RCT_REGULATINGOUTLET:  Max Regulating outlet rule, Assign m_maxRO_Flow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxRO_Flow  = constraintValue  
          elif constraint_array[i].startswith('RO_Min'): #Min Regulating outlet rule, Assign m_maxRO_Flow attribute.
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minRO_Flow  = constraintValue 
               print('The constraint is',constraint_array[i], 'value is',constraintValue)  


          elif constraint_array[i].startswith('Spill_Max'): #  case RCT_SPILLWAY:   //Max Spillway rule
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.maxSpillwayFlow  = constraintValue  
          elif constraint_array[i].startswith('Spill_Min'): #Min Spillway rule
               constraintValue = np.interp(xvalue,constraintRules.iloc[:,0],constraintRules.iloc[:,1])
               name.minSpillwayFlow  = constraintValue 
               print('The constraint is',constraint_array[i], 'value is',constraintValue)  
                
               
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
    #print('for t=', t, 'release is', outflow)

    #return outflow

    