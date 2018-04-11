# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:11:23 2018

@author: Joy Hill
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

reservoirs = flow_model["reservoirs"]






    



#RESERVOIR rules
#in order of RES ID


#HILLS CREEK
HC_area_vol_curve = pd.read_csv('hills_creek_area_capacity.csv')
HC_rule_curve = pd.read_csv('hills_creek_rule_curve.csv')
HC_composite = pd.read_csv('HC_composite_rc.csv')
HC_RO = pd.read_csv('HC_RO_capacity.csv')
HC_rule_prio = pd.read_csv('hills_creek_rule_priorities.csv')
HC_buffer = pd.read_csv('HC_buffer.csv')
HC_spillway = pd.read_csv('HC_spillway_capacity.csv')

#LOOKOUT POINT
LO_area_vol_curve = pd.read_csv('lookout_point_area_capacity.csv')
LO_rule_curve = pd.read_csv('lookout_point_rule_curve.csv')
LO_composite = pd.read_csv('LO_composite_rc.csv')
LO_RO = pd.read_csv('LO_RO_capacity.csv')
LO_rule_prio = pd.read_csv('lookout_rule_priorities.csv')
LO_buffer = pd.read_csv('LO_buffer.csv')
LO_spillway = pd.read_csv('LO_spillway_capacity.csv')

#DEXTER re-regulating
DEX_area_vol_curve = pd.read_csv('dexter_area_capacity.csv')
DEX_composite = pd.read_csv('Dexter_composite_rc.csv')
DEX_RO = pd.read_csv('Dexter_RO_capacity.csv')
DEX_spillway = pd.read_csv('Dexter_Spillway_capacity.csv')


#FALL CREEK
FC_area_vol_curve = pd.read_csv('fall_creek_area_capacity.csv')
FC_rule_curve = pd.read_csv('fall_creek_rule_curve.csv')
FC_composite = pd.read_csv('FC_composite_rc.csv')
FC_RO = pd.read_csv('FC_RO_capacity.csv')
FC_rule_prio = pd.read_csv('fall_creek_rule_priorities.csv')
FC_buffer = pd.read_csv('FC_buffer.csv')
FC_spillway = pd.read_csv('FC_spillway_capacity.csv')

#DORENA
DO_area_vol_curve = pd.read_csv('dorena_area_capacity.csv')
DO_rule_curve = pd.read_csv('dorena_rule_curve.csv')
DO_composite = pd.read_csv('Dorena_composite_rc.csv')
DO_RO = pd.read_csv('Dorena_RO_capacity.csv')
DO_rule_prio = pd.read_csv('dorena_rule_priorities.csv')
DO_buffer = pd.read_csv('Dorena_buffer.csv')
DO_spillway = pd.read_csv('Dorena_spillway_capacity.csv')

#COTTAGE GROVE
CG_area_vol_curve = pd.read_csv('cottage_grove_area_capacity.csv')
CG_rule_curve = pd.read_csv('cottage_grove_rule_curve.csv')
CG_composite = pd.read_csv('CG_composite_rc.csv')
CG_RO = pd.read_csv('CG_RO_capacity.csv')
CG_rule_prio = pd.read_csv('cottage_grove_rule_priorities.csv')
CG_buffer = pd.read_csv('CG_buffer.csv')
CG_spillway = pd.read_csv('CG_spillway_capacity.csv')

#FERN RIDGE
FR_area_vol_curve = pd.read_csv('fern_ridge_area_capacity.csv')
FR_rule_curve = pd.read_csv('fern_ridge_rule_curve.csv')
FR_composite = pd.read_csv('FR_composite_rc.csv')
FR_RO = pd.read_csv('FR_RO_capacity.csv')
FR_rule_prio = pd.read_csv('fern_ridge_rule_priorities.csv')
FR_buffer = pd.read_csv('FR_buffer.csv')
FR_spillway = pd.read_csv('FR_spillway_capacity.csv')

#COUGAR
COU_area_vol_curve = pd.read_csv('cougar_area_capacity.csv')
COU_rule_curve = pd.read_csv('cougar_rule_curve.csv')
COU_composite = pd.read_csv('Cougar_composite_rc.csv')
COU_RO = pd.read_csv('Cougar_RO_capacity.csv')
COU_rule_prio = pd.read_csv('cougar_rule_priorities.csv')
COU_buffer = pd.read_csv('Cougar_buffer.csv')
COU_spillway = pd.read_csv('Cougar_spillway_capacity.csv')

#BLUE RIVER
BR_area_vol_curve = pd.read_csv('blue_river_area_capacity.csv')
BR_rule_curve = pd.read_csv('blue_river_rule_curve.csv')
BR_composite = pd.read_csv('BR_composite_rc.csv')
BR_RO = pd.read_csv('BR_RO_capacity.csv')
BR_rule_prio = pd.read_csv('blue_river_rule_priorities.csv')
BR_buffer = pd.read_csv('BR_buffer.csv')
BR_spillway = pd.read_csv('BR_spillway_capacity.csv')

#GREEN PETER
GP_area_vol_curve = pd.read_csv('green_peter_area_capacity.csv')
GP_rule_curve = pd.read_csv('green_peter_rule_curve.csv')
GP_composite = pd.read_csv('GP_composite_rc.csv')
GP_RO = pd.read_csv('GP_RO_capacity.csv')
GP_rule_prio = pd.read_csv('green_peter_rule_priorities.csv')
GP_buffer = pd.read_csv('GP_buffer.csv')
GP_spillway = pd.read_csv('GP_spillway_capacity.csv')

#FOSTER
FS_area_vol_curve = pd.read_csv('foster_area_capacity.csv')
FS_rule_curve = pd.read_csv('foster_rule_curve.csv')
FS_composite = pd.read_csv('Foster_composite_rc.csv')
FS_RO = pd.read_csv('Foster_RO_capacity.csv')
FS_rule_prio = pd.read_csv('foster_rule_priorities.csv')
FS_buffer = pd.read_csv('Foster_buffer.csv')
FS_spillway = pd.read_csv('Foster_spillway_capacity.csv')

#DETROIT
DT_area_vol_curve = pd.read_csv('detroit_area_capacity.csv')
DT_rule_curve = pd.read_csv('detroit_rule_curve.csv')
DT_composite = pd.read_csv('Detroit_composite_rc.csv')
DT_RO = pd.read_csv('Detroit_RO_capacity.csv')
DT_rule_prio = pd.read_csv('detroit_rule_priorities.csv')
DT_buffer = pd.read_csv('Detroit_buffer.csv')
DT_spillway = pd.read_csv('Detroit_Spillway_capacity.csv')

#BIG CLIFF re-regulating
BC_area_vol_curve = pd.read_csv('Big_Cliff_area_capacity.csv')
BC_composite = pd.read_csv('BC_composite_rc.csv')
BC_RO = pd.read_csv('BC_RO_capacity.csv')
BC_spillway = pd.read_csv('BC_spillway_capacity.csv')


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














    
