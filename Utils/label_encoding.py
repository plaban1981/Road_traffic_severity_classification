# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:57:28 2022

@author: nayak

features =['Time', 'Day_of_week', 'Age_band_of_driver', 'Service_year_of_vehicle',
       'Defect_of_vehicle', 'Area_accident_occured', 'Types_of_Junction',
       'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Sex_of_casualty', 'Casualty_severity']

"""

import pandas as pd
import numpy as np


def label_enc(X_test):
    map_time = {'AM':0,'PM':1}
    map_day = {'Sunday':0,'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6 }
    map_age_driver = {'Under 18':0,'18-30':1,'31-50':2,'Over 51':3,'Unknown':4}
    map_ser_year = {'1-2yr':1, '2-5yrs':2, '5-10yrs':3, 'Above 10yr':4, 'Below 1yr':0, 'Unknown':5 }
    map_ser_def = {'5':1, '7':2, 'Missing':3, 'No defect':0}
    map_area_acc = {'Church areas': 0,
                    'Hospital areas': 1,
                    'Industrial areas': 2,
                    'Market areas': 3,
                    'Office areas': 4,
                    'Other': 5,
                    'Outside rural areas': 6,
                    'Recreational areas': 7,
                    'Residential areas': 8,
                    'Rural village areas': 9,
                    'Rural village areasOffice areas': 10,
                    'School areas': 11,
                    'Unknown': 12}
    map_junction = {'Crossing': 0,
                    'No junction': 1,
                    'O Shape': 2,
                    'Other': 3,
                    'T Shape': 4,
                    'Unknown': 5,
                    'X Shape': 6,
                    'Y Shape': 7}
    map_road_cond =  {'Dry': 0, 'Flood over 3cm. deep': 1, 'Snow': 2, 'Wet or damp': 3}
    map_light_cond =  {'Darkness - lights lit': 0,
                       'Darkness - lights unlit': 1,
                       'Darkness - no lighting': 2,
                       'Daylight': 3}
    map_weather = {'Cloudy': 0,
                   'Fog or mist': 1,
                   'Normal': 2,
                   'Other': 3,
                   'Raining': 4,
                   'Raining and Windy': 5,
                   'Snow': 6,
                   'Unknown': 7,
                   'Windy': 8}
    
    map_vehicle = {'Entering a junction': 0,
                   'Getting off': 1,
                   'Going straight': 2,
                   'Moving Backward': 3,
                   'Other': 4,
                   'Overtaking': 5,
                   'Parked': 6,
                   'Reversing': 7,
                   'Stopping': 8,
                   'Turnover': 9,
                   'U-Turn': 10,
                   'Unknown': 11,
                   'Waiting to go': 12}
    map_sex_cas = {'Female': 0, 'Male': 1, 'na': 2}
    map_acc_sev = {'Fatal injury': 0, 'Serious Injury': 1, 'Slight Injury': 2}
    X_test['Casualty_severity'] = X_test['Casualty_severity'].astype('int64')
    X_test['Number_of_vehicles_involved'] = X_test['Number_of_vehicles_involved'].astype('int64')
    X_test['Number_of_casualties'] = X_test['Number_of_casualties'].astype('int64')
    X_test['Time'] = X_test['Time'].map(map_time)
    X_test['Day_of_week'] = X_test['Day_of_week'].map(map_day)
    X_test['Age_band_of_driver'] = X_test['Age_band_of_driver'].map(map_age_driver)
    X_test['Service_year_of_vehicle'] = X_test['Service_year_of_vehicle'].map(map_ser_year)
    X_test['Defect_of_vehicle'] = X_test['Defect_of_vehicle'].map(map_ser_def)
    X_test['Area_accident_occured'] = X_test['Area_accident_occured'].map(map_area_acc)
    X_test['Types_of_Junction'] = X_test['Types_of_Junction'].map(map_junction)
    X_test['Road_surface_conditions'] = X_test['Road_surface_conditions'].map(map_road_cond)
    X_test['Light_conditions'] = X_test['Light_conditions'].map(map_light_cond)
    X_test['Weather_conditions'] = X_test['Weather_conditions'].map(map_weather)
    X_test['Vehicle_movement'] = X_test['Vehicle_movement'].map(map_vehicle)
    X_test['Sex_of_casualty'] = X_test['Sex_of_casualty'].map(map_sex_cas)
    return X_test
    
    