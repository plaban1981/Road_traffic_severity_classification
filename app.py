# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 16:06:46 2022

@author: nayak
"""

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from Utils.model import predict
from Utils.label_encoding import label_enc
from xgboost import XGBClassifier

#image_path = "acc_sev.jpg"
#image = Image.open(image_path)
features = ['Time', 'Day_of_week', 'Age_band_of_driver', 'Service_year_of_vehicle',
       'Defect_of_vehicle', 'Area_accident_occured', 'Types_of_Junction',
       'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
       'Number_of_vehicles_involved', 'Number_of_casualties',
       'Vehicle_movement', 'Sex_of_casualty', 'Casualty_severity']



st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="centered")

#st.image(image, caption='Road Safety Management')

# page header
st.title(f"Accident Severity Prediction App")
 
with st.form("Prediction_form"):
    # form header
    st.header("Enter the feature specifications contributing to the accident:")
    # input elements
    Time =  st.selectbox('Select Time : ',('AM', 'PM'))
    Day_of_week = st.selectbox('Select Day of the week : ',('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'))
    Age_band_of_driver = st.selectbox('Select Age band of the Driver : ',('Under 18','18-30','31-50','Over 51','Unknown'))
    Service_year_of_vehicle = st.selectbox('Select Service Year of the Vehicle : ',('1-2yr','2-5yrs','5-10yrs','Above 10yr','Below 1yr','Unknown'))
    Defect_of_vehicle = st.selectbox('Select Defect of Vehicle : ',('5','7', 'Missing', 'No defect'))
    Area_accident_occured = st.selectbox('Select Area of Accident occurred : ',('Church areas','Hospital areas','Industrial areas','Market areas','Office areas','Other','Outside rural areas','Recreational areas','Residential areas','Rural village areas','Rural village areasOffice areas','School areas','Unknown'))
    Types_of_Junction = st.selectbox('Select types of Junction : ',('Crossing','No junction','O Shape','Other','T Shape','Unknown','X Shape','Y Shape'))
    Road_surface_conditions = st.selectbox('Select Road Surface Conditions : ',('Dry','Flood over 3cm. deep', 'Snow','Wet or damp'))
    Light_conditions = st.selectbox('Select Light Conditions : ',('Darkness - lights lit','Darkness - lights unlit','Darkness - no lighting','Daylight'))
    Weather_conditions = st.selectbox('Select Weather Conditions : ',('Cloudy','Fog or mist','Normal','Other','Raining','Raining and Windy','Snow','Unknown','Windy'))
    Number_of_vehicles_involved = st.selectbox('Select Number of Vehichles Involved : ',(1,2,3,4,6,7))
    Number_of_casualties = st.selectbox('Select Number of Casualities : ',(1,2,3,4,5,6,7,8))
    Vehicle_movement = st.selectbox('Select Vehicle Movement : ',('Entering a junction','Getting off','Going straight','Moving Backward','Other','Overtaking','Parked','Reversing','Stopping','Turnover','U-Turn','Unknown','Waiting to go'))
    Sex_of_casualty = st.selectbox('Select Sex of the Casuality : ',('Female','Male','na'))
    Casualty_severity = st.selectbox('Select Casuality Severity : ',(0,1,2,3))
    # 
    submit = st.form_submit_button("Predict")
    #
    if submit :
        input_dict = {'Time':Time, 
                      'Day_of_week':Day_of_week, 
                      'Age_band_of_driver':Age_band_of_driver, 
                      'Service_year_of_vehicle' : Service_year_of_vehicle,
                      'Defect_of_vehicle':Defect_of_vehicle, 
                      'Area_accident_occured':Area_accident_occured, 
                      'Types_of_Junction' : Types_of_Junction,
                      'Road_surface_conditions':Road_surface_conditions, 
                      'Light_conditions':Light_conditions, 
                      'Weather_conditions':Weather_conditions,
                      'Number_of_vehicles_involved':Number_of_vehicles_involved, 
                      'Number_of_casualties':Number_of_casualties,
                      'Vehicle_movement' :Vehicle_movement, 
                      'Sex_of_casualty':Sex_of_casualty, 
                      'Casualty_severity':Casualty_severity}
        df = pd.DataFrame(input_dict, index=[1])
        df_enc = label_enc(df)
        #
        prediction = predict(df_enc)[0]
        map_acc_sev = { 0 :'Fatal injury', 1: 'Serious Injury', 2:'Slight Injury'}
        value = map_acc_sev[prediction]
        # output header
        st.header("Predictions")
        # output results
        st.success(f"Accident Severity : {value}")
        
