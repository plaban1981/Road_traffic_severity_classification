# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:49:37 2022

@author: nayak

Prediction logic
"""

import joblib
from xgboost import XGBClassifier

model_path = r"C:\Users\nayak\Documents\TMLC_Course\Model\XGB.pkl"
model = joblib.load(model_path)

def predict(attribues):
    prediction = model.predict(attribues)
    return prediction