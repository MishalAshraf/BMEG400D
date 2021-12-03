#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import pickle

def get_sepsis_score(data, model):
    headers = [
        "ICULOS",
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "DBP",
        "Resp",
        "FiO2",
        "pH",
        "PaCO2",
        "BUN",
        "Calcium",
        "Creatinine",
        "Glucose",
        "Magnesium",
        "Potassium",
        "Hct",
        "Hgb",
        "WBC",
        "Platelets",
        "Age",
        "Sex"]

    # Convert to last row only to pandas data frame
    X = pd.DataFrame(data, columns=headers)

    # Forward fill
    X = X.ffill()

    # Fill rest with 0
    X = X.fillna(0)

    # Take positive sepsis label prediction only
    score = model.predict_proba(X)[-1][1]

    # Set default threshold
    label = score > 0.023

    return score, label

def load_sepsis_model():
    # Comment out on the one that you want to run

    # Logistic Regression
    return pickle.load(open("./models/logreg_pipeline.pkl", "rb"))

    # # SVC
    # return pickle.load(open("./models/svc_pipeline.pkl", "rb"))

    # # MLP
    # return pickle.load(open("./models/mlp_pipeline.pkl", "rb"))