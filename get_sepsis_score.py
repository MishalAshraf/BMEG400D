#!/usr/bin/env python

from sklearn.pipeline import Pipeline
import pickle

from preprocessing import preprocess

def get_sepsis_score(data, model):

    preprocessed = preprocess(data).reshape((1,-1))

    # Take positive sepsis label prediction only
    score = model.predict_proba(preprocessed)[-1][1]

    # Set default threshold
    label = score > 0.030

    return score, label

def load_sepsis_model():
    # Comment out on the one that you want to run

    # Logistic Regression
    #return pickle.load(open("./models/logreg_pipeline.pkl", "rb"))

    # # SVC
    # return pickle.load(open("./models/svc_pipeline.pkl", "rb"))

    # # MLP
    # return pickle.load(open("./models/mlp_pipeline.pkl", "rb"))

    #GBM 
    return pickle.load(open("./models/gbm_pipeline.pkl", "rb"))