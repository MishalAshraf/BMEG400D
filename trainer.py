import sklearn as sk
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# from keras.wrappers.scikit_learn import KerasClassifier
# from keras import regularizers
# from keras.models import Sequential
# from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization

import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm

from preprocessing import preprocess

TRAINING_FOLDER = "../training_2021-11-15"
TEST_FOLDER = "../testing_2021-11-15"
MODEL_FOLDER = "./models"


# Data loading helpers
def list_files(folder):
    # Get all files in folder
    files = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
            files.append(os.path.join(folder, f))
    return files 

def load_data_all(file):
    header = data = None
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split(',')
        data = np.loadtxt(f, delimiter=',')

    return column_names, data

def load_data_stepiter(file):
    column_names, data = load_data_all(file)
    i = 1
    while i <= len(data):
        yield column_names, data[:i]
        i += 1


# Data loading functions
def load_data(folder):
    files = list_files(folder)

    column_names = []
    Xs = []
    ys = []
    for file in tqdm(files, desc=f"Loading and preprocessing data from {folder} ..."):
        for h, d in load_data_stepiter(file):
            column_names = h
            X, y = split_sepsis(h, d)
            Xs.append(preprocess(X))
            ys.append(y[-1])

    return column_names, np.array(Xs), np.array(ys)

def split_sepsis(column_names, data):
    
    if column_names[-1] =='SepsisLabel':
        X = data[:, :-1]
        y = data[:, -1]

        return X, y
    else:
        return data, None


# Models

def create_LRpipeline():
    print("Creating Logistic Regression pipeline ...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(
            random_state=7,
            solver='lbfgs'))
        ])

    return pipeline

def create_SVCpipeline():
    print("Creating SVC pipeline ...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            kernel="rbf",
            probability=True,
            verbose=True))
        ])

    return pipeline

def create_GBMpipeline():
    print("Creating GBM pipeline ...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbm', GradientBoostingClassifier(
            ))
        ])

    return pipeline


def create_MLPpipeline():
    print("Creating MLP Pipeline ...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            solver='adam',
            alpha=0.0001,
            hidden_layer_sizes=(32),
            random_state=7,
            verbose=True))
        ])
    return pipeline


# Trainers and Testers

def fit_model(model, X, y, outfile):
    print("Training model ...")
    model.fit(X, y)
    pickle.dump(model, open(MODEL_FOLDER+"/"+outfile, "wb"))
    return model

def evaluate_model(model, X, y):
    print("Evaluating model ...")
    AUROC = roc_auc_score(y, model.predict_proba(X)[:, 1])
    score = model.score(X, y)
    print(f"AUROC: {AUROC}")
    print(f"Accuracy: {score}")


        
if __name__ == "__main__":

    cnames_train, X_train, y_train = load_data(TRAINING_FOLDER)
    cnames_test, X_test, y_test = load_data(TEST_FOLDER)

    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    print(X_train.shape)
    print(y_train.shape)

    pipe_LR = create_LRpipeline()
    fit_model(pipe_LR, X_train, y_train, "logreg_pipeline.pkl")
    evaluate_model(pipe_LR, X_test, y_test)

    # pipe_SVC = create_SVCpipeline()
    # fit_model(pipe_SVC, X_train, y_train, "svc_pipeline.pkl")
    # evaluate_model(pipe_SVC, X_test, y_test)

    # pipe_MLP = create_MLPpipeline()
    # fit_model(pipe_MLP, X_train, y_train, "mlp_pipeline.pkl")
    # evaluate_model(pipe_MLP, X_test, y_test)

    pipe_GBM = create_GBMpipeline()
    fit_model(pipe_GBM, X_train, y_train, "gbm_pipeline.pkl")
    evaluate_model(pipe_GBM, X_test, y_test)
