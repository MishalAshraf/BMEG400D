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


TRAINING_FOLDER = "../training_2021-11-15"
TEST_FOLDER = "../testing_2021-11-15"
MODEL_FOLDER = "./models"

def load_data(folder):
    print(f"Loading data from {folder} ...")

    # Get all files in folder
    files = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
            files.append(os.path.join(folder, f))

    # Load all data from files
    temp = []
    for file in files:
        df = pd.read_csv(file, index_col=None, header=0)

        # Forward fill
        df = df.ffill()

        temp.append(df)

    # Merge all data into one DataFrame
    df = pd.concat(temp, axis=0, ignore_index=True)

    # Replace remaining NaNs with 0
    df = df.fillna(0)
    
    return df

# def normalize_values(x):
#     x_mean = np.array([
# 23.37, 84.97, 97.09, 36.86, 122.66, 81.97, 63.35, 
# 18.86, 0.49, 7.38, 41.16, 22.92, 7.79, 1.54, 131.05, 2.04, 4.13, 
# 31.14, 10.37, 11.23, 197.32, 62.65,0])
#     x_std = np.array([
# 19.2, 16.74, 2.98, 
# 0.71, 23.28, 16.33, 14.05, 5.09, 0.33, 0.06, 8.78, 17.89, 2.12, 
# 1.91, 46.41, 0.35, 0.59, 5.56, 1.95, 7.55, 101.61, 15.91,1])

#     x_norm = np.nan_to_num((x - x_mean) / x_std)
#     x_norm = np.array(x_norm)
#     x_norm = x_norm.reshape(-1,23)

#     return x_norm

# def load_timeseries_data(folder, horizon=3):
#     print(f"Loading time series data from {folder} with a {horizon} step horizon...")

#     # Get all files in folder
#     files = []
#     for f in os.listdir(folder):
#         if os.path.isfile(os.path.join(folder, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
#             files.append(os.path.join(folder, f))

#     # Load all data from files
#     Xs = []
#     ys = []
#     for i, file in enumerate(files):
#         df = pd.read_csv(file, index_col=None, header=0)

#         # Forward fill
#         df = df.ffill()

#         # Split X and y
#         X = df.drop(columns="SepsisLabel")
#         y = df["SepsisLabel"]

#         # Shift data values down by horizon
#         X = X.reindex(list(range(0, X.shape[0]+horizon))).reset_index(drop=True)
#         X = X.shift(horizon)

#         # Replace remaining NaNs with 0
#         X = X.fillna(0)

#         # Create a data series for each row by selecting it and its previous n rows of data
#         #   where n = horizon
#         for i in range(y.shape[0]):
#             x = X.iloc[i:i+horizon+1].values
#             x = normalize_values(x)
#             Xs.append(x)
#             ys.append(y[i])
    
#     return np.array(Xs), np.array(ys)


def split_sepsis(dataframe):
    X = dataframe.drop(columns="SepsisLabel")
    y = dataframe["SepsisLabel"]

    return X, y

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
            kernel="linear",
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

# def create_LSTM():
#     model = Sequential()
#     model.add(Bidirectional(LSTM(64,
#         kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#         dropout=0.3
#         )))
#     model.add(BatchNormalization())
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model

# def create_LSTMpipeline():
#     print("Creating LSTM Pipeline ...")
#     pipeline = Pipeline([
#         #('scaler', StandardScaler()),
#         ('lstm', KerasClassifier(
#             build_fn=create_LSTM,
#             epochs=20,
#             batch_size=64,
#             verbose=True))
#         ])
#     return pipeline

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
    # X_train_ts, y_train_ts = load_timeseries_data(TRAINING_FOLDER)
    # X_test_ts, y_test_ts = load_timeseries_data(TEST_FOLDER)

    data_train = load_data(TRAINING_FOLDER)
    data_test = load_data(TEST_FOLDER)

    if not os.path.isdir(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    
    X_train, y_train = split_sepsis(data_train)
    X_test, y_test = split_sepsis(data_test)

    #pipe_LR = create_LRpipeline()
    #fit_model(pipe_LR, X_train, y_train, "logreg_pipeline.pkl")
    #evaluate_model(pipe_LR, X_test, y_test)

    #pipe_SVC = create_SVCpipeline()
    #fit_model(pipe_SVC, X_train, y_train, "svc_pipeline.pkl")
    #evaluate_model(pipe_SVC, X_test, y_test)

    #pipe_MLP = create_MLPpipeline()
    #fit_model(pipe_MLP, X_train, y_train, "mlp_pipeline.pkl")
    #evaluate_model(pipe_MLP, X_test, y_test)

    # pipe_lstm = create_LSTMpipeline()
    # fit_model(pipe_lstm, X_train_ts, y_train_ts, "lstm_pipeline.pkl")
    # evaluate_model(pipe_lstm, X_test_ts, y_test_ts)

    pipe_GBM = create_GBMpipeline()
    fit_model(pipe_GBM, X_train, y_train, "gbm_pipeline.pkl")
    evaluate_model(pipe_GBM, X_test, y_test)