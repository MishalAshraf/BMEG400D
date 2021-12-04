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

    column_names = [
    "pct_HR", "pct_O2Sat", "pct_Temp", "pct_SBP", "pct_MAP", "pct_DBP", "pct_Resp", "pct_FiO2", "pct_pH", "pct_PaCO2", "pct_BUN", 
    "pct_Calcium", "pct_Creatinine", "pct_Glucose", "pct_Magnesium", "pct_Potassium", "pct_Hct", "pct_Hgb", "pct_WBC", "pct_Platelets",
    
    "6mean_HR", "6mean_O2Sat", "6mean_Temp", "6mean_SBP", "6mean_MAP", "6mean_DBP", "6mean_Resp", "6mean_FiO2", "6mean_pH", "6mean_PaCO2", "6mean_BUN", 
    "6mean_Calcium", "6mean_Creatinine", "6mean_Glucose", "6mean_Magnesium", "6mean_Potassium", "6mean_Hct", "6mean_Hgb", "6mean_WBC", "6mean_Platelets", 
    
    "6std_HR", "6std_O2Sat", "6std_Temp", "6std_SBP", "6std_MAP", "6std_DBP", "6std_Resp", "6std_FiO2", "6std_pH", "6std_PaCO2", "6std_BUN", 
    "6std_Calcium", "6std_Creatinine", "6std_Glucose", "6std_Magnesium", "6std_Potassium", "6std_Hct", "6std_Hgb", "6std_WBC", "6std_Platelets",
    
    "ICULOS", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2", "pH", "PaCO2", "BUN", 
    "Calcium", "Creatinine", "Glucose", "Magnesium", "Potassium", "Hct", "Hgb", "WBC", "Platelets", 
    "Age", "Sex", "SepsisLabel",
    ]


    # Get all files in folder
    files = []
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)) and not f.lower().startswith('.') and f.lower().endswith('csv'):
            files.append(os.path.join(folder, f))

     # Load all data from files. Mishal: I refactored this to work more like the actual test. msg me for details if this isnt clear

    df_tot = pd.DataFrame(columns=column_names)

    for file in files:
        data = pd.read_csv(file, delimiter=',')
        num_rows = len(data)
        df_file = pd.DataFrame(columns=column_names)
        #######df_tot = pd.DataFrame(columns=column_names)

        for t in range(num_rows):
            current_data = data[:t+1]
            feature_data = current_data.iloc[:,1:21]
    
         ###seperate step for forward fill in dataset

         #print(t, feature_data.iloc[-1])
            pct_fill_temp = 1- ((feature_data.isna().sum()/feature_data.shape[0]))
            mean_temp = (six_hr_window(feature_data).mean())
            std_temp = (six_hr_window(feature_data).std())
    
    
            current_data.ffill()
            #current_data.fillna(0)
            last_temp = (current_data.iloc[t,:]) ###only for training, needs to be updated for actual submission
    
            temp_df = pd.concat([pct_fill_temp, mean_temp,std_temp,last_temp],ignore_index = True).to_frame().T
            temp_df.fillna(0,inplace=True)
    
            for col in temp_df:
                temp_df.rename(columns={col:column_names[col]},inplace=True)
        
        
        df_file = pd.concat([df_file,temp_df],ignore_index=True)

    df_tot = pd.concat([df_tot,df_file],ignore_index=True)

    df_tot.to_csv('SAVE_DATA.csv')
    # Replace remaining NaNs with 0
    #df = df.fillna(0)
    
    return df_tot

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
    for col in dataframe:
        print(col)
    X = dataframe.ilocdrop(columns="SepsisLabel")
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


#Creates an array of all the training data csv filenames
def load(mypath):
    filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
    return filenames

## Taken from Sample Code
# This method adds rows sequentially to a central dataframe. 
def load_challenge_data(file,step_array):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split(',')
        data = np.loadtxt(f, delimiter=',')
        for i in range(0,len(data)):
            if i == 0 and (len(step_array) == 1):
                step_array[len(step_array) - 1] = data[i]
            else:
                step_array = np.vstack([step_array, data[i]])
            # DO Something with step_array. Step array will grow, one row at a time, untill all training data
            # csv's are added
    return data

# Does a step wise ladder through the training data
# Each row is added in load_challenge_data method
def step_wise_loader():
    step_array = np.ndarray((1,24))
    filenames = load(TRAINING_FOLDER)
    for i in range (0,len(filenames)):
        print(filenames[i])
        input_file = os.path.join(TRAINING_FOLDER, filenames[i])
        step_array = load_challenge_data(input_file,step_array)

def six_hr_window(current_data):
    if current_data.shape[0] < 6:
        return current_data
    else:
        return current_data.iloc[-6:]

        
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
