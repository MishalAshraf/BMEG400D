import numpy as np, os, sys
from os import walk
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd


## Taken from Sample Code
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split(',')
        data = np.loadtxt(f, delimiter=',')
    return data,column_names

#Creates an array of all the training data csv filenames
def load(mypath):
    filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
    return filenames

# Takes in the path to the training data files and filenames, and combines all CSV files into a 2D array 
# (Only needs to be run once), returns header of data
def combineTrainingData(filenames,mypath):
    input_file = os.path.join(mypath, filenames[0])
    data,header = load_challenge_data(input_file)
    for i in range(1,len(filenames)):
        input_file = os.path.join(mypath, filenames[i])
        temp,header = load_challenge_data(input_file)
        data = np.concatenate((temp,data))
    print(data[1])
    file = pd.DataFrame(data,columns=header)
    file.to_csv('trainingData.csv', index=False, header=True, sep=",",na_rep='NaN')

#Recieves the data and header from the combined CSV file
def getTrainingData(input_file):
    data,header = load_challenge_data(input_file)
    return data,header

### FIRST Pre-Processing Method ###
## Below we will change all NaN values with 0, and return a normalized X
def pre_processing_zeros(X):
    X_Zeros = np.nan_to_num(X)
    #L12 norm
    X_normalized = preprocessing.normalize(X_Zeros, norm='l2')
    return X_normalized

### Second Pre-Processing Method ###
## Below we will forward fill, and return a normalized X
def pre_processing_fill(X,header):
    tempX = X
    tempX[0] = np.nan_to_num(X[0])
    print(tempX)
    X_Fill = pd.DataFrame(tempX,columns=header[:-1])
    X_Fill = pd.DataFrame.ffill(X_Fill)
    print(X_Fill)
    #L12 norm
    X_normalized = preprocessing.normalize(X_Fill, norm='l2')
    return X_normalized

### Third Pre-processing Method ###
## Below we will replace all NaN values with the mean of the column, and return a normalized X
def pre_processing_Mean(X):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X_Mean = imp.transform(X)
    #L12 norm
    X_normalized = preprocessing.normalize(X_Mean, norm='l2')
    return X_normalized



### ONLY NEED TO RUN THIS ONE TIME ###
mypath = "./training_2021-11-15"
filename = load(mypath)
combineTrainingData(filename,mypath)
### ONLY NEED TO RUN THIS ONE TIME ###

input_file = "./trainingData.csv"
trainingData,header = getTrainingData(input_file)

# Labels
y = trainingData[:,-1]
# Our training Data
X = trainingData[:, :-1]

print(X)


