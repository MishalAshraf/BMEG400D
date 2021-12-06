# BMEG400D

## Python Setup

1. [Download Anaconda](https://www.anaconda.com/products/individual)
2. Create a conda environment
```
conda create -n bmeg400d python=3.8
conda activate bmeg400d
```
3. Install necessary dependencies
```
conda install numpy pandas scikit-learn tqdm
```

## Python Execution
The following is done in the activate anaconda environment.

### Training the models
```
# Assumes ../training_2021-11-15 and ../testing_2021-11-15 exist
python trainer.py
```

### Running the driver on the test folder
```
# Assumes previous step finished
python driver.py ../testing_2021-11-15 ../results
```

### Evaluating the driver results (scoring)
```
# Assumes previous step finished 
python evaluate_sepsis_score.py ../testing_2021-11-15 ../results
```