import numpy as np

def ffill(data, fill=0, inplace=False):
    """
    From https://stackoverflow.com/questions/62038693/numpy-fill-nan-with-values-from-previous-row
    """
    if inplace:
        arr = data
    else:
        arr = np.copy(data)

    mask = np.isnan(arr[0])
    arr[0][mask] = fill
    for i in range(1, len(arr)):
        mask = np.isnan(arr[i])
        arr[i][mask] = arr[i - 1][mask]
    
    return arr

def zero_fill(data):
    return np.nan_to_num(data)

def get_n_hour_window(data, n=6):
    return data[-1:-1-n:-1][::-1]

def mean_fill(data):
    no_sepsis_mean = np.array([  0.12676197,   0.14661883,   0.61020852,   0.15999902,   0.14751986,
    0.29433588,   0.19036034,  0.88045043,   0.88546377,   0.90366314,
   0.91775878,   0.92534062,   0.92702581,   0.77399817,   0.93383354,
   0.87705432,   0.88933529,   0.90864108,   0.92378568,   0.92761405,
  85.70670524,  97.16318288,  37.13735284, 121.49675552,  80.76916626,
  62.13674786,  18.96481826,   0.7743345,    7.35231395,  43.30355058,
  17.85712074,   5.00185185,   1.11104007, 142.85706282,   2.00285088,
   4.13735736,  29.44728293,  10.17916568,  13.42524691, 175.52625571,
   4.54966313,   1.255175,     0.19277748,   9.41160971,   6.50172579,
   5.2817839,    2.39828074,   0.08802783,   0.01648646,   1.96906876,
   0.00940017,   0.40885563,   0.00037632,  17.20338757,   0.00866122,
   0.10924889,   0.60186379,   0.12991395,   0.12466099,   0.6334775,
  29.96274654,  83.9801145,   94.86784959,  34.0494882,  117.89988777,
  79.67283726,  52.02364519,  18.44307804,   0.23880301,   3.70057296,
  20.25188121,  19.74080633,   5.46871641,   1.27188553, 114.89500631,
   1.48901677,   3.43985162,  25.29508724,   8.270406,     8.9250373,
                              153.69347756,  62.74712012,   0.57530749 ])
    return np.nan_to_num(data, nan=no_sepsis_mean)

def preprocess(data):
    d = data[:, 1:21]

    pct_fill = np.sum(np.isnan(d), axis=0)/d.shape[0]
    mean_features = np.mean(get_n_hour_window(d, 6), axis=0)
    std_features = np.std(get_n_hour_window(d, 6), axis=0)

    d_ffill = ffill(data)
    last_row = d_ffill[-1]

    preprocessed = np.concatenate(
        [pct_fill, mean_features, std_features, last_row], axis=0)

    preprocessed = mean_fill(preprocessed)

    return preprocessed
