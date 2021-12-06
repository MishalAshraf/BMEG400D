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


def preprocess(data):
    d = data[:, 1:21]

    pct_fill = np.sum(np.isnan(d), axis=0)/d.shape[0]
    mean_features = np.mean(get_n_hour_window(d, 6), axis=0)
    std_features = np.std(get_n_hour_window(d, 6), axis=0)

    d_ffill = ffill(d)
    last_row = d_ffill[-1]

    preprocessed = np.concatenate(
        [pct_fill, mean_features, std_features, last_row], axis=0)

    preprocessed = zero_fill(preprocessed)

    return preprocessed
