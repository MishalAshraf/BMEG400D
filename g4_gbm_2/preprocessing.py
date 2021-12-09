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
    no_sepsis_mean = np.array(
     [  0.12676197,   0.14661883,   0.61020852,   0.15999902,
         0.14751986,   0.29433588,   0.19036034,   0.88045043,
         0.88546377,   0.90366314,   0.91775878,   0.92534062,
         0.92702581,   0.77399817,   0.93383354,   0.87705432,
         0.88933529,   0.90864108,   0.92378568,   0.92761405,
        82.57064094,  93.26850059,  33.34061027, 115.83188573,
        78.29878104,  51.09467906,  18.08841853,   0.23640201,
         3.63211969,  19.86940833,  19.25782034,   5.30272077,
         1.23816785, 112.32276595,   1.44593559,   3.35476181,
        24.67584082,   8.06330512,   8.70457884, 149.59499735,
         4.61919311,   2.72884646,   0.789973  ,   8.73021105,
         6.1423694 ,   4.14604906,   1.93830912,   0.00986525,
         0.07085723,   0.61218275,   0.60617626,   0.22156671,
         0.04133885,   6.60453028,   0.04902431,   0.10999772,
         0.74221072,   0.24179625,   0.29888328,   5.02266635,
        81.55146887,  92.0933815 ,  32.5363017 , 114.25122019,
        77.28986392,  50.37861379,  17.78354401,   0.23376151,
         3.55383074,  19.43715027,  18.62427829,   5.0806117 ,
         1.19356772, 109.42091765,   1.38753315,   3.24674948,
        23.86998111,   7.78992391,   8.41241175, 144.14978347,
         7.62458521,   5.25370861,   1.78271226,  13.25833654,
         9.25827216,   6.24200141,   2.81109688,   0.02034662,
         0.16068661,   1.29669411,   1.38337585,   0.48707853,
         0.09437482,  13.49078154,   0.11204154,   0.24646898,
         1.68628851,   0.55170219,   0.68091464,  11.48055917,
         1.68480259,   1.90379794,   0.75428219,   2.44987737,
         1.63854549,   1.09786966,   0.40890124,   0.00263462,
         0.07296916,   0.40872491,   0.49939249,   0.1705466 ,
         0.03476691,   2.75323702,   0.04414967,   0.08854544,
         0.64118808,   0.21374563,   0.22758784,   4.22473834,
        29.96274654,  83.9801145 ,  94.86784959,  34.0494882 ,
       117.89988777,  79.67283726,  52.02364519,  18.44307804,
         0.23880301,   3.70057296,  20.25188121,  19.74080633,
         5.46871641,   1.27188553, 114.89500631,   1.48901677,
         3.43985162,  25.29508724,   8.270406  ,   8.9250373 ,
       153.69347756,  62.74712012,   0.57530749,        ])
    return np.nan_to_num(data, nan=no_sepsis_mean)

def get_change(data):
    if data.shape[0] == 1:
        return data[-1] - data[-1]
    else:
        return data[-1] - data[-2]

def preprocess(data):
    ####data is the full frame. ###d, just does calculations on features other than age, sex, iculos

    d = data[:, 1:21]

    pct_fill = np.sum(np.isnan(d), axis=0)/d.shape[0]

    d_ffill = ffill(d) ###fill before window stats. 

    mean_3 = np.mean(get_n_hour_window(d_ffill, 3), axis=0)
    std_3 = np.std(get_n_hour_window(d_ffill, 3), axis=0)
    mean_6 = np.mean(get_n_hour_window(d_ffill, 6), axis=0)
    std_6 = np.std(get_n_hour_window(d_ffill, 6), axis=0)


    delta_features = get_change(d_ffill)

    data_ffill = ffill(data)
    last_row = data_ffill[-1]

    preprocessed = np.concatenate(
        [pct_fill, mean_3, std_3, mean_6, std_6, delta_features, last_row], axis=0)

    
    preprocessed = mean_fill(preprocessed)

    return preprocessed
