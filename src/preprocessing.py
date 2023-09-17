import pandas as pd
import numpy as np
import random

def data_processing(file_name):

    """reads and preprocesses data

    Returns:
        preprocessed data
    """
    ## data reading
    random.seed(42)
    ratio = 0.5
    data = pd.read_csv(file_name,
                           header=0,
                           skiprows=lambda i: i>0 and random.random() < ratio
                           )
    ## isna replacement (isna values replaced by random choice of other values)
    for col in data.columns:
        if any(data[col].isna()):
            m = data[col].isna()
            data.loc[m, col] = np.random.choice(data.loc[~m, col], m.sum())


    return data
