import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

def delete_zeroes(file_path, axis = 1, sep = "\t", index_col = 0):
    # axis = 1 for columns, axis = 0 for rows
    df = pd.read_csv(file_path, sep=sep, index_col=index_col)
    try:
        if axis not in [0, 1]:
            raise ValueError("Invalid axis value. Use 1 for columns or 0 for rows.")
        if axis == 0:
            df = df.transpose()
    except ValueError as e:
        print(e)
        return None
    df["sum"] = df.sum(axis = 1)
    df = df.query("sum != 0.0")
    df = df.drop(columns=["sum"])
    df = df.transpose()
    return df

@njit
def entropy_from_labels(x):
    max_label = x.max()
    counts = np.zeros(max_label + 1, dtype=np.int64)

    for v in x:
        counts[v] += 1

    n = x.size
    h = 0.0

    for c in counts:
        if c > 0:
            p = c / n
            h -= p * np.log(p)

    return h


@njit
def joint_entropy_from_labels(x, y, n_bins_y):
    joint = x * n_bins_y + y
    return entropy_from_labels(joint)


@njit(parallel=True)
def mi_matrix_numba(X):
    n_samples, n_features = X.shape
    mi = np.zeros((n_features, n_features), dtype=np.float32)
    entropies = np.zeros(n_features, dtype=np.float64)

    max_label = X.max()
    n_bins = max_label + 1

    for i in prange(n_features):
        entropies[i] = entropy_from_labels(X[:, i])
        mi[i, i] = 1.0

    for i in prange(n_features):
        x = X[:, i]
        for j in range(i + 1, n_features):
            y = X[:, j]
            hxy = joint_entropy_from_labels(x, y, n_bins)
            score = entropies[i] + entropies[j] - hxy
            mi[i, j] = score
            mi[j, i] = score

    return mi


def create_mi_matrix(df, num_threads=25):
    if num_threads != -1:
        set_num_threads(num_threads)
    genes = df.columns.tolist()
    for col in df.columns:
        df[col] = pd.factorize(df[col], sort=True)[0].astype("int32")
    X = df.to_numpy(dtype=np.int32)
    mi = mi_matrix_numba(X)
    
    df_mi = pl.from_numpy(mi, schema=genes)
    df_mi = df_mi.insert_column(0, pl.Series("", genes))
    return df_mi