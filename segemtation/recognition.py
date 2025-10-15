import numpy as np
from sklearn.preprocessing import OneHotEncoder

def build_mapped(X, n_groups=5, size=50):
    groups = []
    for _ in range(n_groups):
        W = np.random.randn(X.shape[1], size) * 0.1
        b = np.random.randn(size) * 0.01
        M = np.tanh(X @ W + b)
        groups.append(M)
    return np.hstack(groups)

def build_enhanced(M, enh_nodes=1000):
    W = np.random.randn(M.shape[1], enh_nodes) * 0.1
    b = np.random.randn(enh_nodes) * 0.01
    E = np.tanh(M @ W + b)
    return E

def train_bls(X, y, map_groups=8, group_size=50, enh_nodes=500, ridge=1e-3):
    enc = OneHotEncoder(sparse_output=False)
    Y = enc.fit_transform(y.reshape(-1, 1))
    M = build_mapped(X, map_groups, group_size)
    E = build_enhanced(M, enh_nodes)
    A = np.hstack([M, E])
    I = np.eye(A.shape[1])
    Wout = np.linalg.solve(A.T @ A + ridge * I, A.T @ Y)
    return {"Wout": Wout, "enc": enc, "map_groups": map_groups, "group_size": group_size, "enh_nodes": enh_nodes}

def predict_bls(model, X):
    M = build_mapped(X, model["map_groups"], model["group_size"])
    E = build_enhanced(M, model["enh_nodes"])
    A = np.hstack([M, E])
    logits = A @ model["Wout"]
    preds = np.argmax(logits, axis=1)
    return model["enc"].categories_[0][preds]
