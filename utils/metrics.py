import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def QLIKE(pred, true):
    """
    QLIKE loss (Patton, 2011, Eq. 24).
        QLIKE = mean[ σ²/σ²_hat - ln(σ²/σ²_hat) - 1 ]
    Inputs are log-variance (ln RV); exponentiated internally.
    Smaller = better.
    """
    rv_act  = np.exp(true)
    rv_pred = np.exp(pred)
    valid   = (rv_act > 0) & (rv_pred > 0)
    ratio   = rv_act[valid] / rv_pred[valid]
    return float(np.mean(ratio - np.log(ratio) - 1))


def metric(pred, true):
    mae   = MAE(pred, true)
    mse   = MSE(pred, true)
    rmse  = RMSE(pred, true)
    mape  = MAPE(pred, true)
    mspe  = MSPE(pred, true)
    rse   = RSE(pred, true)
    corr  = CORR(pred, true)
    qlike = QLIKE(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, qlike
