import yfinance as yf
import datetime
import numpy as np


def get_relative_stock_data(ticker: str, backdate=55, lb=60, lf=12, interval='5m'):

    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(backdate)

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    vals = stock_data['Open'].values

    rel_data = [[0 for _ in range(lb)] for _ in range(lb, len(vals) - lf)]

    for i in range(lb, len(vals) - lf):
        for j in range(lb):
            rel_data[i - lb][j] = vals[i] - vals[i - lb + j]

            if vals[i] > 1e-15:
                rel_data[i - lb][j] = rel_data[i - lb][j]/vals[i - lb + j]
            else:
                print("Divide by Zero")

    print("Finished Processing Lookback Data")

    rel_rn = [[0 for _ in range(lb)] for _ in range(lf)]

    for i in range(len(vals) - lf, len(vals)):
        for j in range(lb):
            rel_rn[i - len(vals) + lf][j] = vals[i] - vals[i - lb + j]
            rel_rn[i - len(vals) + lf][j] = rel_rn[i - len(vals) + lf][j] / vals[i - lb + j]

    print("Finished Processing Forward Data")

    return rel_rn, rel_data, vals


def momentum_check(vals, lb=60, lf=12, delta=0.005):
    up_mom = [0 for i in range(lb, len(vals) - lf)]
    down_mom = [0 for i in range(lb, len(vals) - lf)]

    for i in range(lb, len(vals) - lf):
        for j in range(1, lf):
            r = vals[i + j] - vals[i]
            r = r / vals[i]

            if r > delta:
                up_mom[i - lb] = 1
                break

            elif r < -delta:
                down_mom[i - lb] = 1
                break

    return up_mom, down_mom


def single_mom(vals, lb=60, lf=12, delta=0.005):
    mom = [1 for _ in range(lb, len(vals) - lf)]

    for i in range(lb, len(vals) - lf):
        for j in range(1, lf):
            r = vals[i + j] - vals[i]
            r = r / vals[i]

            if r > delta:
                mom[i - lb] = 2
            elif r < -delta:
                mom[i - lb] = 0
    return np.array(mom)


def single_mom_list(vals, lb=60, lf=12, delta=0.005):
    mom = [1 for _ in range(lb, len(vals) - lf)]

    for i in range(lb, len(vals) - lf):
        for j in range(1, lf):
            r = vals[i + j] - vals[i]
            r = r / vals[i]

            if r > delta:
                mom[i - lb] = 2
            elif r < -delta:
                mom[i - lb] = 0
    return mom


def get_last_hour(ticker, interval='5m', lb=60):
    end_time = datetime.datetime.now()
    start_time = datetime.timedelta(minutes=5*lb)

    stock_data = yf.download(ticker, start=start_time, end=end_time, interval=interval)
    vals = stock_data['Open'].values

    rel_rn = [[0 for j in range(lb)] for i in range(12)]

    for i in range(len(vals) - 12, len(vals)):
        for j in range(lb):
            rel_rn[i - len(vals) + 12][j] = vals[i] - vals[i - lb + j]
            rel_rn[i - len(vals) + 12][j] = rel_rn[i - len(vals) + 12][j] / vals[i - lb + j]

    return rel_rn


def get_relative_dict(ticker: str, backdate=55, lb=60, lf=12, interval='5m', inputs=None):
    if inputs is None:
        inputs = ['Open', 'High', 'Low']

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(backdate)

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    vals = dict()
    rel_rets = dict()
    rel_rn = dict()

    for inp in inputs:
        vals[inp] = stock_data[inp].values

        rel_data = [[0 for _ in range(lb)] for _ in range(lb, len(vals[inp]) - lf)]

        for i in range(lb, len(vals[inp]) - lf):
            for j in range(lb):
                rel_data[i - lb][j] = vals[inp][i] - vals[inp][i - lb + j]
                rel_data[i - lb][j] = rel_data[i - lb][j] / vals[inp][i - lb + j]

        rel_rets[inp] = np.array(rel_data)

    print("Finished Processing Lookback Data")

    for inp in inputs:
        rel_data = [[0 for _ in range(lb)] for _ in range(lf)]

        for i in range(len(vals[inp]) - lf, len(vals[inp])):
            for j in range(lb):
                rel_data[i - len(vals[inp]) + lf][j] = vals[inp][i] - vals[inp][i - lb + j]
                rel_data[i - len(vals[inp]) + lf][j] = rel_data[i - len(vals[inp]) + lf][j] / vals[inp][i - lb + j]

        rel_rn[inp] = np.array(rel_data)

    print("Finished Processing Forward Data")

    return vals, rel_rets, rel_rn


def ret_histograms(vals, rets, edges, lb, lf, batch_size):
    print("Starting Histograms")
    ret_hists = list()

    for i in range(batch_size, len(rets) - lf):
        tmp = list()

        for j in range(batch_size):
            hist, _ = np.histogram(rets[i - j], bins=edges, density=True)
            hist = hist.tolist()
            tmp.append(hist)
        ret_hists.append(tmp)

    print("Finished Processing Lookback Histograms")

    rn_hists = list()

    for i in range(len(rets) - lf, len(rets)):
        tmp = list()

        for j in range(batch_size):
            hist, _ = np.histogram(rets[i - j], bins=edges, density=True)
            hist = hist.tolist()
            tmp.append(hist)

        rn_hists.append(tmp)

    print("Finished Processing Current Histograms")

    target_hists = list()

    for i in range(lb + batch_size, len(vals) - lf):
        targets = list()

        for j in range(1, lf):
            tmp = vals[i + j] - vals[i]
            tmp = tmp / vals[i]
            targets.append(tmp)
        hist, _ = np.histogram(targets, bins=edges, density=True)

        hist = hist.tolist()

        target_hists.append(hist)

    print("Finished Histograms")

    return ret_hists, target_hists, rn_hists
