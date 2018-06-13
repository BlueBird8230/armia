# coding: utf-8
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt
# from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA


def obtain_data():
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y')
    return pd.read_csv('data.csv', parse_dates=['Year'], index_col='Year', date_parser=dateparse)


def stability(ts):
    adf, p_value = adfuller(ts, autolag='AIC')[:2]
    print('adf: ', adf)
    print('p_value: ', p_value, '\n')
    return p_value


def best_diff(df, max_diff_turn=8):
    ret = []
    for i in range(0, max_diff_turn):
        temp = df.copy()
        if i == 0:
            temp['diff'] = temp[temp.columns[0]]
        else:
            temp['diff'] = temp[temp.columns[0]].diff(i)
            temp = temp.drop(temp.iloc[:i].index)
        p_value = stability(temp['diff'])
        if p_value < 0.01:
            t = {
                'iter': i,
                'p_val': p_value
            }
            print(t)
            ret.append(t)
    ret = sorted(ret, key=lambda val: val['iter'])
    print(ret)
    return ret[0]['iter']


def proper_model(data_ts, max_lag):
    init_bic = float("inf")
    init_p = 0
    init_q = 0
    init_proper_model = None
    for p in np.arange(max_lag):
        for q in np.arange(max_lag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results = model.fit(disp=-1, method='css')
            except Exception as e:
                print(e, 'continue')
                continue
            bic = results.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_proper_model = results
                init_bic = bic
    return init_bic, init_p, init_q, init_proper_model
