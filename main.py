import itertools
import math

import pandas as pd
import numpy as np  # математика
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from tkinter import *
from tkinter.scrolledtext import ScrolledText
import easygui
import warnings  # UI

# from plotly import graph_objs as plt
# from plotly.offline import plot  # графики
import matplotlib.pyplot as plt

db = pd.DataFrame  # пустой DataFrame для работы с данными


def load_data():
    fin_name = easygui.fileopenbox(filetypes=["*.csv"], default='*.csv')
    try:
        global db
        db = pd.read_csv(fin_name, index_col=[0])  # устанавливаем считываемое значение от первого столбика
        dataView.insert(0.0, db)
        # amount = len(db.columns) + 1
        exponBut["state"] = NORMAL
        arimaBut["state"] = NORMAL
        fourierBut["state"] = NORMAL
    except Exception as e:
        easygui.msgbox(msg="ОШИБКА: {}".format(e), title="ERROR", ok_button="OK")


def print_data():
    global db
    print(db)


def ARIMA():
    warnings.filterwarnings("ignore")

    global db
    results = []
    text = ""  # для дебаггинга

    minaic = math.inf
    optimal_params = ((0, 0, 0), (0, 0, 0, 12))

    p = d = q = range(0, 2)  # p - AR, d - I, q - MA
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]  # 12 - годовые периоды
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(db, order=param, seasonal_order=param_seasonal,
                                                enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()
                text += 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic) + '\n'
                if results.aic < minaic:
                    minaic = results.aic
                    optimal_params = param, param_seasonal
            except:
                continue
    print(text)
    print(optimal_params, minaic, sep='\t')

    mod = sm.tsa.SARIMAX(db, order=optimal_params[0], seasonal_order=optimal_params[1], enforce_stationarity=False,
                         enforce_invertibility=False)
    results = mod.fit()
    print(results.summary().tables[1])
    # results.plot_diagnostics(figsize=(15,9))
    # plt.show()
    prediction = results.get_forecast(steps=100)
    pred_interval = prediction.conf_int()
    db_extended = pd.DataFrame(prediction.predicted_mean)
    db = pd.concat([db, prediction.predicted_mean])
    db.columns = ['Original data', 'Predicted data']

    ax = db.plot(label='observed', figsize=(20, 15))
    #prediction.predicted_mean.plot(ax=ax, label='Forecast')
    #ax.fill_between(pred_interval.index,
    #pred_interval.iloc[:, 0],
    #pred_interval.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Data')
    plt.legend()
    plt.show()
    db.drop(['Predicted data'], axis=1)
    print(db)
    return results


def get_matrix_from_series(input: pd.Series, m, l):
    return input.values.reshape(m, l)


def get_matrix_and_vector(period: np.array) -> (np.ndarray, np.ndarray):
    l = len(period) - 1
    copy_l = l

    y = np.empty((0,))
    matrix = np.empty((0, copy_l + 1))

    for t in range(0, l + 1):
        row = np.array([.5])

        for k in range(1, copy_l + 1):
            row = np.append(row, math.cos(math.pi * k * t / l))

        row = np.reshape(row, (1, copy_l + 1))
        matrix = np.append(matrix, row, axis=0)
        y = np.append(y, period[t])

    return matrix, y


def solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert np.linalg.det(A) != 0
    return np.linalg.solve(A, b)


def get_new_fourier(periods: np.array, p: int = 1) -> list:
    new = []
    all_periods = []
    for period in periods:
        x, y = get_matrix_and_vector(period)
        all_periods.append(solve_system(x, y))
    all_periods = np.array(all_periods)

    for i in range(all_periods.shape[1]):
        new.append(predict(all_periods[:, i], p=p))

    return new


def get_delay(input: np.ndarray, p: int = 1) -> np.ndarray:
    input_copy = np.copy(input)

    m = input_copy.shape[0] % p

    if m != 0:
        input_copy = np.delete(input_copy, range(m))

    row_dim = input_copy.shape[0] // p
    col_dim = p

    delay = np.resize(input_copy, new_shape=(row_dim, col_dim)).T

    return delay

def find_nearest(row: np.ndarray, p: int) -> set:
    neighbour = 2 * p + 1

    last = row[-1]
    all_neighbours = row[:-1]

    index = set(np.argsort(np.abs(all_neighbours - last))[:neighbour])

    return index


def predict(input: np.ndarray, p: int = 1) -> float:
    delay = get_delay(input, p)
    last_row = delay[-1, :]
    nearest = find_nearest(last_row, p)

    y = np.empty((0,))
    x = np.empty((0, p + 1))
    for i in nearest:
        y = np.append(y, delay[0, i + 1])
        row = np.append(np.array([1]), delay[:, i])
        row = np.reshape(row, (1, p + 1))
        x = np.append(x, row, axis=0)

    c = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    prediction = sum(np.append(np.array([1]), delay[:, -1]) * c)

    return prediction


def predict_next_period(new: list, l: int):
    new_period = []
    for t in range(0, l):
        s = new[0] / 2
        for k in range(1, len(new)):
            s += new[k] * math.cos(math.pi * k * t / (l - 1))
        new_period.append(s)

    return new_period


def FOURIER():
    global db

    start_date = db.index[0]
    #print(start_date)

    data = pd.Series()
    #for _, row in db.drop('Date', axis=1).iterrows():
    data = data._append(pd.Series(db['#Passengers'].values))

    data.index = pd.date_range(start_date, freq='M', periods=12 * 12)

    training = data[data.index.year < 1960]
    test_series = data[data.index.year == 1960]

    size = 11  # количество периодов
    l = 12  # длина периода
    p = 1  # величина задержек

    matrix = get_matrix_from_series(training, size, l)
    new_data = get_new_fourier(matrix, p)
    test_pred = pd.Series(predict_next_period(new_data, l), index=test_series.index)

    mae = round(mean_absolute_error(test_pred, test_series), 2)
    mape = round(mean_absolute_percentage_error(test_series, test_pred), 3)

    test_pred.plot(figsize=(12, 6), label='Прогноз', linewidth=3)
    test_series.plot(figsize=(12, 6), label='Факт', linewidth=3)

    plt.legend()
    plt.text(test_series.index[4], 55000, f'Mean absolute error = {mae}', fontsize=15)
    plt.text(test_series.index[4], 54000, f'Mean absolute percentage error = {mape}', fontsize=15)
    plt.show()

    return


def initial_seasonal_components(db, slen):
    seasonals = {}
    season_avg = []
    n_seasons = int(len(db) / slen)
    for j in range(n_seasons):
        season_avg.append(sum(db[slen * j:slen * j + slen]) / float(slen))
    for i in range(slen):
        sums = 0.0
        for j in range(n_seasons):
            sums += db[slen * j + i] - season_avg[j]
        seasonals[i] = sums / n_seasons
    return seasonals


def initial_trend(db, slen):
    sums = 0.0
    for i in range(slen):
        sums += float(db[i + slen] - db[i]) / slen
    return sums / slen


def TSE(db, slen, alpha, beta, gamma, n_preds,
        scaling_factor):  # модель Хольта-Винтерса - три компоненты: тренд, сезонность и уровень
    result = []
    smooth = []
    season = []
    trend = []
    predictDeviation = []
    upperBond = []
    lowerBond = []

    seasonals = initial_seasonal_components(db, slen)

    for i in range(len(db) + n_preds):
        if i == 0:
            smoothConst = db[0]
            trendConst = initial_trend(db, slen)
            result.append(db[0])
            smooth.append(smoothConst)
            trend.append(trendConst)
            season.append(seasonals[i % slen])

            predictDeviation.append(0)

            upperBond.append(result[0] + scaling_factor * predictDeviation[0])

            lowerBond.append(result[0] - scaling_factor * predictDeviation[0])

            continue
        if i >= len(db):
            m = i - len(db) + 1
            result.append((smoothConst + m * trendConst) + seasonals[i % slen])
            predictDeviation.append(predictDeviation[-1] * 1.01)
        else:
            val = db[i]
            lastSmoothConst, smoothConst = smoothConst, alpha * (val - seasonals[i % slen]) + (1 - alpha) * (
                    smoothConst + trendConst)
            trendConst = beta * (smoothConst - lastSmoothConst) + (1 - beta) * trendConst
            seasonals[i % slen] = gamma * (val - smoothConst) + (1 - gamma) * seasonals[i % slen]
            result.append(smoothConst + trendConst + seasonals[i % slen])

            predictDeviation.append(gamma * np.abs(db[i] - result[i]) + (1 - gamma) * predictDeviation[-1])

        upperBond.append(result[-1] + scaling_factor * predictDeviation[-1])
        lowerBond.append(result[-1] - scaling_factor * predictDeviation[-1])

        smooth.append(smoothConst)
        trend.append(trendConst)
        season.append(seasonals[i % slen])
    # plot_test(db, title='START DATA')
    return result


def plot_original():
    global db
    # print(db[db.columns[0]].values)
    with plt.style.context('ggplot'):
        plt.figure(figsize=(20, 8))
        for alpha in [0.9]:
            for beta in [0.9, 0.02]:
                for gamma in [0.9]:
                    # print(alpha, beta, gamma, sep='\t', end='\n')
                    plt.plot(TSE(db[db.columns[0]].values, 6, alpha, beta, gamma, 10, 2.5),
                             label="Alpha {}, beta {}, gamma {}".format(alpha, beta, gamma))
        plt.plot(db[db.columns[0]].values, label='Actual')
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title('Comparison of different parameters')
        plt.grid(True)
        plt.show()


window = Tk()
window.title("Прогнозирование")
window.geometry('900x600')

frame = Frame(window, padx=10, pady=10)
frame.pack(expand=True)

window.grid_columnconfigure(1, minsize=300)
window.grid_columnconfigure(2, minsize=300)
window.grid_columnconfigure(3, minsize=300)

loadData = Button(frame, text='Загрузить данные', command=load_data)
loadData.grid(row=1, column=1, padx=50, sticky='s')

printData = Button(frame, text='Вывести данные', command=print_data)
printData.grid(row=3, column=1, padx=50, sticky='n')

dataView = ScrolledText(frame, width=32, height=32)
dataView.grid(column=2, row=1, rowspan=3, sticky='nsew')

arimaBut = Button(frame, text='Метод ARIMA', command=ARIMA, state=DISABLED)
arimaBut.grid(row=1, column=3, padx=50, sticky='s')

fourierBut = Button(frame, text='Метод рядов Фурье', command=FOURIER, state=DISABLED)
fourierBut.grid(row=2, column=3, padx=50)

exponBut = Button(frame, text='Метод экспоненциального сглаживания', command=plot_original, state=DISABLED)
exponBut.grid(row=3, column=3, sticky='n')

window.mainloop()
