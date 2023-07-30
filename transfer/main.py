import pandas as pd
import sqlite3
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from tkinter import *
import tkinter.scrolledtext as scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os
import subprocess
from datetime import datetime, timedelta
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import sys
import shutil
import webbrowser

conn = sqlite3.connect('models.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS models
             (model_name TEXT, max_accuracy REAL, min_loss REAL, f1_class_0 REAL, f1_class_1 REAL, f1_class_2 REAL, delta REAL)''')

device = torch.device('cpu')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.drop_out = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(26 * 26 * 64, 200)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(200, 200)
        self.act4 = nn.Tanh()
        self.fc3 = nn.Linear(200, 3)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.pool1(out)
        out = self.act2(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.act3(self.fc1(out))
        out = self.act4(self.fc2(out))
        out = self.soft(self.fc3(out))
        return out


class Net(nn.Module):
    def __init__(self, in_shape):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_shape, 200),
            nn.Tanh(),
            nn.Dropout(p=0.9),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 3)
        )

    def forward(self, x):
        out = self.layer1(x)
        return out


def windows(indicator, time_series, n_step):
    windowed_time_series = np.zeros((len(indicator), n_step + 1))
    i = 0
    for index, row in indicator.iterrows():
        our_index = time_series[time_series['date_time']
                                == row['date_time']].index.values.astype(int)[0]

        if our_index < n_step:
            windowed_time_series = np.delete(windowed_time_series, [-1], 0)
            continue

        windowed_time_series[i][1: n_step +
                                1] = time_series['target'].iloc[our_index - n_step + 1: our_index + 1]
        windowed_time_series[i][0] = np.array(row['class'])
        i += 1

    return windowed_time_series


def indicator(time_series, pips):

    last_point = time_series.iloc[0]
    ans = []
    ans.append([last_point['date_time'], last_point['target']])
    difference = time_series['target'].max() - time_series['target'].min()
    delta_down = round((difference / 100) * pips, 4)
    delta_up = -round((difference / 100) * pips, 4)
    print(delta_up, delta_down)
    is_up = False
    is_down = False

    for date_time, target in zip(time_series['date_time'], time_series['target']):

        if (last_point['target'] - target) <= delta_up:
            ans.append([date_time, target])
            last_point['target'] = target
            is_up = True
            is_down = False
        elif (last_point['target'] - target) >= delta_down:
            ans.append([date_time, target])
            last_point['target'] = target
            is_down = True
            is_up = False
        else:
            if last_point['target'] < target and is_up:
                ans.append([date_time, target])
                last_point['target'] = target
            elif last_point['target'] > target and is_down:
                ans.append([date_time, target])
                last_point['target'] = target

    ans_df = pd.DataFrame(ans, columns=('date_time', 'target'))
    ans_df['class'] = ans_df['target'].shift(periods=-1)
    ans_df = ans_df.dropna()
    ans_df['class'] = ans_df['target'] < ans_df['class']
    ans_df['class'] = ans_df['class'].astype(int)

    true_ans = []
    true_ans.append([ans_df['date_time'].iloc[0],
                    ans_df['target'].iloc[0], ans_df['class'].iloc[0]])
    for i in range(len(ans_df) - 1):
        if ans_df['class'].iloc[i] != ans_df['class'].iloc[i+1]:
            true_ans.append([ans_df['date_time'].iloc[i+1],
                            ans_df['target'].iloc[i+1], ans_df['class'].iloc[i+1]])
    true_ans_df = pd.DataFrame(
        true_ans, columns=('date_time', 'target', 'class'))
    return true_ans_df


def GenerateGAF(all_ts, window_size):

    rescaled_ts = np.zeros((window_size, window_size), float)
    min_ts, max_ts = np.min(all_ts), np.max(all_ts)
    diff = max_ts - min_ts
    if diff != 0:
        rescaled_ts = (all_ts - min_ts) / diff
    else:
        return np.zeros((1, window_size, window_size), float)

    this_gam = np.zeros((1, window_size, window_size), float)
    sin_ts = np.sqrt(np.clip(1 - rescaled_ts**2, 0, 1))

    this_gam[0] = np.outer(rescaled_ts, rescaled_ts) - np.outer(sin_ts, sin_ts)

    return this_gam


def show_model_info(model_name, max_accuracy, min_loss, f10, f11, f12, delta):
    model_select_window = tk.Toplevel(root)
    model_select_window.title("Информация о модели")

    currency_var = model_name[8:14]
    if model_name[:3] == 'GAN':
        dataset_var = 'Граммианское поле'
    else:
        dataset_var = 'MLP'

    window_width = 450
    window_height = 350
    screen_width = model_select_window.winfo_screenwidth()
    screen_height = model_select_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    model_select_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    main_frame = ttk.Frame(model_select_window, padding=20)
    main_frame.pack()

    select_text = f"""
    Model: {model_name}
    Max Accuracy: {round(max_accuracy, 4)}
    Min Loss: {round(min_loss, 4)}
    f1-мера для нейтральных точек: {round(f10, 4)}
    f1-мера для точек разворота снизу вверх: {round(f11, 4)}
    f1-мера для точек разворота сверху вниз: {round(f12, 4)}
    """

    select_label = ttk.Label(main_frame, text=select_text,
                             font=('Courier', 10), justify='left')
    select_label.pack(pady=10)

    button_style = ttk.Style()
    button_style.configure("Custom.TButton", font=("Arial", 12), width=20)

    testing_button = ttk.Button(main_frame, text="Протестировать модель", style="Custom.TButton",
                                command=lambda m=model_name, r=model_select_window: test_model(m, r, delta))
    testing_button.pack(pady=5)

    select_button = ttk.Button(main_frame, text="Выбрать модель", style="Custom.TButton",
                               command=lambda m=model_name: select_param(m, model_select_window))
    select_button.pack(pady=5)

    retrain_button = ttk.Button(main_frame, text="Дообучить модель", style="Custom.TButton", command=lambda m=model_name,
                                r=model_select_window: stop_criterion_retrain_model(m, r, currency_var, dataset_var, delta))
    retrain_button.pack(pady=5)

    delete_button = ttk.Button(main_frame, text="Удалить модель", style="Custom.TButton",
                               command=lambda m=model_name, r=model_select_window: delete_model(m, r))
    delete_button.pack(pady=5)

    model_select_window.mainloop()


def select_param(model_name, root_window):
    root_window.destroy()

    def start_trading(model_name):
        stop_loss = stop_loss_entry.get()
        take_profit = take_profit_entry.get()
        select_model(model_name, stop_loss, take_profit)
        select_param_window.destroy()  # Закрыть окно после выполнения функции select_model

    select_param_window = tk.Tk()
    select_param_window.title("Запуск")
    screen_width = select_param_window.winfo_screenwidth()
    screen_height = select_param_window.winfo_screenheight()
    x = (screen_width // 2) - (450 // 2)
    y = (screen_height // 2) - (265 // 2)
    select_param_window.geometry(f"450x265+{x}+{y}")
    select_param_window.resizable(False, False)

    main_frame = ttk.Frame(select_param_window, padding=20)
    main_frame.pack()

    # Название модели
    model_label = ttk.Label(
        main_frame, text=f"Модель: {model_name}", font=("Arial", 12))
    model_label.pack(pady=10)

    # Поле ввода stop loss
    stop_loss_label = ttk.Label(
        main_frame, text="Stop Loss:", font=("Arial", 10))
    stop_loss_label.pack()
    stop_loss_entry = ttk.Entry(main_frame, width=20)
    stop_loss_entry.pack()

    # Поле ввода take profit
    take_profit_label = ttk.Label(
        main_frame, text="Take Profit:", font=("Arial", 10))
    take_profit_label.pack()
    take_profit_entry = ttk.Entry(main_frame, width=20)
    take_profit_entry.pack()

    # Кнопка запуска
    start_button = ttk.Button(
        main_frame, text="Запустить", command=lambda m=model_name: start_trading(m))
    start_button.pack(pady=20)

    select_param_window.mainloop()


def retrain_model(model_name, window, delta, currency, dataset, f10_stop=0, f11_stop=0, f12_stop=0, loss_stop=999, accuracy_stop=0):
    window.destroy()
    app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_directory)

    csv_name = currency
    print(csv_name)
    df_eurusd = pd.read_csv(
        f'data/{csv_name}.csv', sep=';', parse_dates=[['<DATE>', '<TIME>']])
    columns = ('date_time', 'target')
    df_eurusd.columns = columns

    MODEL_PATH = f'models/{model_name}'

    if dataset == "Граммианское поле":
        X_train, X_test, y_train, y_test = GAN_dataset(df_eurusd, float(delta))
    else:
        X_train, X_test, y_train, y_test = MLP_dataset(df_eurusd, float(delta))

    losses = []
    accuracy = []
    f1_class_0 = []
    f1_class_1 = []
    f1_class_2 = []
    losses_val = []
    accuracy_val = []
    f1_class_0_val = []
    f1_class_1_val = []
    f1_class_2_val = []

    # Переменная для отслеживания состояния обучения
    training_stopped = False

    # Функция для остановки обучения
    def stop_training():
        nonlocal training_stopped
        training_stopped = True

    train_window = tk.Toplevel(root)
    train_window.title("Дообучение модели")
    stop_button = tk.Button(
        train_window, text="Остановить обучение", command=stop_training)
    stop_button.config(font=("Arial", 12))
    stop_button.pack()

    window_width = 650
    window_height = 600
    screen_width = train_window.winfo_screenwidth()
    screen_height = train_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    train_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    train_label = tk.Label(train_window)

    fig = Figure(figsize=(5, 3))
    plot = fig.add_subplot(111)
    plot.plot(losses[:], label="обучающая")
    plot.plot(losses_val[:], label="контрольная")
    plot.legend()
    plot.grid()

    canvas = FigureCanvasTkAgg(fig, master=train_window)
    canvas.draw()

    canvas.get_tk_widget().pack()

    model = torch.load(MODEL_PATH, map_location=torch.device(device))
    for name, param in model.named_parameters():
        if name != 'fc2.weight' and name != 'fc2.bias' and name != 'fc3.weight' and name != 'fc3.bias':
            param.requires_grad = False

    conv = model
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(conv.parameters(), lr=0.002)

    if X_train.shape[0] < 100:
        part = 4
    elif (X_train.shape[0] >= 100) and (X_train.shape[0] < 300):
        part = 8
    elif (X_train.shape[0] >= 300) and (X_train.shape[0] < 1000):
        part = 16
    else:
        part = 32

    for i in range(0, 300):
        for x_batch, y_batch in zip(torch.tensor_split(X_train, part), torch.tensor_split(y_train, part)):
            if training_stopped == True:
                break
            conv = conv.train()
            outputs = conv(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            conv = conv.eval()
            outputs_val = conv(X_test)
            loss_val = criterion_val(outputs_val, y_test)

            correct = 0
            total = 0
            _, predicted = torch.max(outputs_val.data, 1)
            _, y_test_value = torch.max(y_test.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test_value).sum().item()
            accuracy_val.append((correct / total) * 100)
            losses_val.append(float(loss_val))
            f1_class_0_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[0])
            f1_class_1_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[1])
            f1_class_2_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[2])

            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            _, y_batch_value = torch.max(y_batch.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch_value).sum().item()
            accuracy.append((correct / total) * 100)
            losses.append(float(loss))

            f1_class_0.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[0])
            f1_class_1.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[1])
            f1_class_2.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[2])
            train_f10 = f1_class_0[-1]
            train_f11 = f1_class_1[-1]
            train_f12 = f1_class_2[-1]
            f10 = f1_class_0_val[-1]
            f11 = f1_class_1_val[-1]
            f12 = f1_class_2_val[-1]
            if f10_stop == 0 and f11_stop == 0 and f12_stop == 0 and loss_stop == 999 and accuracy_stop == 0:
                if (losses_val[-1] == np.array(losses_val).min()):
                    f10 = f1_class_0_val[-1]
                    f11 = f1_class_1_val[-1]
                    f12 = f1_class_2_val[-1]
                    model_path_for_save = f"models/{model_name}"
                    torch.save(conv, model_path_for_save)
            else:
                if f1_class_0_val[-1] >= f10_stop and f1_class_1_val[-1] >= f11_stop \
                        and f1_class_2_val[-1] >= f12_stop and losses_val[-1] <= loss_stop and accuracy_val[-1] >= accuracy_stop:
                    f10 = f1_class_0_val[-1]
                    f11 = f1_class_1_val[-1]
                    f12 = f1_class_2_val[-1]
                    model_path_for_save = f"models/{model_name}"
                    torch.save(conv, model_path_for_save)
                    stop_training()

            train_text = f'''кол-во эпох: {i}
Средняя точность на обучающей выборке за последние 10 итераций {round(np.array(accuracy)[-10:-1].mean(), 4)}
Средняя точность на контрольной выборке за последние 10 итераций {round(np.array(accuracy_val)[-10:-1].mean(), 4)}
минимальная кросс-энтропийная ошибка на обучающей выборке {round(np.array(losses).min(), 4)}
минимальный кросс-энтропийная ошибка на контрольной выборке {round(np.array(losses_val).min(), 4)}
f1-мера на обучающей выборке для 
    нейтральных точек: {round(f1_class_0[-1], 4)}, 
    точек разворота вверх: {round(f1_class_1[-1], 4)}, 
    точек разворота вниз: {round(f1_class_2[-1], 4)}
f1-мера на контрольной выборке для 
    нейтральных точек: {round(f1_class_0_val[-1], 4)}, 
    точек разворота вверх: {round(f1_class_1_val[-1], 4)}, 
    точек разворота вниз: {round(f1_class_2_val[-1], 4)}
максимальная точность на обучающей выборке {round(np.array(accuracy).max(), 4)}%
максимальная точность на контрольной выборке {round(np.array(accuracy_val).max(), 4)}% \n'''

            train_label.config(text=train_text, font=(
                'Courier', 10), justify='left')
            train_label.pack()

            train_window.update()

            plot.clear()
            plot.plot(losses[:], label="обучающая")
            plot.plot(losses_val[:], label="контрольная")
            plot.legend()
            plot.grid()
            canvas.draw()

    max_accuracy = round(np.array(accuracy_val).max(), 4)
    min_loss = round(np.array(losses_val).min(), 4)
    c.execute("DELETE FROM models WHERE model_name=?", (model_name,))
    conn.commit()
    c.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?)",
              (model_name, max_accuracy, min_loss, f10, f11, f12, delta))
    conn.commit()
    update_models()


def delete_model(model_name, window):
    c.execute("DELETE FROM models WHERE model_name=?", (model_name,))
    conn.commit()
    window.destroy()
    update_models()


def load_models():
    c.execute("SELECT * FROM models")
    rows = c.fetchall()
    root.title("Выбор модели")
    window_width = 810
    window_height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    currency_var = tk.StringVar(root)
    currency_var.set("USDRUB")
    currency_options = ["USDRUB", "CNYRUB", "EURRUB", "AMDRUB",
                        "KZTRUB", "TRYRUB", "BYNRUB", "HKDRUB", "GLDRUB", "SLVRUB"]
    currency_dropdown = tk.OptionMenu(root, currency_var, *currency_options)
    currency_dropdown.config(font=("Arial", 12), width=10)

    dataset_var = tk.StringVar(root)
    dataset_var.set("Граммианское поле")
    dataset_options = ["Граммианское поле", "MLP"]
    dataset_dropdown = tk.OptionMenu(root, dataset_var, *dataset_options)
    dataset_dropdown.config(font=("Arial", 12), width=15)

    input_frame = tk.Frame(root)

    delta_label = tk.Label(input_frame, text="Дельта(%):")
    delta_label.config(font=("Arial", 12))

    delta_entry = tk.Entry(input_frame)
    delta_entry.insert(0, 1)
    delta_entry.config(validate="key", validatecommand=(
        root.register(validate_percentage), "%P"))
    delta_entry.config(font=("Arial", 12), width=10)

    currency_dropdown.grid(row=0, column=0, padx=10, pady=10)
    dataset_dropdown.grid(row=0, column=1, padx=10, pady=10)
    input_frame.grid(row=0, column=2, padx=10, pady=10, sticky="W")
    delta_label.grid(row=0, column=0, padx=5)
    delta_entry.grid(row=0, column=1, padx=5)

    # Плашки в три столбца
    for i, row in enumerate(rows):
        model_name, max_accuracy, min_loss, f1_class_0, f1_class_1, f1_class_2, delta = row
        button_text = f"{model_name}\nAccuracy: {round(max_accuracy, 4)}\nLoss: {round(min_loss, 4)}\nf1-мера класса 0: {round(f1_class_0, 4)}\nf1-мера класса 1: {round(f1_class_1, 4)}\nf1-мера класса 2: {round(f1_class_2, 4)}"
        button = tk.Button(root, text=button_text, command=lambda m=model_name, a=max_accuracy, l=min_loss,
                           f10=f1_class_0, f11=f1_class_1, f12=f1_class_2: show_model_info(m, a, l, f10, f11, f12, delta))
        button.config(font=("Arial", 10), bg="#b4e5a1",
                      fg="black", width=30, height=6)
        button.grid(row=i // 3 + 1, column=i % 3, padx=10, pady=10)

    # Кнопка "Обучить новую модель"
    train_button = tk.Button(root, text="Обучить новую модель", command=lambda: stop_criterion_model(
        currency_var.get(), dataset_var.get(), delta_entry.get()))
    train_button.config(font=("Arial", 12))
    train_button.grid(row=(len(rows) + 2) // 3 + 1, columnspan=3, pady=10)


def stop_criterion_retrain_model(model_name, window, currency, dataset, delta):
    # Создание нового окна для ввода параметров
    stop_criterion_window = tk.Toplevel(root)
    stop_criterion_window.title("Критерий остановки")

    # Размещение окна по центру экрана
    window_width = 500
    window_height = 280
    screen_width = stop_criterion_window.winfo_screenwidth()
    screen_height = stop_criterion_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    stop_criterion_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Метки и поля ввода для параметров
    neutral_label = tk.Label(stop_criterion_window,
                             text="F1-мера для класса 'Нейтральная точка':")
    neutral_label.grid(row=0, column=0, padx=10, pady=10)
    neutral_entry = tk.Entry(stop_criterion_window)
    neutral_entry.grid(row=0, column=1, padx=10, pady=10)

    up_label = tk.Label(stop_criterion_window,
                        text="F1-мера для класса 'Точка разворота вверх':")
    up_label.grid(row=1, column=0, padx=10, pady=10)
    up_entry = tk.Entry(stop_criterion_window)
    up_entry.grid(row=1, column=1, padx=10, pady=10)

    down_label = tk.Label(stop_criterion_window,
                          text="F1-мера для класса 'Точка разворота вниз':")
    down_label.grid(row=2, column=0, padx=10, pady=10)
    down_entry = tk.Entry(stop_criterion_window)
    down_entry.grid(row=2, column=1, padx=10, pady=10)

    loss_label = tk.Label(
        stop_criterion_window, text="Значение кросс-энтропийной ошибки для остановки:")
    loss_label.grid(row=3, column=0, padx=10, pady=10)
    loss_entry = tk.Entry(stop_criterion_window)
    loss_entry.grid(row=3, column=1, padx=10, pady=10)

    accuracy_label = tk.Label(stop_criterion_window,
                              text="Общая точность для остановки:")
    accuracy_label.grid(row=4, column=0, padx=10, pady=10)
    accuracy_entry = tk.Entry(stop_criterion_window)
    accuracy_entry.grid(row=4, column=1, padx=10, pady=10)

    # Функция для сохранения параметров и закрытия окна
    def save_parameters():
        neutral_f1 = float(neutral_entry.get()) if neutral_entry.get(
        ).isdigit() or '.' in neutral_entry.get() else 0
        up_f1 = float(up_entry.get()) if up_entry.get(
        ).isdigit() or '.' in up_entry.get() else 0
        down_f1 = float(down_entry.get()) if down_entry.get(
        ).isdigit() or '.' in down_entry.get() else 0
        loss = float(loss_entry.get()) if loss_entry.get(
        ).isdigit() or '.' in loss_entry.get() else 999
        accuracy = float(accuracy_entry.get()) if accuracy_entry.get(
        ).isdigit() or '.' in accuracy_entry.get() else 0

        # Закрытие окна
        stop_criterion_window.destroy()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!{currency}!!!!!!!!!!!!!!!!!!!")
        # Вызов функции для обработки параметров
        retrain_model(model_name, window, delta, currency, dataset,
                      neutral_f1, up_f1, down_f1, loss, accuracy)

    # Кнопка для сохранения параметров
    save_button = tk.Button(stop_criterion_window,
                            text="Сохранить", command=save_parameters)
    save_button.grid(row=5, columnspan=2, padx=10, pady=10)


def stop_criterion_model(currency, dataset, delta):
    # Создание нового окна для ввода параметров
    stop_criterion_window = tk.Toplevel(root)
    stop_criterion_window.title("Критерий остановки")

    # Размещение окна по центру экрана
    window_width = 500
    window_height = 280
    screen_width = stop_criterion_window.winfo_screenwidth()
    screen_height = stop_criterion_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    stop_criterion_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    # Метки и поля ввода для параметров
    neutral_label = tk.Label(stop_criterion_window,
                             text="F1-мера для класса 'Нейтральная точка':")
    neutral_label.grid(row=0, column=0, padx=10, pady=10)
    neutral_entry = tk.Entry(stop_criterion_window)
    neutral_entry.grid(row=0, column=1, padx=10, pady=10)

    up_label = tk.Label(stop_criterion_window,
                        text="F1-мера для класса 'Точка разворота вверх':")
    up_label.grid(row=1, column=0, padx=10, pady=10)
    up_entry = tk.Entry(stop_criterion_window)
    up_entry.grid(row=1, column=1, padx=10, pady=10)

    down_label = tk.Label(stop_criterion_window,
                          text="F1-мера для класса 'Точка разворота вниз':")
    down_label.grid(row=2, column=0, padx=10, pady=10)
    down_entry = tk.Entry(stop_criterion_window)
    down_entry.grid(row=2, column=1, padx=10, pady=10)

    loss_label = tk.Label(
        stop_criterion_window, text="Значение кросс-энтропийной ошибки для остановки:")
    loss_label.grid(row=3, column=0, padx=10, pady=10)
    loss_entry = tk.Entry(stop_criterion_window)
    loss_entry.grid(row=3, column=1, padx=10, pady=10)

    accuracy_label = tk.Label(stop_criterion_window,
                              text="Общая точность для остановки:")
    accuracy_label.grid(row=4, column=0, padx=10, pady=10)
    accuracy_entry = tk.Entry(stop_criterion_window)
    accuracy_entry.grid(row=4, column=1, padx=10, pady=10)

    # Функция для сохранения параметров и закрытия окна
    def save_parameters():
        neutral_f1 = float(neutral_entry.get()) if neutral_entry.get(
        ).isdigit() or '.' in neutral_entry.get() else 0
        up_f1 = float(up_entry.get()) if up_entry.get(
        ).isdigit() or '.' in up_entry.get() else 0
        down_f1 = float(down_entry.get()) if down_entry.get(
        ).isdigit() or '.' in down_entry.get() else 0
        loss = float(loss_entry.get()) if loss_entry.get(
        ).isdigit() or '.' in loss_entry.get() else 999
        accuracy = float(accuracy_entry.get()) if accuracy_entry.get(
        ).isdigit() or '.' in accuracy_entry.get() else 0

        # Закрытие окна
        stop_criterion_window.destroy()

        # Вызов функции для обработки параметров
        train_model(currency, dataset, delta, neutral_f1,
                    up_f1, down_f1, loss, accuracy)

    # Кнопка для сохранения параметров
    save_button = tk.Button(stop_criterion_window,
                            text="Сохранить", command=save_parameters)
    save_button.grid(row=5, columnspan=2, padx=10, pady=10)


def select_model(model_name, stop_loss, take_profit):
    app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_directory)

    models_directory = os.path.join(app_directory, 'models')
    selected_model_path = os.path.join(models_directory, model_name)

    # if os.path.exists(selected_model_path):
    #     app_directory = os.path.dirname(os.path.dirname(os.path.abspath(
    #         sys.argv[0]))) + '\\trading-robot\\app\\strategies\\neural_network'
    #     os.chdir(app_directory)

    #     destination_path = os.path.join(app_directory, 'GAN_USDRUB_32x32.zip')
    #     shutil.copy(selected_model_path, destination_path)

    # app_directory = os.path.dirname(os.path.dirname(
    #     os.path.abspath(sys.argv[0]))) + '/robot-api'
    # os.chdir(app_directory)

    # uvicorn_command = 'uvicorn main:app'

    # process = subprocess.Popen(uvicorn_command, shell=True)
    # subprocess.Popen(
    #     ['cmd', '/k', f'cd {app_directory} & venv\Scripts\\activate & uvicorn main:app'])

    # app_directory = os.path.dirname(os.path.dirname(
    #     os.path.abspath(sys.argv[0]))) + '/robot-interface'
    # os.chdir(app_directory)

    # start_command = 'npm start'

    # process = subprocess.Popen(start_command, shell=True)
    # subprocess.Popen(['cmd', '/k', f'cd {app_directory} & npm start'])

    webbrowser.open('http://localhost:3000')
    update_models()


def validate_percentage(value):
    if value.isdigit() or value == '' or '.' in value:
        if value == '':
            value = 0
        if 0 <= float(value) <= 100:
            return True
    return False


def test_model(model_name, root_window, delta):

    app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_directory)

    test_window = tk.Toplevel(root_window)
    test_window.title(f"Тестирование модели {model_name}")
    window_width = 600
    window_height = 650
    screen_width = test_window.winfo_screenwidth()
    screen_height = test_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    test_window.geometry(f"{window_width}x{window_height}+{x}+{y-30}")

    csv_name = model_name[8:14]
    df = pd.read_csv(f'data/{csv_name}.csv', sep=';',
                     parse_dates=[['<DATE>', '<TIME>']])
    columns = ('date_time', 'target')
    df.columns = columns

    current_date = datetime.now()

    # Вычисление даты, отстоящей на два месяца от текущей даты
    two_months_ago = current_date - timedelta(days=60)

    # Фильтрация данных только для последних двух месяцев
    df = df[(df['date_time'] >= two_months_ago)
            & (df['date_time'] <= current_date)]

    fig = Figure(figsize=(5, 3))
    plot = fig.add_subplot(111)
    plot.tick_params(axis='x', rotation=-20)
    plot.plot(range(len(df)), df['target'], label=f"{csv_name}")
    plot.set_xticks(range(len(df)))
    plot.set_xticklabels(df['date_time'].dt.strftime('%Y-%m-%d'))

    num_ticks = 8
    x_locator = ticker.MaxNLocator(num_ticks)
    plot.xaxis.set_major_locator(x_locator)

    plot.tick_params(axis='x', which='major', labelsize=8)
    plot.legend()
    plot.grid()

    canvas = FigureCanvasTkAgg(fig, master=test_window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    canvas.get_tk_widget().configure(width=600, height=350)

    stop_loss_label = tk.Label(test_window, text="Stop Loss (руб.):")
    stop_loss_entry = tk.Entry(test_window)
    stop_loss_entry.insert(0, 0)
    stop_loss_label.pack()
    stop_loss_entry.pack()

    take_profit_label = tk.Label(test_window, text="Take Profit (руб.):")
    take_profit_entry = tk.Entry(test_window)
    take_profit_entry.insert(0, 0)
    take_profit_label.pack()
    take_profit_entry.pack()

    reversal_up_prob_label = tk.Label(
        test_window, text="Мин. вероятность точки разворота вверх (%):")
    reversal_up_prob_entry = tk.Entry(test_window)
    reversal_up_prob_entry.insert(0, 0)
    reversal_up_prob_entry.config(validate="key", validatecommand=(
        test_window.register(validate_percentage), "%P"))
    reversal_up_prob_label.pack()
    reversal_up_prob_entry.pack()

    reversal_down_prob_label = tk.Label(
        test_window, text="Мин. вероятность точки разворота вниз (%):")
    reversal_down_prob_entry = tk.Entry(test_window)
    reversal_down_prob_entry.insert(0, 0)
    reversal_down_prob_entry.config(validate="key", validatecommand=(
        test_window.register(validate_percentage), "%P"))
    reversal_down_prob_label.pack()
    reversal_down_prob_entry.pack()

    buy_size_label = tk.Label(
        test_window, text="Размер покупки целевой валюты (единиц):")
    buy_size_label_entry = tk.Entry(test_window)
    buy_size_label_entry.insert(0, 1)
    buy_size_label.pack()
    buy_size_label_entry.pack()

    start_capital_label = tk.Label(
        test_window, text="Начальный капитал (руб.):")
    start_capital_entry = tk.Entry(test_window)
    start_capital_entry.insert(0, 1)
    start_capital_label.pack()
    start_capital_entry.pack()

    start_button = tk.Button(test_window, text="Начать тест", command=lambda m=model_name, r=test_window: start_test(m, r,
                                                                                                                     stop_loss_entry.get(),
                                                                                                                     take_profit_entry.get(),
                                                                                                                     reversal_up_prob_entry.get(),
                                                                                                                     reversal_down_prob_entry.get(),
                                                                                                                     canvas,
                                                                                                                     buy_size_label_entry.get(),
                                                                                                                     delta,
                                                                                                                     float(start_capital_entry.get())))
    start_button.config(font=("Arial", 12), pady=5)
    start_button.pack()


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def start_test(model_name, root_window, stop_loss, take_profit, reversal_up_prob, reversal_down_prob, canvas, buy_size, delta, start_capital):

    app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_directory)

    MODEL_PATH = f'models/{model_name}'
    model_1 = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model_1 = model_1.eval()

    # Переменная для отслеживания состояния обучения
    testing_stopped = False

    # Функция для остановки обучения
    def stop_testing():
        nonlocal testing_stopped
        testing_stopped = True

    # Создание нового окна для отображения графика
    result_window = tk.Toplevel(root_window)
    result_window.title("Тестирование модели")
    window_width = 830
    window_height = 500
    screen_width = result_window.winfo_screenwidth()
    screen_height = result_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    result_window.geometry(f"{window_width}x{window_height}+{x}+{y-30}")

    stop_button = tk.Button(
        result_window, text="Остановить тестирование", command=stop_testing)
    stop_button.pack()

    # Создание графика и осей
    figure = plt.figure(figsize=(5, 4), dpi=100)
    subplot = figure.add_subplot(1, 1, 1)

    subplot.tick_params(axis='x', rotation=-20)

    num_ticks = 4  # Установите желаемое количество отображаемых дат
    x_locator = ticker.MaxNLocator(num_ticks)
    subplot.xaxis.set_major_locator(x_locator)

    subplot.tick_params(axis='x', which='major', labelsize=8)

    def save_text():
        # Получить текст из текстового поля
        text = result_text.get("1.0", "end-1c")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv")  # Открыть окно "Сохранить как"
        if save_path:
            lines = text.split("\n")  # Разделить текст на строки
            # Разделить каждую строку на значения столбцов
            data = [line.split(",") for line in lines]
            # Создать DataFrame из данных
            df = pd.DataFrame(data, columns=['сумма', 'сигнал', 'вероятность'])
            df.to_csv(save_path, index=False, encoding="ANSI")

    # Создание холста для графика
    canvas = tkagg.FigureCanvasTkAgg(figure, master=result_window)
    canvas.get_tk_widget().pack(side=tk.LEFT, padx=10,
                                pady=10)  # Изменили pack параметры

    result_frame = tk.Frame(result_window)
    result_frame.pack(padx=10, pady=10)

    result_text = Text(result_frame)
    result_text.pack(side=tk.TOP)

    save_button = tk.Button(result_frame, text="Сохранить", command=save_text)
    save_button.pack(side=tk.BOTTOM, pady=10)

    scrollbar = Scrollbar(result_frame)
    scrollbar.pack(side=tk.RIGHT, fill=Y)

    result_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=result_text.yview)

    def scroll_to_bottom():
        result_text.yview_moveto(1.0)

    if model_name[:3] == 'GAN':
        DATA_CSV_PATH = f'{model_name[8:14]}.csv'
        TAKE_PROFIT = float(take_profit)
        STOP_LOSS = float(stop_loss)
        BUY_SIZE = float(buy_size)
        prob_1 = float(reversal_up_prob) * 0.01
        prob_2 = float(reversal_down_prob) * 0.01

        data = parse_data(DATA_CSV_PATH)

        activ_pos = []
        close_pos = []
        capital_usd = []
        quote_currency = start_capital
        base_currency = 0

        for row in data:
            if testing_stopped:
                break

            matrix = GenerateGAF(row['tenzor'], 32)
            tenzor = torch.FloatTensor(matrix)
            predict = model_1(np.reshape(tenzor, (1, 1, 32, 32)))
            _1, signal_1 = torch.max(predict[0].data, 0)

            if signal_1 == 0:
                signal_output = 'neutral'
            if signal_1 == 1:
                signal_output = 'buy    '
            if signal_1 == 2:
                signal_output = 'sell   '

            output_text = f"{toFixed((quote_currency + (base_currency * row['target'])), 4)}руб., {signal_output}, {round(float(_1) * 100, 2)}%"
            result_text.insert(END, output_text + '\n')

            # Scroll to the bottom after inserting each line
            scroll_to_bottom()
            capital_usd.append({
                'date_time': row['date_time'],
                'value': round(quote_currency + (base_currency * row['target']), 5)
            })

            for i in reversed(activ_pos):
                if i['type'] == 'long':
                    if row['target'] - i['target'] >= TAKE_PROFIT or i['target'] - row['target'] >= STOP_LOSS:
                        if base_currency - BUY_SIZE >= 0:
                            base_currency -= BUY_SIZE
                            quote_currency += BUY_SIZE * row['target']

                            if row['target'] - i['target'] >= TAKE_PROFIT:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'take_profit',
                                    'value': round(row['target'] - i['target'], 5) * BUY_SIZE
                                })
                            if i['target'] - row['target'] >= STOP_LOSS:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'stop_loss',
                                    'value': -round((i['target'] - row['target']), 5) * BUY_SIZE
                                })
                            activ_pos.remove(i)
                elif i['type'] == 'short':
                    if i['target'] - row['target'] >= TAKE_PROFIT or row['target'] - i['target'] >= STOP_LOSS:
                        if quote_currency - BUY_SIZE * row['target'] >= 0:
                            base_currency += BUY_SIZE
                            quote_currency -= BUY_SIZE * row['target']
                            if i['target'] - row['target'] >= TAKE_PROFIT:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'take_profit',
                                    'value': round(i['target'] - row['target'], 5) * BUY_SIZE
                                })
                            if row['target'] - i['target'] >= STOP_LOSS:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'stop_loss',
                                    'value': -round((row['target'] - i['target']), 5) * BUY_SIZE
                                })
                            activ_pos.remove(i)

            if (signal_1 == 1) and (_1 > prob_1):
                if quote_currency - BUY_SIZE * row['target'] >= 0:
                    activ_pos.append({
                        'date_time': row['date_time'],
                        'target': row['target'],
                        'type': 'long'
                    })
                    base_currency += BUY_SIZE
                    quote_currency -= BUY_SIZE * row['target']

            elif (signal_1 == 2) and (_1 > prob_2):
                if base_currency - BUY_SIZE >= 0:
                    activ_pos.append({
                        'date_time': row['date_time'],
                        'target': row['target'],
                        'type': 'short'
                    })
                    base_currency -= BUY_SIZE
                    quote_currency += BUY_SIZE * row['target']
                    # Получение данных для построения графика
            dates = [item['date_time'] for item in capital_usd]
            values = [item['value'] for item in capital_usd]

            subplot.clear()

            subplot.plot(dates, values)
            subplot.set_xlabel("Дата")
            subplot.set_ylabel("Значение")

            canvas.draw()
            result_window.update()

        print(f'незакрытых позиций: {len(activ_pos)}')
        print(activ_pos)

    else:
        DATA_CSV_PATH = f'{model_name[8:14]}.csv'
        TAKE_PROFIT = float(take_profit)
        STOP_LOSS = float(stop_loss)
        BUY_SIZE = float(buy_size)
        prob_1 = float(reversal_up_prob) * 0.01
        prob_2 = float(reversal_down_prob) * 0.01

        df_eurusd = pd.read_csv(
            f'data/{DATA_CSV_PATH}', sep=';', parse_dates=[['<DATE>', '<TIME>']])
        columns = ('date_time', 'target')
        df_eurusd.columns = columns

        current_date = datetime.now()

        # Вычисление даты, отстоящей на два месяца от текущей даты
        two_months_ago = current_date - timedelta(days=60)

        # Фильтрация данных только для последних двух месяцев
        df_eurusd = df_eurusd[(df_eurusd['date_time'] >= two_months_ago) & (
            df_eurusd['date_time'] <= current_date)]

        indicator_df_eurusd = indicator(df_eurusd, delta)
        indicator_df_eurusd['class'] = indicator_df_eurusd['class'].replace(
            0, 2)
        binary_indicator_eurusd = indicator_df_eurusd
        indicator_df_eurusd = pd.merge(
            df_eurusd, indicator_df_eurusd, how='left', on='date_time')
        indicator_df_eurusd = indicator_df_eurusd.drop(['target_y'], axis=1)
        indicator_df_eurusd = indicator_df_eurusd.fillna(0)
        indicator_df_eurusd.columns = ('date_time', 'target', 'class')
        indicator_df_eurusd['class'] = indicator_df_eurusd['class'].astype(int)

        indicator_for_test = indicator_df_eurusd.copy()
        indicator_for_test.columns = ('date_time', 'target', 'class')
        for i in range(1, 11):
            indicator_for_test[f'{i}'] = indicator_for_test['target'].shift(-i)
        indicator_for_test = indicator_for_test.dropna()

        X_eurusd_dataset, y_eurusd_dataset, c, date_time, target = windows_conv(
            indicator_for_test, indicator_for_test, 10)

        X_eurusd_dataset = X_eurusd_dataset[c:]
        y_eurusd_dataset = y_eurusd_dataset[c:]

        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        activ_pos = []
        close_pos = []
        capital_usd = []
        quote_currency = start_capital
        base_currency = 0

        for row, dt, targ in zip(X_eurusd_dataset, date_time, target):

            if testing_stopped:
                break

            tenzor = torch.FloatTensor(row)
            predict = model(np.reshape(tenzor, (1, 3, 10, 10)))

            _, signal = torch.max(predict[0].data, 0)

            capital_usd.append({
                'date_time': dt,
                'value': round(quote_currency + (base_currency * targ), 5)
            })

            if signal == 0:
                signal_output = 'neutral'
            if signal == 1:
                signal_output = 'buy'
            if signal == 2:
                signal_output = 'sell'

            output_text = f"{toFixed((quote_currency + (base_currency * targ)), 4)}руб., {signal_output}, {round(float(_) * 100, 4)}"
            result_text.insert(END, output_text + '\n')

            scroll_to_bottom()

            for i in reversed(activ_pos):
                if i['type'] == 'long':
                    if targ - i['target'] >= TAKE_PROFIT or i['target'] - targ >= STOP_LOSS:
                        if base_currency - BUY_SIZE >= 0:
                            base_currency -= BUY_SIZE
                            quote_currency += BUY_SIZE * targ

                            if targ - i['target'] >= TAKE_PROFIT:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'take_profit',
                                    'value': round(targ - i['target'], 5) * BUY_SIZE
                                })
                            if i['target'] - targ >= STOP_LOSS:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'stop_loss',
                                    'value': -round((i['target'] - targ), 5) * BUY_SIZE
                                })
                            activ_pos.remove(i)
                elif i['type'] == 'short':
                    if i['target'] - targ >= TAKE_PROFIT or targ - i['target'] >= STOP_LOSS:
                        if quote_currency - BUY_SIZE * targ >= 0:
                            base_currency += BUY_SIZE
                            quote_currency -= BUY_SIZE * targ
                            if i['target'] - targ >= TAKE_PROFIT:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'take_profit',
                                    'value': round(i['target'] - targ, 5) * BUY_SIZE
                                })
                            if targ - i['target'] >= STOP_LOSS:
                                close_pos.append({
                                    'date_time': i['date_time'],
                                    'open_type': i['type'],
                                    'close_type': 'stop_loss',
                                    'value': -round((targ - i['target']), 5) * BUY_SIZE
                                })
                            activ_pos.remove(i)

            if signal == 1:
                if quote_currency - BUY_SIZE * targ >= 0:
                    activ_pos.append({
                        'date_time': dt,
                        'target': targ,
                        'type': 'long'
                    })
                    base_currency += BUY_SIZE
                    quote_currency -= BUY_SIZE * targ

            elif signal == 2:
                if base_currency - BUY_SIZE >= 0:
                    activ_pos.append({
                        'date_time': dt,
                        'target': targ,
                        'type': 'short'
                    })
                    base_currency -= BUY_SIZE
                    quote_currency += BUY_SIZE * targ
            # Получение данных для построения графика
            dates = [item['date_time'] for item in capital_usd]
            values = [item['value'] for item in capital_usd]

            subplot.clear()

            subplot.plot(dates, values)
            subplot.set_xlabel("Дата")
            subplot.set_ylabel("Значение")

            canvas.draw()
            result_window.update()
        print(f'незакрытых позиций: {len(activ_pos)}')
        print(activ_pos)


def parse_data(csv_path, window_size=32):
    df = pd.read_csv(f'data/{csv_path}', sep=';',
                     parse_dates=[['<DATE>', '<TIME>']])
    df.columns = ('date_time', 'target')

    current_date = datetime.now()

    # Вычисление даты, отстоящей на два месяца от текущей даты
    two_months_ago = current_date - timedelta(days=60)

    # Фильтрация данных только для последних двух месяцев
    df = df[(df['date_time'] >= two_months_ago)
            & (df['date_time'] <= current_date)]

    result = []
    tmp = []

    for i in range(len(df)):
        tmp.append(df.iloc[i]['target'])
        if i >= window_size:
            result.append({
                'date_time': df.iloc[i]['date_time'],
                'target': df.iloc[i]['target'],
                'tenzor': tmp[-window_size:]
            })
            tmp = tmp[1:]
    return result


def windows_conv(indicator, time_series, size):

    indicator = indicator.reset_index(drop=True)
    list_columns = list(time_series.columns)
    list_columns.remove('date_time')
    list_columns.remove('target')
    list_columns.remove('class')
    date_time = []
    target = []

    MODEL_PATH = 'model_for_eurusd_10x10_delta005'
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    all_matrix = np.zeros((len(indicator), 3, size, size))
    all_target = np.zeros(len(indicator))
    c = 0
    for idx, row in indicator.iterrows():

        if time_series[time_series['date_time'] == row['date_time']].index.values.astype(int):
            our_index = time_series[time_series['date_time']
                                    == row['date_time']].index.values.astype(int)[0]
        else:
            continue

        if our_index < size * size:
            c += 1
            # all_matrix = np.delete(all_matrix, [-1], 0)
            continue

        date_time.append(row['date_time'])
        target.append(row['target'])
        batch = time_series.iloc[our_index - size *
                                 size + 1: our_index + 1][list_columns].to_numpy()
        tenzor = torch.FloatTensor(batch)
        predicts = model(tenzor)
        criterion = nn.Softmax(dim=1)
        predicts = criterion(predicts)

        matrix = np.zeros((3, size, size))
        matrix[0] = np.reshape(predicts.detach().numpy().T[0], (size, size))
        matrix[1] = np.reshape(predicts.detach().numpy().T[1], (size, size))
        matrix[2] = np.reshape(predicts.detach().numpy().T[2], (size, size))

        all_matrix[idx] = matrix
        all_target[idx] = row['class']
        print(idx)

    return all_matrix, all_target, c, date_time, target


def GAN_dataset(df_eurusd, delta):
    indicator_df_eurusd = indicator(df_eurusd, delta)

    indicator_df_eurusd['class'] = indicator_df_eurusd['class'].replace(0, 2)
    binary_indicator_eurusd = indicator_df_eurusd
    indicator_df_eurusd = pd.merge(
        df_eurusd, indicator_df_eurusd, how='left', on='date_time')
    indicator_df_eurusd = indicator_df_eurusd.drop(['target_y'], axis=1)
    indicator_df_eurusd = indicator_df_eurusd.fillna(0)
    indicator_df_eurusd.columns = ('date_time', 'target', 'class')
    indicator_df_eurusd['class'] = indicator_df_eurusd['class'].astype(int)

    indices = pd.DataFrame(
        indicator_df_eurusd[indicator_df_eurusd['class'] != 0].index, columns=['current'])
    indices['next'] = indices['current'].shift(-1)
    indices['mean'] = (indices['current'] + indices['next'])/2
    indices = indices.dropna()
    indices = indices.astype(int)

    list_ind = list(indicator_df_eurusd[(indicator_df_eurusd['class'] == 2) | (
        indicator_df_eurusd['class'] == 1)].index)
    list_ind = list_ind + list(indices['mean'])
    list_ind.sort()

    indicator_df_eurusd = indicator_df_eurusd.iloc[list_ind]

    indicator_df_eurusd = indicator_df_eurusd.reset_index(drop=True)

    indicator_df_eurusd = indicator_df_eurusd.drop_duplicates(subset=[
                                                              'date_time'])

    remove_n = int((indicator_df_eurusd[indicator_df_eurusd['class'] == 0]['class'].count(
    ) - indicator_df_eurusd[indicator_df_eurusd['class'] == 1]['class'].count()) / 2)
    drop_indices = np.random.choice(
        indicator_df_eurusd[indicator_df_eurusd['class'] == 0].index, remove_n, replace=False)
    indicator_df_eurusd = indicator_df_eurusd.drop(drop_indices)

    window_size = 32
    eurusd_dataset = pd.DataFrame(
        windows(indicator_df_eurusd, df_eurusd, window_size))
    y_eurusd_dataset = eurusd_dataset[0]
    X_eurusd_dataset = eurusd_dataset.drop([0], axis=1)

    windowSize = len(X_eurusd_dataset.iloc[0])

    ts_img = GenerateGAF(all_ts=list(X_eurusd_dataset.iloc[0]),
                         window_size=windowSize)
    ts_images = ts_img

    for i in range(1, len(X_eurusd_dataset)):
        timeSeries = np.array(X_eurusd_dataset.iloc[i])
        print(i)

        ts_img = GenerateGAF(all_ts=timeSeries,
                             window_size=windowSize)

        ts_images = np.concatenate((ts_images, ts_img))

    X_eurusd_dataset = np.reshape(
        ts_images, (len(ts_images), 1, window_size, window_size))

    y_one_hot = pd.get_dummies(y_eurusd_dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X_eurusd_dataset, y_one_hot, test_size=0.2, shuffle=False, random_state=42)

    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test.to_numpy()).to(device)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train.to_numpy()).to(device)
    return X_train, X_test, y_train, y_test


def MLP_dataset(df_eurusd, delta):
    indicator_df_eurusd = indicator(df_eurusd,  delta)
    indicator_df_eurusd['class'] = indicator_df_eurusd['class'].replace(0, 2)
    binary_indicator_eurusd = indicator_df_eurusd
    indicator_df_eurusd = pd.merge(
        df_eurusd, indicator_df_eurusd, how='left', on='date_time')
    indicator_df_eurusd = indicator_df_eurusd.drop(['target_y'], axis=1)
    indicator_df_eurusd = indicator_df_eurusd.fillna(0)
    indicator_df_eurusd.columns = ('date_time', 'target', 'class')
    indicator_df_eurusd['class'] = indicator_df_eurusd['class'].astype(int)

    indicator_for_test = indicator_df_eurusd.copy()
    indicator_for_test.columns = ('date_time', 'target', 'class')
    for i in range(1, 11):
        indicator_for_test[f'{i}'] = indicator_for_test['target'].shift(-i)
    indicator_for_test = indicator_for_test.dropna()

    indices = pd.DataFrame(
        indicator_df_eurusd[indicator_df_eurusd['class'] != 0].index, columns=['current'])
    indices['next'] = indices['current'].shift(-1)
    indices['mean'] = (indices['current'] + indices['next'])/2
    indices = indices.dropna()
    indices = indices.astype(int)

    list_ind = list(indicator_df_eurusd[(indicator_df_eurusd['class'] == 2) | (
        indicator_df_eurusd['class'] == 1)].index)
    list_ind = list_ind + list(indices['mean'])
    list_ind.sort()

    indicator_df_eurusd = indicator_df_eurusd.iloc[list_ind]
    indicator_df_eurusd = indicator_df_eurusd.reset_index(drop=True)
    indicator_df_eurusd = indicator_df_eurusd.drop_duplicates(subset=[
                                                              'date_time'])

    remove_n = int((indicator_df_eurusd[indicator_df_eurusd['class'] == 0]['class'].count(
    ) - indicator_df_eurusd[indicator_df_eurusd['class'] == 1]['class'].count()) / 2)
    drop_indices = np.random.choice(
        indicator_df_eurusd[indicator_df_eurusd['class'] == 0].index, remove_n, replace=False)
    indicator_df_eurusd = indicator_df_eurusd.drop(drop_indices)

    window_size = 10
    eurusd_dataset = pd.DataFrame(
        windows(indicator_df_eurusd, df_eurusd, window_size))
    y_eurusd_dataset = eurusd_dataset[0]
    X_eurusd_dataset = eurusd_dataset.drop([0], axis=1)

    X_eurusd_dataset, y_eurusd_dataset, c, _, _ = windows_conv(
        indicator_df_eurusd, indicator_for_test, 10)

    X_eurusd_dataset = X_eurusd_dataset[c:]
    y_eurusd_dataset = y_eurusd_dataset[c:]

    y_one_hot = pd.get_dummies(y_eurusd_dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X_eurusd_dataset, y_one_hot, test_size=0.2, shuffle=False, random_state=42)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test.to_numpy()).to(device)

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train.to_numpy()).to(device)

    return X_train, X_test, y_train, y_test


def train_model(currency, dataset, delta, f10_stop=0, f11_stop=0, f12_stop=0, loss_stop=999, accuracy_stop=0):

    app_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.chdir(app_directory)

    csv_name = currency
    print(csv_name)
    df_eurusd = pd.read_csv(
        f'data/{csv_name}.csv', sep=';', parse_dates=[['<DATE>', '<TIME>']])
    columns = ('date_time', 'target')
    df_eurusd.columns = columns

    timestr = time.strftime("%Y%m%d-%H%M%S")
    if dataset == "Граммианское поле":
        X_train, X_test, y_train, y_test = GAN_dataset(df_eurusd, float(delta))
        MODEL_PATH = 'GAN_CNN_model_for_eurusd_32x32_delta005'
        model_name = f'GAN-CNN_{csv_name}_{timestr}'
    else:
        X_train, X_test, y_train, y_test = MLP_dataset(df_eurusd, float(delta))
        MODEL_PATH = 'conv_model_for_eurusd_10x10_delta005'
        model_name = f'MLP-CNN_{csv_name}_{timestr}'

    losses = []
    accuracy = []
    f1_class_0 = []
    f1_class_1 = []
    f1_class_2 = []
    losses_val = []
    accuracy_val = []
    f1_class_0_val = []
    f1_class_1_val = []
    f1_class_2_val = []

    # Переменная для отслеживания состояния обучения
    training_stopped = False

    # Функция для остановки обучения
    def stop_training():
        nonlocal training_stopped
        training_stopped = True

    train_window = tk.Toplevel(root)
    train_window.title("Обучение модели")
    stop_button = tk.Button(
        train_window, text="Остановить обучение", command=stop_training)
    stop_button.config(font=("Arial", 12))
    stop_button.pack()

    window_width = 650
    window_height = 600
    screen_width = train_window.winfo_screenwidth()
    screen_height = train_window.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    train_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    train_label = tk.Label(train_window)

    fig = Figure(figsize=(5, 3))
    plot = fig.add_subplot(111)
    plot.plot(losses[:], label="обучающая")
    plot.plot(losses_val[:], label="контрольная")
    plot.legend()
    plot.grid()

    canvas = FigureCanvasTkAgg(fig, master=train_window)
    canvas.draw()

    canvas.get_tk_widget().pack()

    model = torch.load(MODEL_PATH, map_location=torch.device(device))
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(200, 200)
    model.fc3 = nn.Linear(200, 3)

    conv = model
    criterion = nn.CrossEntropyLoss()
    criterion_val = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(conv.parameters(), lr=0.002)

    if X_train.shape[0] < 100:
        part = 4
    elif (X_train.shape[0] >= 100) and (X_train.shape[0] < 300):
        part = 8
    elif (X_train.shape[0] >= 300) and (X_train.shape[0] < 1000):
        part = 16
    else:
        part = 32

    for i in range(0, 300):
        for x_batch, y_batch in zip(torch.tensor_split(X_train, part), torch.tensor_split(y_train, part)):
            if training_stopped == True:
                break
            conv = conv.train()
            outputs = conv(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            conv = conv.eval()
            outputs_val = conv(X_test)
            loss_val = criterion_val(outputs_val, y_test)

            correct = 0
            total = 0
            _, predicted = torch.max(outputs_val.data, 1)
            _, y_test_value = torch.max(y_test.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test_value).sum().item()
            accuracy_val.append((correct / total) * 100)
            losses_val.append(float(loss_val))
            f1_class_0_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[0])
            f1_class_1_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[1])
            f1_class_2_val.append(
                f1_score(y_test_value.cpu(), predicted.cpu(), average=None)[2])

            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            _, y_batch_value = torch.max(y_batch.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch_value).sum().item()
            accuracy.append((correct / total) * 100)
            losses.append(float(loss))

            f1_class_0.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[0])
            f1_class_1.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[1])
            f1_class_2.append(f1_score(y_batch_value.cpu(),
                              predicted.cpu(), average=None)[2])

            if f10_stop == 0 and f11_stop == 0 and f12_stop == 0 and loss_stop == 999 and accuracy_stop == 0:
                if (losses_val[-1] == np.array(losses_val).min()):
                    f10 = f1_class_0_val[-1]
                    f11 = f1_class_1_val[-1]
                    f12 = f1_class_2_val[-1]
                    model_path_for_save = f"models/{model_name}"
                    torch.save(conv, model_path_for_save)
            else:
                if f1_class_0_val[-1] >= f10_stop and f1_class_1_val[-1] >= f11_stop \
                        and f1_class_2_val[-1] >= f12_stop and losses_val[-1] <= loss_stop and accuracy_val[-1] >= accuracy_stop:
                    f10 = f1_class_0_val[-1]
                    f11 = f1_class_1_val[-1]
                    f12 = f1_class_2_val[-1]
                    model_path_for_save = f"models/{model_name}"
                    torch.save(conv, model_path_for_save)
                    stop_training()
                elif (losses_val[-1] == np.array(losses_val).min()):
                    f10 = f1_class_0_val[-1]
                    f11 = f1_class_1_val[-1]
                    f12 = f1_class_2_val[-1]
                    model_path_for_save = f"models/{model_name}"
                    torch.save(conv, model_path_for_save)

            train_text = f'''кол-во эпох: {i}
Средняя точность на обучающей выборке за последние 10 итераций {round(np.array(accuracy)[-10:-1].mean(), 4)}
Средняя точность на контрольной выборке за последние 10 итераций {round(np.array(accuracy_val)[-10:-1].mean(), 4)}
минимальная кросс-энтропийная ошибка на обучающей выборке {round(np.array(losses).min(), 4)}
минимальный кросс-энтропийная ошибка на контрольной выборке {round(np.array(losses_val).min(), 4)}
f1-мера на обучающей выборке для 
    нейтральных точек: {round(f1_class_0[-1], 4)}, 
    точек разворота вверх: {round(f1_class_1[-1], 4)}, 
    точек разворота вниз: {round(f1_class_2[-1], 4)}
f1-мера на контрольной выборке для 
    нейтральных точек: {round(f1_class_0_val[-1], 4)}, 
    точек разворота вверх: {round(f1_class_1_val[-1], 4)}, 
    точек разворота вниз: {round(f1_class_2_val[-1], 4)}
максимальная точность на обучающей выборке {round(np.array(accuracy).max(), 4)}%
максимальная точность на контрольной выборке {round(np.array(accuracy_val).max(), 4)}% \n'''

            train_label.config(text=train_text, font=(
                'Courier', 10), justify='left')
            train_label.pack()

            train_window.update()

            plot.clear()
            plot.plot(losses[:], label="обучающая")
            plot.plot(losses_val[:], label="контрольная")
            plot.legend()
            plot.grid()
            canvas.draw()

    max_accuracy = round(np.array(accuracy_val).max(), 4)
    min_loss = round(np.array(losses_val).min(), 4)
    c.execute("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?)",
              (model_name, max_accuracy, min_loss, f10, f11, f12, delta))
    conn.commit()
    update_models()


def update_models():
    for widget in root.winfo_children():
        widget.destroy()
    load_models()


root = tk.Tk()
load_models()
root.mainloop()
conn.close()
