import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset

from lstm.modeling_lstm import LSTM
from utils import load_model

def create_window(input, seq_length):
    data_raw = input.values
    # data_index = input.index
    # data_index = data_index[window_size:-1]
    data = []

    # create all possible sequences of window size
    for index in range(len(data_raw) - seq_length):
        data.append(data_raw[index: index + seq_length])

    data = np.array(data)

    x_data = data[:, :-1, :]
    y_data = data[:, -1, :]

    return x_data, y_data

def run_train(input_data, to_model, hps):
    # data load
    train_data = pd.read_csv(input_data)

    # pre-processing
    train_data = train_data.set_index('date')
    # train_data_index = train_data.index
    # column_name = list(train_data)

    # scaler = MinMaxScaler()
    # train_data = scaler.fit_transform(train_data)

    # train_data = pd.DataFrame(train_data, columns=column_name, index=train_data_index)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    if hps.target == 'close':
        train_data = train_data[['close']]
        train_data['close'] = scaler.fit_transform(train_data['close'].values.reshape(-1, 1))

        # train_data = pd.DataFrame(train_data, columns=['close'], index=train_data_index)

    # creating window for lstm
    if hps.model_type == 'lstm':
        window_size = hps.window_size

        X_train, y_train = create_window(train_data, window_size+1)

        # print(X_train)

        # for s in range(1, window_size+1):
        #     train_data['close_{}'.format(s)] = train_data['close'].shift(s)
        #
        # X_train = train_data.dropna().drop('close', axis=1)
        # y_train = train_data.dropna()[['close']]
        #
        # X_train = X_train.values
        #
        # y_train = y_train.values
        #
        # X_train = X_train.reshape(X_train.shape[0], window_size, 1)

        print("------------------ LSTM Data ------------------")
        print("Num Examples : {}".format(X_train.shape[0]))
        print("X Train Window Size : {}".format(X_train.shape[1]))
        print("-----------------------------------------------")

        model = LSTM(input_dim=1, hidden_size=32, output_size=1, num_layer=2, hps=hps)

    # not implement for another model
    else:
        pass

    # numpy to tensor
    X_train = torch.from_numpy(X_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)

    # For Testing
    # print(X_train.shape)  # (data_num, window_size(sequence_length), 1)
    # print(y_train.shape)  # (data_num, 1)

    if hps.model_type == "lstm":
        from lstm.train import train
    else:
        pass

    # Train DataSet
    train_dataset = TensorDataset(X_train, y_train)

    train(train_dataset, model, hps, to_model)


def run_eval(input_data, from_model, batch_size):
    model = load_model(from_model)
    hps = model.hps

    # data load
    test_data = pd.read_csv(input_data)

    # pre-processing
    test_data = test_data.set_index('date')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if hps.target == 'close':
        test_data = test_data[['close']]
        test_data['close'] = scaler.fit_transform(test_data['close'].values.reshape(-1, 1))

    # creating window for lstm
    if hps.model_type == 'lstm':
        window_size = hps.window_size

        X_test, y_test = create_window(test_data, window_size+1)

        print("------------------ LSTM Data ------------------")
        print("Num Examples : {}".format(X_test.shape[0]))
        print("X Test Window Size : {}".format(X_test.shape[1]))
        print("-----------------------------------------------")

    # not implement for another model
    else:
        pass

    # numpy to tensor
    X_test = torch.from_numpy(X_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    if hps.model_type == "lstm":
        from lstm.eval import eval
    else:
        pass

    # Test DataSet
    test_dataset = TensorDataset(X_test, y_test)

    y_preds = eval(test_dataset, model, batch_size)
    y_test = y_test.detach().numpy()

    y_preds = scaler.inverse_transform(y_preds)
    y_test = scaler.inverse_transform(y_test)

    result_to_dict = {'label' : [y[0] for y in y_test], 'pred' : [y[0] for y in y_preds]}
    predict_results = pd.DataFrame(result_to_dict)
    print(predict_results.head())

    plt.plot(y_test, label="Original")
    plt.plot(y_preds, label="Predict")
    plt.legend()
    plt.show()
    # print(y_test)

    return None
