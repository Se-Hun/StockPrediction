import torch
from torch.utils.data import TensorDataset

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from lstm.modeling_lstm import LSTM
from utils import load_model

def run_train(input_data, to_model, hps):
    # data load
    train_data = pd.read_csv(input_data)

    # pre-processing
    train_data = train_data.set_index('date')
    train_data_index = train_data.index
    column_name = list(train_data)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)

    train_data = pd.DataFrame(train_data, columns=column_name, index=train_data_index)

    # deciding data column for domain
    if hps.domain == 'close':
        train_data = pd.DataFrame(train_data, columns=['close'], index=train_data_index)

    # creating window for lstm
    if hps.model_type == 'lstm':
        window_size = hps.window_size

        for s in range(1, window_size+1):
            train_data['close_{}'.format(s)] = train_data['close'].shift(s)

        X_train = train_data.dropna().drop('close', axis=1)
        y_train = train_data.dropna()[['close']]

        X_train = X_train.values

        y_train = y_train.values

        X_train = X_train.reshape(X_train.shape[0], window_size, 1)

        print("------------------ LSTM Data ------------------")
        print("Num Examples : {}".format(X_train.shape[0]))
        print("Window Size : {}".format(X_train.shape[1]))
        print("-----------------------------------------------")

        model = LSTM(input_dim=1, hidden_dim=32, output_size=1, num_layer=2, hps=hps)

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
    test_data_index = test_data.index
    column_name = list(test_data)

    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(test_data)

    test_data = pd.DataFrame(test_data, columns=column_name, index=test_data_index)

    # deciding data column for domain
    if hps.domain == 'close':
        test_data = pd.DataFrame(test_data, columns=['close'], index=test_data_index)

    # creating window for lstm
    if hps.model_type == 'lstm':
        window_size = hps.window_size

        for s in range(1, window_size+1):
            test_data['close_{}'.format(s)] = test_data['close'].shift(s)

        X_test = test_data.dropna().drop('close', axis=1)
        y_test = test_data.dropna()[['close']]

        X_test = X_test.values
        y_test = y_test.values

        X_test = X_test.reshape(X_test.shape[0], window_size, 1)

        print("------------------ LSTM Data ------------------")
        print("Num Examples : {}".format(X_test.shape[0]))
        print("Window Size : {}".format(X_test.shape[1]))
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

    eval(test_dataset, model, hps, batch_size)

    return None
