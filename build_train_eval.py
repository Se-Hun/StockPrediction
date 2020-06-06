import os

import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split

def main():
    data_root = "./data"
    data_path = os.path.join(data_root, "data_2004-04-08_2020-06-05.csv")
    data = pd.read_csv(data_path)

    split_index = 3507 # 2018-06-05

    train_data = data.loc[:split_index]
    test_data = data.loc[split_index+1:]

    train_data.index = [i for i in range(len(train_data.index))]
    test_data.index = [i for i in range(len(test_data.index))]

    train_path = os.path.join(data_root, "train.csv")
    test_path = os.path.join(data_root, "test.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # For Testing
    # print("train ------------")
    # print(train_data)

    # print("test -------------")
    # print(test_data)

    return None

if __name__ == "__main__":
    main()