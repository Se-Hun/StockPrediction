import os

import pandas as pd

def main():
    data_root = "./data"
    data_path = os.path.join(data_root, "samsung.csv")
    data = pd.read_csv(data_path)

    split_index = 3593 # samsung -> 2016-04-27
    # split_index = 618 # celtrion -> 2020-02-07
    # split_index = 3507 # 2018-06-05

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