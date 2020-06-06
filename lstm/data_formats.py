from torch.utils.data import Dataset

class StockDataSet(Dataset):
    """A single set of features of data."""

    def __init__(self, input, label=None):
        self.input = input
        self.label = label

    def __len__(self):
        return len(self.input)

    def __getitem__(self, i):
        return (self.input[i], self.label[i])