from tqdm.auto import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

def eval(test_dataset, model, batch_size):
    # use gpu ?
    device = torch.device("cpu")  # default
    if torch.cuda.is_available(): device = torch.device("cuda")
    model.to(device)

    # Data Loader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # Evaluating !
    model.eval()
    eval_losses = []
    for inputs in tqdm(test_dataloader, desc="Evaluation"):

        for idx in range(len(inputs)):
            inputs[idx] = inputs[idx].to(device)

        with torch.no_grad():
            x_features = inputs[0]
            label = inputs[1]

            outputs = model(x_features)
            loss = criterion(outputs, label)

            eval_losses.append(loss.item())

    mean_loss = np.mean(eval_losses)

    print("MSE Loss : {}".format(mean_loss))