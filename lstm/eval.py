from tqdm.auto import tqdm
import math

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
    output_list = []
    for inputs in tqdm(test_dataloader, desc="Evaluation"):

        for idx in range(len(inputs)):
            inputs[idx] = inputs[idx].to(device)

        with torch.no_grad():
            x_features = inputs[0]
            labels = inputs[1]

            outputs = model(x_features)
            # print("------------")
            # print(outputs)
            # print("------------")
            for output in outputs:
                output_list.append(output.cpu().detach().numpy())
            # output_list.append(outputs.cpu().detach().numpy())

            loss = criterion(outputs, labels)

            eval_losses.append(loss.item())

    test_score = math.sqrt(np.mean(eval_losses))

    print('Test Score: %.2f RMSE' % (test_score))

    return np.array(output_list)