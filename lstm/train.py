from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

def train(train_dataset, model, hps, to_model):
    # batch and epoch
    batch_size = hps.per_gpu_train_batch_size
    epoch_num = hps.num_train_epochs

    # use gpu if set
    device = torch.device("cpu")  # default
    if torch.cuda.is_available():
        if hps.use_gpu:
            device = torch.device("cuda")
    if device.type == 'cuda': model = model.cuda()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.learning_rate)

    # Train !
    logging_steps = 20
    tr_loss = 0.0
    logging_loss = 0.0

    model.zero_grad()

    # model.hidden = model.init_hidden()

    for epoch in range(int(epoch_num)):
        epoch_iterator = tqdm(train_dataloader, desc="Train")

        for step, inputs in enumerate(epoch_iterator):
            model.train()

            for idx in range(len(inputs)):
                inputs[idx] = inputs[idx].to(device)

            x_features = inputs[0]
            label = inputs[1]

            model.hidden = model.init_hidden(x_features.size()[1])

            # model.hidden = model.init_hidden()

            outputs = model(x_features)
            loss = criterion(outputs, label)

            loss.backward()

            optimizer.step()

            model.zero_grad()

            tr_loss += loss.item()

            if (step % logging_steps == 0) and step != 0:
                loss_scalar = (tr_loss - logging_loss) / logging_steps
                logging_loss = tr_loss

                print("\nEpoch : {} Step : {} Loss : {} ".format(epoch + 1, step, loss_scalar))

    # save model
    import os
    to_model_fn = os.path.abspath(to_model)
    torch.save(model, to_model_fn)
    print("Model saved at {}".format(to_model_fn))

