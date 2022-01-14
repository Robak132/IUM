import torch.nn as nn
import torch.optim as optim
import numpy as np

from microservice.features import prepare_data_for_train, load_into_train_val


def train(model, x, y):
    numeric_data, categorical_data, targets = prepare_data_for_train(x, y)
    train_loader, val_loader = load_into_train_val(0.3, numeric_data, categorical_data, targets)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    iters = []
    losses = []
    for n in range(100):
        epoch_losses = []
        for x, cat_x, labels in iter(train_loader):
            x, cat_x, labels = x, cat_x, labels
            model.train()
            out = model(x, cat_x).squeeze()

            loss = criterion(out, labels)
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        loss_mean = np.array(epoch_losses).mean()
        iters.append(n)
        losses.append(loss_mean)
        print(f"Epoch {n} loss {loss_mean:.3}")