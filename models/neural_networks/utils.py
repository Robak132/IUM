import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as data


def prepare_data_for_predict(extracted_users_data):
    categorical_columns = ['city']
    categorical_values = pd.get_dummies(extracted_users_data[categorical_columns])
    numeric_values = extracted_users_data.drop(columns=['user_id', *categorical_columns])
    return numeric_values, categorical_values


def prepare_data_for_train(extracted_users_data, targets):
    extracted_users_data = pd.merge(extracted_users_data, targets, on="user_id")
    targets = extracted_users_data['expenses_y']
    numerical_values, categorical_values = prepare_data_for_predict(extracted_users_data)
    numerical_values = numerical_values.drop(columns=['expenses_y'])
    return numerical_values, categorical_values, targets


def load_into_train_val(ratio, numerical_values, categorical_values, objectives):
    train_indices = np.random.rand(len(numerical_values)) > ratio
    numerical_data = torch.from_numpy(numerical_values.values[train_indices]).float()
    categorical_data = torch.from_numpy(categorical_values.values[train_indices]).float()
    targets = torch.from_numpy(objectives.values[train_indices]).float()

    val_numerical_data = torch.from_numpy(numerical_values.values[~train_indices]).float()
    val_categorical_data = torch.from_numpy(categorical_values.values[~train_indices]).float()
    val_targets = torch.from_numpy(objectives.values[~train_indices]).float()

    train_dataset = data.TensorDataset(numerical_data, categorical_data, targets)
    val_dataset = data.TensorDataset(val_numerical_data, val_categorical_data, val_targets)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_targets), shuffle=False)

    return train_loader, val_loader


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