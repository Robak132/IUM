import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                               worker_init_fn=lambda _: np.random.seed(seed))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False,
                                             worker_init_fn=lambda _: np.random.seed(seed))

    return train_loader, val_loader