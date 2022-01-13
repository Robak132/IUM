from pandas import DataFrame
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random

from client_analiser.models import ModelInterface

seed = 213769420
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class NeuralNetworkRegressor(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Linear(7, 7)
        self.act_emb = nn.Tanh()
        self.layer1 = nn.Linear(6 + 7, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.act_1 = nn.LeakyReLU()
        self.d1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(100, 40)
        self.bn2 = nn.BatchNorm1d(40)
        self.act_2 = nn.LeakyReLU()
        self.d2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.act_emb(cat_x_embedded)
        x = torch.cat([x,cat_x_embedded],dim=1)
        activation1 = self.act_1(self.bn1(self.layer1(x)))
        activation1 = self.d1(activation1)
        activation2 = self.act_2(self.layer2(activation1))
        activation2 = self.d1(activation2)
        output = self.layer3(activation2)
        return output


class ModelB(ModelInterface):

    def __init__(self):
        self.net = NeuralNetworkRegressor()

    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        extracted_users_data = self.extract_users_data(sessions, users, products)
        x, cat_x = self.prepare_data_for_test(extracted_users_data)
        x = torch.from_numpy(x.values).float()
        cat_x = torch.from_numpy(cat_x.values).float()
        self.net.eval()
        out = self.net(x, cat_x).squeeze()
        out = out.detach().numpy()
        _dict = {user_id: round(float(out[i]), 2) for i, user_id in enumerate(extracted_users_data["user_id"].to_list())}
        return _dict

    def get_user_id_from_session(self, session):
        sample_user_id = session['user_id'].iloc[0]
        for user_id in session['user_id']:
            if sample_user_id != user_id:
                raise Exception("How it is even possible")
        return sample_user_id

    def get_user_information(self, user_session_data):
        d = {
            'user_id': [self.get_user_id_from_session(user_session_data)],
            'expenses': [user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()],
            'products_bought': [len(user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"])],
            'events_number': [len(user_session_data)],
            'sessions_number': [len(user_session_data['session_id'].unique())],
            'average_discount': [user_session_data['offered_discount'].mean()],
            'average_discount_on_bought': [
                user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['offered_discount'].mean()
            ]
        }
        if pd.isna(d['average_discount_on_bought']):
            d['average_discount_on_bought'] = 0
        df = pd.DataFrame(data=d)
        return df.set_index('user_id')

    def extract_users_data(self, sessions_data, users_data, products_data):
        enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
        extracted_users = []
        for user_id in enriched_sessions_data['user_id'].unique():
            extracted_users.append(
                self.get_user_information(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]))
        enriched_users_data = pd.concat(extracted_users)
        return pd.merge(enriched_users_data, users_data, on="user_id").drop(columns=['name', 'street'])

    def prepare_data_for_test(self, extracted_users_data):
        categorical_columns = ['city']
        categorical_values = pd.get_dummies(extracted_users_data[categorical_columns])
        numeric_values = extracted_users_data.drop(columns=['user_id', *categorical_columns])
        return numeric_values, categorical_values

    def prepare_data_for_train(self, extracted_users_data, targets):
        extracted_users_data = pd.merge(extracted_users_data, targets, on="user_id")
        targets = extracted_users_data['expenses_y']
        numerical_values, categorical_values = self.prepare_data_for_test(extracted_users_data)
        numerical_values = numerical_values.drop(columns=['expenses_y'])
        return numerical_values, categorical_values, targets

    def load_into_train_val(self, ratio, numerical_values, categorical_values, objectives):
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

    def train_and_extract(self, sessions_data, users_data, products_data, targets):
        self.train(self.extract_users_data(sessions_data, users_data, products_data), targets)

    def train(self, x, y):
        extracted_data = x
        numeric_data, categorical_data, targets = self.prepare_data_for_train(extracted_data, y)
        train_loader, val_loader = self.load_into_train_val(0.3, numeric_data, categorical_data, targets)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=0.001)

        iters = []
        losses = []
        train_acc = []
        val_acc = []
        val_roc = []
        for n in range(100):
            epoch_losses = []
            for x, cat_x, labels in iter(train_loader):
                x, cat_x, labels = x, cat_x, labels
                #         x, cat_x, labels = x.to(device), cat_x.to(device), labels.to(device)
                self.net.train()
                out = self.net(x, cat_x).squeeze()

                loss = criterion(out, labels)
                loss.backward()
                epoch_losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

            loss_mean = np.array(epoch_losses).mean()
            iters.append(n)
            losses.append(loss_mean)
            # vtest_acc = get_accuracy(self.net, val_loader, threshold)
            # roc_value = get_roc(model, val_numerical_data, val_categorical_data, val_targets)
            print(f"Epoch {n} loss {loss_mean:.3}")
            # train_acc.append(get_accuracy(model, train_loader, threshold))  # compute training accuracy
            # val_acc.append(vtest_acc)  # compute validation accuracy
            # val_roc.append(roc_value)


        # print("Final Training Accuracy: {}".format(train_acc[-1]))
        # print("Final Validation Accuracy: {}".format(val_acc[-1]))
