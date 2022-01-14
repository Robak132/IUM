from pandas import DataFrame
import torch
import torch.nn as nn

from microservice.models import ModelInterface
from microservice.features import extract_users_data, prepare_data_for_predict


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
        x = torch.cat([x, cat_x_embedded], dim=1)
        activation1 = self.act_1(self.bn1(self.layer1(x)))
        activation1 = self.d1(activation1)
        activation2 = self.act_2(self.layer2(activation1))
        activation2 = self.d1(activation2)
        output = self.layer3(activation2)
        return output


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Linear(7, 7)
        self.act_emb = nn.Tanh()
        self.layer1 = nn.Linear(6 + 7, 100)
        self.act_1 = nn.LeakyReLU()
        self.layer2 = nn.Linear(100, 40)
        self.act_2 = nn.LeakyReLU()
        self.layer3 = nn.Linear(40, 1)

    def forward(self, x, cat_x):
        cat_x_embedded = self.emb_layer(cat_x)
        cat_x_embedded = self.act_emb(cat_x_embedded)
        x = torch.cat([x, cat_x_embedded], dim=1)
        activation1 = self.act_1(self.layer1(x))
        activation2 = self.act_2(self.layer2(activation1))
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
        extracted_users_data = extract_users_data(sessions, users, products)
        x, cat_x = prepare_data_for_predict(extracted_users_data)
        x = torch.from_numpy(x.values).float()
        cat_x = torch.from_numpy(cat_x.values).float()
        self.net.eval()
        out = self.net(x, cat_x).squeeze()
        out = out.detach().numpy()
        return {f"{int(user_id)}": out[i] for i, user_id in enumerate(extracted_users_data["user_id"].to_list())}

    def load_model(self, string: str):
        self.net.load_state_dict(torch.load(string))

    def save_model(self, string: str):
        torch.save(self.net.state_dict(), string)