import math

from pandas import DataFrame
import torch
from features.build_features import aggregate_users_data
from microservice.model import ModelInterface
from models import SimpleRegressor

from models.neural_networks.utils import prepare_data_for_predict


class ModelB(ModelInterface):
    def __init__(self):
        self.net = SimpleRegressor()

    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        extracted_users_data = aggregate_users_data(sessions, users, products)
        x, cat_x = prepare_data_for_predict(extracted_users_data)
        x = torch.from_numpy(x.values).float()
        cat_x = torch.from_numpy(cat_x.values).float()
        self.net.eval()
        out = self.net(x, cat_x).squeeze()
        out = out.detach().numpy()
        _dict = {}
        for i, user_id in enumerate(extracted_users_data["user_id"].to_list()):
            if math.isnan(out[i]):
                _dict[user_id] = 0
            else:
                _dict[user_id] = round(max(float(out[i]), 0), 2)
        return _dict

    def load_model(self, string):
        self.net.load_state_dict(torch.load(string))

    def save_model(self, string):
        torch.save(self.net.state_dict(), string)
