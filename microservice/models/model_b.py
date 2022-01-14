from pandas import DataFrame
import torch
from features.build_features import aggregate_users_data
from microservice.models import ModelInterface
from models.neural_networks.NeuralNetworkRegressor import NeuralNetworkRegressor
from models.neural_networks.utils import prepare_data_for_predict


class ModelB(ModelInterface):
    def __init__(self):
        self.net = NeuralNetworkRegressor()

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
        _dict = {user_id: round(float(out[i]), 2) for i, user_id in enumerate(extracted_users_data["user_id"].to_list())}
        return _dict

    def load_model(self, string):
        self.net.load_state_dict(torch.load(string))

    def save_model(self, string):
        torch.save(self.net.state_dict(), string)
