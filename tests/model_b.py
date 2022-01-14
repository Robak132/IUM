import unittest
import pandas as pd

from microservice.features import extract_users_data
from microservice.models import ModelInterface, ModelB
from models.train_utils_nn import train
from models.utils import calculate_expenses


class ModelBTests(unittest.TestCase):
    def test_load_save(self):
        iteration_path = "iteration_3/"
        deliveries_path = "../data/" + iteration_path + "raw/deliveries.jsonl"
        products_path = "../data/" + iteration_path + "raw/products.jsonl"
        sessions_path = "../data/" + iteration_path + "raw/sessions.jsonl"
        users_path = "../data/" + iteration_path + "raw/users.jsonl"
        # %%
        deliveries_data = pd.read_json(deliveries_path, lines=True)
        products_data = pd.read_json(products_path, lines=True)
        sessions_data = pd.read_json(sessions_path, lines=True)
        users_data = pd.read_json(users_path, lines=True)

        sessions_data = sessions_data.sort_values(by=['timestamp'])
        sessions_data['timestamp_month'] = sessions_data['timestamp'].apply(lambda x: x.month)

        train_data = sessions_data[sessions_data.timestamp_month < 12]
        test_data = sessions_data[sessions_data.timestamp_month >= 12]

        model_NN_v1: ModelInterface = ModelB()
        model_NN_v2: ModelInterface = ModelB()

        observations = calculate_expenses(test_data, products_data, users_data)

        train(model_NN_v1.net, extract_users_data(sessions_data, users_data, products_data), observations)
        model_NN_v1.save_model("../models/model_nn_v1")
        model_NN_v2.load_model("../models/model_nn_v1")

        self.assertEqual(model_NN_v1.predict_expenses(products_data, deliveries_data, train_data, users_data), model_NN_v2.predict_expenses(products_data, deliveries_data, train_data, users_data))  # add assertion here


if __name__ == '__main__':
    unittest.main()