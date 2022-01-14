import pandas as pd


def load_default_data(iteration_path: str = "iteration_3/"):
    deliveries_path = "../data/" + iteration_path + "raw/deliveries.jsonl"
    products_path = "../data/" + iteration_path + "raw/products.jsonl"
    sessions_path = "../data/" + iteration_path + "raw/sessions.jsonl"
    users_path = "../data/" + iteration_path + "raw/users.jsonl"

    deliveries_data = pd.read_json(deliveries_path, lines=True)
    products_data = pd.read_json(products_path, lines=True)
    sessions_data = pd.read_json(sessions_path, lines=True)
    users_data = pd.read_json(users_path, lines=True)

    sessions_data = sessions_data.sort_values(by=['timestamp'])
    sessions_data['timestamp_month'] = sessions_data['timestamp'].apply(lambda x: x.month)

    train_data = sessions_data[sessions_data.timestamp_month < 12]
    test_data = sessions_data[sessions_data.timestamp_month >= 12]
    return train_data, test_data, products_data, users_data, deliveries_data
