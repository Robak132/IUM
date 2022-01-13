import pandas as pd
import matplotlib.pyplot as plt
from client_analiser.models import ModelInterface, ModelA, ModelB


def get_user_id_from_session(session):
    sample_user_id = session['user_id'].iloc[0]
    for user_id in session['user_id']:
        if sample_user_id != user_id:
            raise Exception("How it is even possible")
    return sample_user_id


def get_user_expenses(user_session_data):
    d = {
        'user_id': get_user_id_from_session(user_session_data),
        'expenses': user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()
    }
    # df = pd.DataFrame(data=d)
    return d


def calculate_expenses(sessions_data, products_data, users_data):
    enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
    user_expenses = []
    for user_id in range(users_data['user_id'].min(), users_data['user_id'].max() + 1):
        # for user_id in enriched_sessions_data['user_id'].unique():
        user_session_data = enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]
        user_expenses.append(
            {
                'user_id': user_id,
                'expenses': user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()
            }
        )
        # user_expenses.append(get_user_expenses(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]))

    return pd.DataFrame(data=user_expenses)


def loss(predictions, observations):
    unified_data = pd.merge(predictions, observations, on="user_id").sort_values(by=['user_id'])
    unified_data['difference'] = unified_data['user_expenses'] - unified_data['expenses']
    unified_data['difference_square'] = unified_data['difference'].apply(lambda x: x ** 2)
    return unified_data


if __name__ == "__main__":
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
    # %%
    sessions_data = sessions_data.sort_values(by=['timestamp'])
    # sessions_data['timestamp_date'] = sessions_data['timestamp'].apply(lambda x: x.date())
    sessions_data['timestamp_week'] = sessions_data['timestamp'].apply(lambda x: x.week)
    sessions_data['timestamp_month'] = sessions_data['timestamp'].apply(lambda x: x.month)
    sessions_data['timestamp_quarter'] = sessions_data['timestamp'].apply(lambda x: x.quarter)
    # %%
    train_data = sessions_data[sessions_data.timestamp_month < 12]
    test_data = sessions_data[sessions_data.timestamp_month >= 12]
    # %%
    # Models
    model_A: ModelInterface = ModelA()
    model_B: ModelInterface = ModelB()

    observations = calculate_expenses(test_data, products_data, users_data)
    # %%
    # print(model_B.prepare_data(model_B.extract_users_data(sessions_data, users_data, products_data)))
    # model_B.train_and_extract(sessions_data, users_data, products_data, observations)
    # model_B.save_model("../models/model_b_v1")
    print(model_B.predict_expenses(products_data, deliveries_data, train_data, users_data))
    # observations = calculate_expenses(test_data, products_data, users_data)
































































































