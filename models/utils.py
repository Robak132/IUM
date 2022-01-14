import pandas as pd


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