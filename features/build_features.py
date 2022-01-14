import pandas as pd


def get_user_information(user_session_data):
    d = {
        'user_id': [get_user_id_from_session(user_session_data)],
        'expenses': [user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()],
        'products_bought': [len(user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"])],
        'events_number': [len(user_session_data)],
        'sessions_number': [len(user_session_data['session_id'].unique())],
        'average_discount': [user_session_data['offered_discount'].mean()],
        'average_discount_on_bought': [
            user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['offered_discount'].mean()],
        'age': [user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['timestamp'].max().value/10**15 -
                user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['timestamp'].min().value/10**15]
    }
    if pd.isna(d['average_discount_on_bought']):
        d['average_discount_on_bought'] = 0
    df = pd.DataFrame(data=d)
    return df.set_index('user_id')


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
    return d


def aggregate_users_data(sessions_data, users_data, products_data):
    enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
    extracted_users = []
    for user_id in enriched_sessions_data['user_id'].unique():
        extracted_users.append(
            get_user_information(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]))
    enriched_users_data = pd.concat(extracted_users)
    return pd.merge(enriched_users_data, users_data, on="user_id").drop(columns=['name', 'street'])


def calculate_expenses(sessions_data, products_data, users_data):
    enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
    user_expenses = []
    for user_id in range(users_data['user_id'].min(), users_data['user_id'].max() + 1):
        user_session_data = enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]
        user_expenses.append(
            {
                'user_id': user_id,
                'expenses': user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()
            }
        )
    return pd.DataFrame(data=user_expenses)


def calculate_expenses_with_interval(user_session_data, min_interval_value, max_interval_value):
    expenses = []
    for interval in range(min_interval_value, max_interval_value + 1):
        d = {
            "interval_number": interval,
            "expenses": user_session_data[(user_session_data['timestamp_interval'] == interval) & (
                        user_session_data['event_type'] == "BUY_PRODUCT")]['price'].sum()}
        expenses.append(d)
    df = pd.DataFrame(data=expenses)
    return df


def extract_time_series(sessions_data, products_data):
    enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
    users_time_series = []
    min_value = sessions_data['timestamp_interval'].min()
    max_value = sessions_data['timestamp_interval'].max()
    for user_id in enriched_sessions_data['user_id'].unique():
        users_time_series.append(
            {
                "user_id": user_id,
                "expenses": calculate_expenses_with_interval(
                    enriched_sessions_data[enriched_sessions_data['user_id'] == user_id], min_value, max_value)
            }
        )
    return users_time_series
