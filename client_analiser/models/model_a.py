import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression


def calculate_expenses_with_interval(user_session_data, min_interval_value, max_interval_value):
    expenses=[]
    for interval in range (min_interval_value, max_interval_value + 1):
        d = {
            "interval_number": interval,
            "expenses": user_session_data[(user_session_data['timestamp_interval'] == interval) & (user_session_data['event_type'] == "BUY_PRODUCT")]['price'].sum()}
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
                "expenses": calculate_expenses_with_interval(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id], min_value, max_value)
            }
        )
    return users_time_series


def predict_expenses_for_user(time_series):
    model = LinearRegression()
    model.fit(time_series['interval_number'].values.reshape(-1, 1), time_series['expenses'].values.reshape(-1, 1))
    return round(max(model.predict((time_series['interval_number'].max() + 1).reshape(1, -1))[0, 0], 0), 2)


def predict_expenses_for_all_users(sessions_data, products_data):
    extracted_time_series = extract_time_series(sessions_data, products_data)
    user_future_expenses = []
    for record in extracted_time_series:
        user_future_expenses.append({
            "user_id": record["user_id"],
            "user_expenses": predict_expenses_for_user(record["expenses"])
        })
    return pd.DataFrame(data=user_future_expenses).set_index('user_id')


def predict(products: DataFrame, deliveries: DataFrame, sessions: DataFrame, user: DataFrame):
    if sessions.empty:
        return 0

    sessions["timestamp"] = pd.to_datetime(sessions["timestamp"])
    sessions['timestamp_interval'] = sessions['timestamp'].apply(lambda x: x.month)
    predictions = predict_expenses_for_all_users(sessions, products)

    return predictions.to_dict()["user_expenses"][user['user_id'].item()]
