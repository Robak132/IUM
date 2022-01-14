import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from microservice.models import ModelInterface
from models.utils import extract_time_series


class ModelA(ModelInterface):

    @staticmethod
    def predict_expenses_for_user(time_series):
        model = LinearRegression()
        model.fit(time_series['interval_number'].values.reshape(-1, 1), time_series['expenses'].values.reshape(-1, 1))
        return round(max(model.predict((time_series['interval_number'].max() + 1).reshape(1, -1))[0, 0], 0), 2)

    def predict_expenses_for_all_users(self, sessions_data, products_data):
        extracted_time_series = extract_time_series(sessions_data, products_data)
        user_future_expenses = []
        for record in extracted_time_series:
            user_future_expenses.append({
                "user_id": record["user_id"],
                "user_expenses": self.predict_expenses_for_user(record["expenses"])
            })
        return pd.DataFrame(data=user_future_expenses).set_index('user_id')

    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        if sessions.empty:
            return {}

        sessions["timestamp"] = pd.to_datetime(sessions["timestamp"])
        sessions['timestamp_interval'] = sessions['timestamp'].apply(lambda x: x.month)
        predictions = self.predict_expenses_for_all_users(sessions, products)
        return predictions.to_dict()["user_expenses"]

    def load_model(self, string: str):
        raise Exception("This model is not trainable")

    def save_model(self, string: str):
        raise Exception("This model is not trainable")
