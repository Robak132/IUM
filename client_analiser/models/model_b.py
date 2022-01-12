from pandas import DataFrame
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random

from client_analiser.models import ModelInterface


class ModelB(ModelInterface):
    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        return {}

    def get_user_id_from_session(self, session):
        sample_user_id = session['user_id'].iloc[0]
        for user_id in session['user_id']:
            if sample_user_id != user_id:
                raise Exception("How it is even possible")
        return sample_user_id

    def get_user_information(self, user_session_data):
        d = {
            'user_id': [self.get_user_id_from_session(user_session_data)],
            'expenses': [user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()],
            'products_bought': [len(user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"])],
            'events_number': [len(user_session_data)],
            'sessions_number': [len(user_session_data['session_id'].unique())],
            'average_discount': [user_session_data['offered_discount'].mean()],
            'average_discount_on_bought': [
                user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['offered_discount'].mean()]
        }
        df = pd.DataFrame(data=d)
        return df.set_index('user_id')

    # %%
    def extract_users_data(self, sessions_data, users_data, products_data):
        enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
        extracted_users = []
        for user_id in enriched_sessions_data['user_id'].unique():
            extracted_users.append(
                self.get_user_information(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]))
        enriched_users_data = pd.concat(extracted_users)
        return pd.merge(enriched_users_data, users_data, on="user_id").drop(columns=['name', 'street'])


    # def train(self, x: DataFrame, y:DataFrame):
