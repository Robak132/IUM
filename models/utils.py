import pandas as pd


def loss(predictions, observations):
    unified_data = pd.merge(predictions, observations, on="user_id").sort_values(by=['user_id'])
    unified_data['difference'] = unified_data['user_expenses'] - unified_data['expenses']
    unified_data['difference_square'] = unified_data['difference'].apply(lambda x: x ** 2)
    return unified_data
