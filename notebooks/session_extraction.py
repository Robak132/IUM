import pandas as pd


def get_user_id_from_session(session):
    sample_user_id = session['user_id'].iloc[0]
    for user_id in session['user_id']:
        if sample_user_id != user_id:
            raise Exception("How it is even possible")
    return sample_user_id


def get_session_id_from_session(session):
    sample_session_id = session['user_id'].iloc[0]
    for session_id in session['user_id']:
        if sample_session_id != session_id:
            raise Exception("How it is even possible")
    return sample_session_id


def check_if_user_bought_something(session):
    for event_type in session['event_type']:
        if event_type == 'BUY_PRODUCT':
            return True
    return False


def get_session_information(session):
    d = {
        'session_id': get_session_id_from_session(session),
        'beginning': [min(session['timestamp'])],
        'end': [max(session['timestamp'])],
        'user_id': get_user_id_from_session(session),
        'bought_product': check_if_user_bought_something(session)
    }
    df = pd.DataFrame(data=d)
    return df.set_index('session_id')


def extract_session_batch(sessions_data):
    sessions = []
    for session_id in sessions_data['session_id'].unique():
        sessions.append(get_session_information(sessions_data[sessions_data['session_id'] == session_id]))
    enriched_session_batch = pd.concat(sessions)
    return enriched_session_batch


def get_user_information(user_session_data):
    d = {
        'user_id': [get_user_id_from_session(user_session_data)],
        'expenses': [user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"]['price'].sum()],
        'products_bought': [len(user_session_data[user_session_data['event_type'] == "BUY_PRODUCT"])],
        'events_number': [len(user_session_data)]
    }
    df = pd.DataFrame(data=d)
    return df.set_index('user_id')


def extract_users_data(sessions_data, users_data, products_data):
    enriched_sessions_data = pd.merge(sessions_data, products_data, on="product_id").sort_values(by=['timestamp'])
    extracted_users = []
    for user_id in enriched_sessions_data['user_id'].unique():
        extracted_users.append(
            get_user_information(enriched_sessions_data[enriched_sessions_data['user_id'] == user_id]))
    enriched_users_data = pd.concat(extracted_users)
    return pd.merge(enriched_users_data, users_data, on="user_id").drop(columns=['name', 'street'])


def find_returned_users(extracted_sessions_data):
    user_counts = extracted_sessions_data['user_id'].value_counts()
    return user_counts[user_counts >= 2].index


def find_never_returned_users(extracted_sessions_data):
    user_counts = extracted_sessions_data['user_id'].value_counts()
    return user_counts[user_counts < 2].index
