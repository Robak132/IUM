import requests
import random
import json


def extract_data(users_num: float):
    json_object = {}
    with open('data/iteration_3/raw/users.jsonl') as f:
        users = [json.loads(line) for line in f]
        user_ids = []
        random.shuffle(users)
        users = users[:int(len(users) * users_num)]
        for user in users:
            user_ids.append(user["user_id"])
        json_object["users"] = users
    with open('data/iteration_3/raw/sessions.jsonl') as f:
        sessions = []
        for line in f:
            session = json.loads(line)
            if session["user_id"] in user_ids:
                sessions.append(session)
        json_object["sessions"] = sessions
    with open('data/iteration_3/raw/products.jsonl') as f:
        products = [json.loads(line) for line in f]
        json_object["products"] = products
    return json_object


if __name__ == '__main__':
    json_request = extract_data(0.1)
    x = requests.post("http://localhost:8080/predict/A", json=json_request)
    print(x.text)
