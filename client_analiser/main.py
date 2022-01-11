import json
import random
import time

import pandas as pd
import uvicorn
import os.path
from fastapi import FastAPI, Request
from pandas import DataFrame

from client_analiser.models.model_a import predict as predict_a
from client_analiser.models.model_b import predict as predict_b
from client_analiser.utils import PrettyJSONResponse

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    if not os.path.exists("logs/log.csv"):
        await reset_log()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/A")
async def get_prediction_a(request: Request):
    input_data = await request.json()
    result = predict_group(*make_dataframes(input_data), ab_ratio=1)
    return result


@app.post("/predict/B")
async def get_prediction_b(request: Request):
    input_data = await request.json()
    result = predict_group(*make_dataframes(input_data), ab_ratio=0)
    return result


@app.get("/predict", response_class=PrettyJSONResponse)
async def make_report():
    log_data = pd.read_csv("logs/log.csv", header=0, sep=";")
    log_data['good'] = log_data.apply(lambda row: ab_get_valid_result(row["user_id"]), axis=1)
    log_data['error'] = (log_data['result'] - log_data['good'])**2
    log_json = json.loads(log_data.to_json(orient='records'))

    log_data = log_data.drop(columns=["good", "result", "user_id"])
    grouped_data = log_data.groupby(by=["model", "timestamp"]).mean()
    total_grouped_data = log_data.drop(columns="timestamp").groupby(by=["model"]).mean()
    model_results = {}

    for row, data in grouped_data.iterrows():
        model, timestamp = row
        error = data["error"]
        json_object = {"model": model, "MSE": error}
        timestamp_results = model_results.get(timestamp, [])
        timestamp_results.append(json_object)
        model_results[timestamp] = timestamp_results

    for model, data in total_grouped_data.iterrows():
        error = data["error"]
        json_object = {"model": model, "MSE": error}
        timestamp_results = model_results.get("_total", [])
        timestamp_results.append(json_object)
        model_results["_total"] = timestamp_results

    return {"model_results": model_results, "log": log_json}


def ab_get_valid_result(unit_id):
    with open("../data/ab_test/ab_test_good_data.json") as file:
        json_object = json.load(file)
        return json_object[str(unit_id)]


@app.post("/predict")
async def get_prediction_ab(request: Request):
    input_data = await request.json()
    result = predict_group(*make_dataframes(input_data))
    return result


@app.post("/predict/reset_log")
async def reset_log():
    os.remove("logs/log.csv")
    with open("logs/log.csv", "a+", encoding="utf-8") as file:
        file.write(f"timestamp;user_id;model;result\n")


def predict_group(products: DataFrame, deliveries: DataFrame, sessions: DataFrame, users: DataFrame,
                  ab_ratio: float = 0.5):
    now = time.strftime('%Y-%m-%dT%H:%M:%S')
    result_dict = {}
    for row, user in users.iterrows():
        if random.random() <= ab_ratio:
            result = predict_one_user(user['user_id'], (deliveries, products, sessions, users), now, predict_a)
            result_dict[user["user_id"]] = result
        else:
            result = predict_one_user(user['user_id'], (deliveries, products, sessions, users), now, predict_b)
            result_dict[user["user_id"]] = result
    return result_dict


def predict_one_user(user_id: int, data: tuple[DataFrame, DataFrame, DataFrame, DataFrame], now: str, model):
    deliveries, products, sessions, users = data
    _user = users[users['user_id'] == user_id]
    _sessions = sessions[sessions['user_id'] == user_id]
    result = model(products, deliveries, _sessions, _user)
    with open("logs/log.csv", "a+", encoding="utf-8") as file:
        file.write(f"{now};{user_id};{model.__module__}.{model.__name__};{result}\n")
    return result


def make_dataframes(input_data: dict):
    users = DataFrame.from_dict(input_data.get('users', {}))
    sessions = DataFrame.from_dict(input_data.get('sessions', {}))
    products = DataFrame.from_dict(input_data.get('products', {}))
    deliveries = DataFrame.from_dict(input_data.get('deliveries', {}))
    return products, deliveries, sessions, users


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
