import json
import random
import time

import pandas as pd
import uvicorn
import os.path
from fastapi import FastAPI, Request
from pandas import DataFrame

from microservice.models import ModelInterface, ModelA, ModelB
from microservice.utils import PrettyJSONResponse

app = FastAPI()

# Models
model_A: ModelInterface = ModelA()
model_B: ModelInterface = ModelB()


@app.on_event("startup")
async def startup_event():
    if not os.path.exists("logs/log.csv"):
        await reset_log()


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
    try:
        log_data = pd.read_csv("logs/log.csv", header=0, sep=";")
        good_results = json.load(open("logs/good_results.json"))
        if log_data.empty:
            raise FileNotFoundError("Empty log")
    except FileNotFoundError:
        return "AB test was not initiated"

    log_data['good'] = log_data.apply(lambda row: good_results[str(row["user_id"])], axis=1)
    log_data['error'] = (log_data['result'] - log_data['good']) ** 2
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


@app.get("/predict/good_results", response_class=PrettyJSONResponse)
async def get_good_results():
    try:
        json_object = json.load(open("logs/good_results.json"))
        return json_object
    except FileNotFoundError:
        return "No data was found"


@app.post("/predict/good_results")
async def set_good_results(request: Request):
    input_data = await request.json()
    file = open("logs/good_results.json", "w+")
    json.dump(input_data, file, indent=4)
    return {}


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


def predict_group(products: DataFrame,
                  deliveries: DataFrame,
                  sessions: DataFrame,
                  users: DataFrame,
                  ab_ratio: float = 0.5):
    now = time.strftime('%Y-%m-%dT%H:%M:%S')

    users_a = users.sample(frac=ab_ratio)
    sessions_a = sessions[sessions["user_id"].isin(users_a["user_id"].to_list())]

    users_b = users.drop(users_a.index)
    sessions_b = sessions[sessions["user_id"].isin(users_b["user_id"].to_list())]

    result = predict_using_model(products, deliveries, sessions_a, users_a, now, model_A)
    result.update(predict_using_model(products, deliveries, sessions_b, users_b, now, model_B))
    return result


def predict_using_model(products: DataFrame,
                        deliveries: DataFrame,
                        sessions: DataFrame,
                        users: DataFrame,
                        now: str,
                        model: ModelInterface()) -> dict[str, float]:
    result = model.predict_expenses(products, deliveries, sessions, users)
    for key in result.keys():
        with open("logs/log.csv", "a+", encoding="utf-8") as file:
            file.write(f"{now};{key};{model.__module__}.{model.predict_expenses.__name__};{result[key]}\n")
    return result


def make_dataframes(input_data: dict):
    users = DataFrame.from_dict(input_data.get('users', {}))
    sessions = DataFrame.from_dict(input_data.get('sessions', {}))
    products = DataFrame.from_dict(input_data.get('products', {}))
    deliveries = DataFrame.from_dict(input_data.get('deliveries', {}))
    return products, deliveries, sessions, users


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
