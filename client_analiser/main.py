import random
from time import strftime

import numpy as np
import uvicorn
from random import choice
from fastapi import FastAPI, Response, Request
from pandas import DataFrame
import client_analiser.models.model_a as model_a
import client_analiser.models.model_b as model_b

app = FastAPI()
log_filename = ""


@app.on_event("startup")
async def startup_event():
    global log_filename
    log_filename = strftime("log_%Y%m%d%H%M%S.tsv")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict/A")
async def get_prediction_a(request: Request, response: Response):
    input_data = await request.json()
    set_a, set_b = split_input_data(input_data, ratio=0.0)
    result_a = model_a.predict(*set_a)
    print(result_a)
    result_b = model_b.predict(*set_b)
    print(result_b)

    result = result_a.copy()
    result.update(result_b)
    return result


@app.post("/predict/B")
async def get_prediction_b(request: Request, response: Response):
    input_data = await request.json()
    result = get_result(input_data, model_b.predict)
    return result


@app.post("/predict")
async def get_prediction_ab(request: Request, response: Response):
    global log_filename
    input_data = await request.json()
    model = model_a.predict if choice([True, False]) else model_b.predict
    result = get_result(input_data, model)
    return result


def get_result(input_data, model):
    result = model(*split_input_data(input_data))

    with open(f"logs/{log_filename}", "a+", encoding="utf-8") as file:
        file.write(f"{input_data}\t{model.__name__}\t{result}\n")
    return result


def split_input_data(input_data: dict, ratio: float = 0.5):
    users = DataFrame.from_dict(input_data.get('users', {}))
    sessions = DataFrame.from_dict(input_data.get('sessions', {}))
    products = DataFrame.from_dict(input_data.get('products', {}))
    deliveries = DataFrame.from_dict(input_data.get('deliveries', {}))

    users_a = users.sample(frac=ratio)
    users_b = users.drop(users_a.index)
    sessions = sessions[sessions["user_id"] in users_a["user_id"].to_list()]

    set_a = [products, deliveries, sessions, users_a]
    set_b = [products, deliveries, sessions, users_b]

    return set_a, set_b


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
