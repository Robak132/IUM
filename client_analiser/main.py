from time import strftime

import uvicorn
from random import choice
from fastapi import FastAPI, Response, Request

from client_analiser.models.predict_model import predict_a, predict_b

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
    result = get_result(input_data, predict_a)

    return result


@app.post("/predict/B")
async def get_prediction_b(request: Request, response: Response):
    input_data = await request.json()
    result = get_result(input_data, predict_b)

    return result


@app.post("/predict")
async def get_prediction_ab(request: Request, response: Response):
    global log_filename
    input_data = await request.json()
    model = predict_a if choice([True, False]) else predict_b
    result = get_result(input_data, model)

    with open(f"logs/{log_filename}", "a+", encoding="utf-8") as file:
        file.write(f"{input_data}\t{model.__name__}\t{result}\n")

    return result


def get_result(input_data, model):
    result = model(*split_input_data(input_data))
    print(f"Using model {model.__name__}")
    return result


def split_input_data(input_data):
    products = input_data['products']
    deliveries = input_data['deliveries']
    sessions = input_data['sessions']
    users = input_data['users']
    return products, deliveries, sessions, users


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
