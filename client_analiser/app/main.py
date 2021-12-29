import random
from random import randint

import uvicorn
from fastapi import FastAPI, Response, Request
from starlette import status

from client_analiser.models.predict_model import predict_a, predict_b

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predictA")
async def root(request: Request, response: Response):
    input_data = await request.json()
    return {"Result": predict_a(input_data)}


@app.post("/predictB")
async def root(request: Request, response: Response):
    input_data = await request.json()
    return {"Result": predict_b(input_data)}


@app.post("/predict")
async def root(request: Request, response: Response):
    input_data = await request.json()

    if random.choice([True, False]):
        model = predict_a
    else:
        model = predict_b

    return {"Result": model(input_data)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
