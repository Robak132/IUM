import uvicorn
from random import choice
from fastapi import FastAPI, Response, Request

from client_analiser.models.predict_model import predict_a, predict_b

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predictA")
async def get_prediction_a(request: Request, response: Response):
    input_data = await request.json()
    return {"Result": predict_a(input_data)}


@app.post("/predictB")
async def get_prediction_b(request: Request, response: Response):
    input_data = await request.json()
    return {"Result": predict_b(input_data)}


@app.get('/form')
async def get_prediction_ab(request: Request, response: Response):
    input_data = await request.json()
    if choice([True, False]):
        model = predict_a
    else:
        model = predict_b
    return {"Result": model(input_data)}


@app.post("/predict")
async def get_predictction_a(request: Request, response: Response):
    input_data = await request.json()

    if choice([True, False]):
        model = predict_a
    else:
        model = predict_b

    print(f"Using model {model}")

    return {"Result": model(input_data)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, log_level="info", reload=True)
