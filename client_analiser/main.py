import random
import uvicorn
from fastapi import FastAPI, Response, Request
from starlette.templating import Jinja2Templates

from client_analiser.models.predict_model import predict_a, predict_b

app = FastAPI()
template = Jinja2Templates(directory="templates/")


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


@app.get('/form')
def form_post(request: Request, response: Response):
    return template.TemplateResponse('index.html', {"request": request, "response": response})


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
