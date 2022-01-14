Client Analyser
==============================

Regressive model to predict clients' probability to return to the online shop.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data               <- Data files and methods to transform it 
    │   ├── ab_test        <- Test data for AB test
    │   │
    │   ├── iteration_1
    │   ├── iteration_2
    │   └── iteration_3    <- Subsequent iterations of data delivered by client
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── microsevice        <- Source code for microservice used in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Methods to turn raw data into features for modeling
    │   │
    │   └── models         <- Methods to train models and then use trained models to predictions
    │
    ├── tests              <- Tests and scripts to test AB test
    |
    └── requirements.txt   <- File with all python dependecies


--------


Setting up the environment
------------

1. Create an environment with `python -m venv venv`
2. Activate environment with `./venv/Scripts/activate`
3. Install requirements with `pip install -r requirements.txt`
4. Run microservice with `python ./microservice/main.py`

<p><small>Project partially based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
