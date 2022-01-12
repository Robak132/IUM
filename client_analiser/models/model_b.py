from pandas import DataFrame

from client_analiser.models import ModelInterface


class ModelB(ModelInterface):
    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        return {}
