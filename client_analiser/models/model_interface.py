from pandas import DataFrame


class ModelInterface:
    def predict_expenses(self, products: DataFrame, deliveries: DataFrame, sessions: DataFrame, user: DataFrame) -> int:
        raise Exception("This is a interface")
