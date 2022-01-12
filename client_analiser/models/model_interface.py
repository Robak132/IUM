from pandas import DataFrame


class ModelInterface:
    def predict_expenses(self,
                         products: DataFrame,
                         deliveries: DataFrame,
                         sessions: DataFrame,
                         users: DataFrame) -> dict[str, float]:
        raise Exception("This is a interface")
