from pydantic import BaseModel
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List


class Predictor:
    __pipeline = Pipeline

    def __init__(self, predictor: Pipeline):
        self.__predictor = predictor

    def predict(self, items: List[BaseModel]) -> List[BaseModel]:
        data = [item.model_dump() for item in items]
        df = pd.DataFrame(data=data, index=np.arange(len(data)))
        predicted = self.__predictor.predict(df)
        ret = []
        for i in range(len(items)):
            items[i].selling_price = predicted[i]
            obj = items[i]
            ret.append(obj)
        return ret
