from fastapi import FastAPI, Depends
from fastapi import File, UploadFile

import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from starlette.responses import JSONResponse
from fastapi.responses import StreamingResponse
import pandas as pd

from predictor.predictor import Predictor
from trained_pipeline.pipeline import get_pipeline
from trained_pipeline.trim_transformer import make_data
from sklearn.metrics import r2_score
from io import BytesIO

class Item(BaseModel):
    name: str
    year: int
    selling_price: int = 0
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


clf = get_pipeline('./model/pipeline.pkl')
predictor = Predictor(clf)

app = FastAPI()


@app.get('/')
async def root():
    '''
    Этот метод используется для тестирования работы предиктора
    :return:
    '''

    sX_train, sX_test, sy_train, sy_test = make_data()
    pred = clf.predict(sX_test)
    print(f"test r2 score = {r2_score(sy_test, pred)}")
    return {'test r2 score': r2_score(sy_test, pred)}


@app.post("/predict_item")
def predict_item(item: Item) -> Item:
    '''
    Возвращает переданный в запросе объект с предсказанной ценой
    :return:
    '''
    predicted_value = predictor.predict([item])
    return predicted_value[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[Item]:
    '''
    Возвращает список переданных объектов в запросе с предсказанной ценой
    :return:
    '''
    predicts = predictor.predict(items)
    return predicts


@app.post("/predict_item_csv")
def upload(file: UploadFile = File(...)):
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer, sep=';')
    buffer.close()
    file.file.close()
    res = predictor.predict_csv(df)
    output = res.to_csv(index=True)
    return StreamingResponse(
        iter([output]),
        media_type='text/csv',
        headers={"Content-Disposition":
                     "attachment;filename=prediction.csv"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
