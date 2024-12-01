import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import math
import cloudpickle

def try_str_to_int(val: str):
    try:
        i = math.floor(float(val))
        return str(i)
    except Exception as ex:
        return '0'


def split_val(row, column_name):
    if pd.isna(row[column_name]) == False:
        val = row[column_name].strip().split(" ")
        if len(val) > 0:
            row[column_name] = try_str_to_int(val[0])
        else:
            row[column_name] = '0'
    return row


class TrimTransformer(BaseEstimator, TransformerMixin):
    __feature_names_out = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.apply(lambda row: split_val(row, 'mileage'), axis=1)
        X = X.apply(lambda row: split_val(row, 'engine'), axis=1)
        X = X.apply(lambda row: split_val(row, 'max_power'), axis=1)

        X["mileage"] = X["mileage"].astype(float)
        X["engine"] = X["engine"].astype(float)
        X["max_power"] = X["max_power"].astype(float)
        self.__feature_names_out = X.columns
        return X

    def get_feature_names_out(self, feature_names_out):
        return self.__feature_names_out


def load_estimator(filename):
    f = open(filename, "rb")
    model = cloudpickle.load(f)
    return model


def make_data():
    sdf_train = pd.read_csv(
        'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    sdf_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')

    source = list(sdf_train.columns)
    excluded_selling_price = list(set(source).difference(set(['selling_price'])))
    sdf_train = sdf_train.drop_duplicates(subset=excluded_selling_price, keep='first').reset_index(drop=True)

    sdf_train["seats"] = sdf_train["seats"].astype(str)
    sdf_test["seats"] = sdf_test["seats"].astype(str)
    y_train = sdf_train["selling_price"]
    X_train = sdf_train.drop(columns=["selling_price", "torque"])

    y_test = sdf_test["selling_price"]
    X_test = sdf_test.drop(columns=["selling_price", "torque"])

    return X_train, X_test, y_train, y_test
