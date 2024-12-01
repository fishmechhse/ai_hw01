from trained_pipeline.pipeline import joblib_load_estimator
from sklearn.metrics import r2_score

from trained_pipeline.trim_transformer import make_data


def main():
    clf = joblib_load_estimator('./../model/pipeline.pkl')
    sX_train, sX_test, sy_train, sy_test = make_data()
    pred = clf.predict(sX_test)
    print(f"test r2 score = {r2_score(sy_test, pred)}")


if __name__ == '__main__':
    main()
