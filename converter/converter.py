import csv
import json
import pandas as pd



if __name__ == "__main__":
    sdf_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    sdf_test = sdf_test.drop(columns=['selling_price'])
    json_data = json.dumps(json.loads(sdf_test.to_json(orient='records')), indent=2)
    text_file = open("test_set.txt", "w")
    text_file.write(json_data)
    text_file.close()
    pass