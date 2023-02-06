import json
import dill
import pandas as pd
import os
import sys
from _datetime import datetime

path = os.path.expanduser('~/airflow_hw')
os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)


def predict():
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    predicted_df = pd.DataFrame(columns=['id', 'predict'])

    def prediction(data_path):
        with open(data_path, 'r') as f:
            data = pd.DataFrame.from_dict([dict(json.load(f))])
            y = model.predict(data)[0]
            predicted_df.loc[len(predicted_df.index)] = [int(data.id), y]

    for file_name in os.listdir('data/test'):
        prediction('data/test/' + file_name)

    predicted_df.to_csv(f'data/predictions/predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
