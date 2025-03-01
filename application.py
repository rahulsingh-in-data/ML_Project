from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sys

from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predicted_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )
            print(data.race_ethnicity)
            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])
    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)

