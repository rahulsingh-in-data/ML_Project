import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_filepath = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {'CatBoostRegressor': CatBoostRegressor(),
                  'AdaBoostRegressor': AdaBoostRegressor(),
                  'GradientBoostingRegressor': GradientBoostingRegressor(), 
                  'RandomForestRegressor': RandomForestRegressor(),
                  'LinearRegression': LinearRegression(),
                  'KNeighborsRegressor': KNeighborsRegressor(),
                  'DecisionTreeRegressor': DecisionTreeRegressor(),
                  'XGBRegressor': XGBRegressor()
                  }

    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting Training & Test Data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, self.models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                (list(model_report.values()).index(best_model_score))]
            
            best_model = self.models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            logging.info(f"Best founded model is {best_model_name}")

            save_object(self.model_trainer_config.train_model_filepath, obj=best_model)

            predicted = best_model.predict(X_test)
            model_r2_score = r2_score(y_test, predicted)

            return model_r2_score

        except Exception as e:
            raise CustomException(e, sys)