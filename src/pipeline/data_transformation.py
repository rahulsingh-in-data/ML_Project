import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import CustomException
from src.pipeline.data_ingestion import DataIngestion
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This Function is responsible for getting data transformation
        '''
        try:
            obj = DataIngestion()
            X_path, y_path = obj.initiate_data_ingestion()
            df = pd.read_csv(X_path)
            numerical_features = [col for col in df.columns if df[col].dtype != 'O']
            numerical_features.pop(0)

            
            categoricaL_features = [col for col in df.columns if df[col].dtype == 'O']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= 'median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder())]
            )

            logging.info("Numerical Columns Standard Scaling Done")

            logging.info("Categorical Columns Encoding Done")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categoricaL_features)
            ])
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transforrmation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Test and Train Data Completed")

            logging.info("Obtaining Pre-Processor Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name  = "math_score"
            numerical_column = ["wriying_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Pre-Processing on Train and Test Data Set"
            )

            infut_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            infut_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[infut_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[infut_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving Pre-Processed Object's")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_data, test_data =  data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transforrmation(train_data, test_data)







