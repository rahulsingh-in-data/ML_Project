import sys
from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.data_transformation import DataTransformation
from src.pipeline.model_trainer import ModelTrainer
from src.exception import CustomException

def main():
    try:
        dataingestion = DataIngestion()
        train_path, test_path = dataingestion.initiate_data_ingestion()

        datatransformation = DataTransformation()
        train_data, test_data, _ = datatransformation.initiate_data_transforrmation(train_path, test_path)

        modeltrainer = ModelTrainer()
        score = modeltrainer.initiate_model_trainer(train_data, test_data)
        print(score)
    
    except Exception as e:
        raise CustomException(e, sys)


if __name__=="__main__":
    main()