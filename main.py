from abstract_main.logger import logging
from abstract_main.exception import AbstractException
from abstract_main.utils import get_collection_as_dataframe
import sys,os
from abstract_main.entity import config_entity
from abstract_main.components.data_ingestion import DataIngestion
# from abstract_main.components.data_validation import DataValidation

if __name__=="__main__":
    try:
          #Data Ingestion
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise AbstractException(e, sys)