import os,sys
from abstract_main.exception import AbstractException
from abstract_main.logger import logging
from datetime import datetime
from dotenv import load_dotenv

FILE_NAME = "abstract.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception  as e:
            raise AbstractException(e,sys)     


class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig)->None:
        try:
            self.database_name="abstracttotitle"
            self.collection_name="abstract_collection"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.data_file_path = os.path.join(self.data_ingestion_dir,"dataset")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            # self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            # self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.25
            self.test_val=0.5
        except Exception  as e:
            raise AbstractException(e,sys)     

    def to_dict(self,)->dict:
        try:
            return self.__dict__
        except Exception  as e:
            raise AbstractException(e,sys)     