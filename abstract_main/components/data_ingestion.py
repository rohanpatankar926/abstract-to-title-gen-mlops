from abstract_main import utils
from abstract_main.entity import config_entity
from abstract_main.entity import artifact_entity
from abstract_main.exception import AbstractException
from abstract_main.logger import logging
from datasets import load_dataset,load_from_disk,load_metric
import os,sys
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig ):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AbstractException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            #Exporting collection data as pandas dataframe
            df:pd.DataFrame  = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)
            print(df)

            logging.info("Save data in feature store")
            #Save data in feature store
            logging.info("Create feature store folder if not available")
            #Create feature store folder if not available
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            logging.info("split dataset into train and test set")
            #split dataset into train and test set
            full_dataset=load_dataset("csv",data_files=self.data_ingestion_config.feature_store_file_path)
            cols_to_remove=list(full_dataset["train"].features.keys())
            cols_to_remove.remove("title")
            cols_to_remove.remove("text")
            data=full_dataset.remove_columns(cols_to_remove)
            dataset = data["train"].train_test_split(test_size=self.data_ingestion_config.test_size)
            test_val=dataset["test"].train_test_split(test_size=self.data_ingestion_config.test_val)
            dataset["val"]=test_val["train"]
            dataset["test"]=test_val["test"]
            dataset.save_to_disk(self.data_ingestion_config.data_file_path)
            print(dataset)
            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            # dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            # os.makedirs(dataset_dir,exist_ok=True)

            # logging.info("Save df to feature store folder")
            # #Save df to feature store folder
            # train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            # test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)
            
            #Prepare artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise AbstractException(error_message=e, error_detail=sys)