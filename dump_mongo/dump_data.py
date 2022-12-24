import sys
import os 
sys.path.append(os.getcwd())
import pymongo 
import json
from dataclasses import dataclass
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
client=pymongo.MongoClient(os.getenv("MONGO_URL"))

@dataclass
class EnvironmentVariable(object):
    DATA_FILE_PATH:str=os.getenv("DATA_FILE_PATH")
    DATABASE_NAME:str=os.getenv("DATABASE_NAME")
    COLLECTION_NAME:str=os.getenv("COLLECTION_NAME")

env=EnvironmentVariable()
print(env.COLLECTION_NAME,env.DATA_FILE_PATH,env.DATABASE_NAME)
if __name__=="__main__":
    db=pd.read_csv(env.DATA_FILE_PATH)
    print(f"Data file path: {env.DATA_FILE_PATH}")
    print(f"Rows->{db.shape[0]} and columns->{db.shape[1]}")
    db.reset_index(drop=True,inplace=True)
    json_record=list(json.loads(db.T.to_json()).values())
    print(json_record[0])
    client[env.DATABASE_NAME][env.COLLECTION_NAME].insert_many(json_record)
    print("successfully uploaded data to mongodb")