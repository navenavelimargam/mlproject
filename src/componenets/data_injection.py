import os
import sys
from src.exception import costomexception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass 
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingectionConfig=DataIngestionConfig()

    def initiate_data_ingection(self):
        logging.info("entered the data ingection method or components")
        try:
            df=pd.read_csv(r'notebook\notebook\data\stud.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingectionConfig.train_data_path),exist_ok=True)
            df.to_csv(self.ingectionConfig.raw_data_path,index=False,header=True)
            logging.info("train test split initated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingectionConfig.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingectionConfig.test_data_path,index=False,header=True)
            logging.info("ingection of the data is completed")
            return(
                self.ingectionConfig.train_data_path,
                self.ingectionConfig.test_data_path
                
            )
        except Exception as es:
            raise costomexception(es,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingection()
            