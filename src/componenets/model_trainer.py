import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import costomexception 
from src.logger import logging
from src.utils import save_obj,evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_train_config=ModelTrainerConfig()
    
    def initate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting training and testing the data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "linearRegression":LinearRegression(),
                "KNeighbouring":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "Catboost":CatBoostRegressor(verbose=False),
                "gradientboost":GradientBoostingRegressor(),
                "decision tree":DecisionTreeRegressor()
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
        ###to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
        ###to get the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise costomexception("no best model found")
            logging.info("best found model in training and testing data")

            save_obj(
                file_path=self.model_train_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
    
        except Exception as es:
                raise costomexception(es,sys)
            
