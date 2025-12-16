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

            params={
                 "DecisionTreeRegressor":{
                      'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                      'splitter':['best','random'],
                      'max_features':['sqrt','log2']
                 },
                 "Random Forest":{
                      'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                      'max_features':['sqrt','log2',None],
                      'n_estimators':[8,16,32,64,128,256]
                 },
                 "GradientBoostingRegressor":{
                      'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                      'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                      'criterion':['friedman_mse','squared_error'],
                      'max_features':['sqrt','log2','auto'],
                      'n_estimators':[8,16,32,64,128,256]
                 },
                 "LinearRegression":{},
                 "KNeighborsRegressor":{
                      'learning_rate':[0.1,0.01,0.05,0.001],
                      'n_estimators':[8,16,32,64,128,256]
                 },
                 "CatBoostRegressor":{
                      'depth':[6,8,10],
                      'learning_rate':[0.01,0.05,0.1],
                      'iterations':[30,50,100]
                    
                 },
                 "AdaBoostRegressor":{
                      'learning_rate':[0.1,0.01,0.05,0.001],
                      'n_estimators':[8,16,32,64,128,256],
                      'loss':['linear', 'square', 'exponential']
                 },
            }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,param=params)
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
            
