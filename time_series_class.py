import os
from prophet import Prophet
import numpy as np
import itertools
from prophet.diagnostics import cross_validation

import json
from prophet.serialize import model_to_json, model_from_json

from datetime import datetime, timedelta

import sklearn.metrics as metrics


# from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
# from pmdarima.arima import ADFTest
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
# from pmdarima.arima import auto_arima

import uuid 

import import_ipynb


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

from Utilities import generate_prophet_time_frame,initialize_model_db,initialize_forecast_db,save_pandas_dataframe


print("hello")

class timedata:
    '''Class that represents a data instance with a Train / Test splits
       
       Takes in data which have a date index and can have other columns (other measures)
       - Divide into Train and Test ( hold out sets)
       -
    
    
    '''
    
    def __init__(self,data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be Pandas dataframe")
        self.initial_data=data
        self.data=data
        try :
            self.frequency=pd.infer_freq(self.data.index)
        except :
             self.frequency='D'
        
                
        self.first_timestamp=data.index[0]
        self.end_timestamp=data.index[-1]
        self.nexttimestamp=self.end_timestamp + timedelta(days=1)
        
    @property    
    def length(self):
        print(f"Total length of data is {len(self.data)}")
        
    def hold_out_data(self,hold_out_percentage=0.2):
        ''' Method to produce train and test based on hold out percentage'''
        hold_out_size=round(hold_out_percentage * len(self.data))
        
        return self.data[:len(self.data)- hold_out_size],self.data[len(self.data)- hold_out_size:]

#         self.train=self.data[:len(self.data)- hold_out_size]
#         self.test=self.data[len(self.data)- hold_out_size:]
    
class data_extended(timedata):  
    
    def __init__(self,data):
        super().__init__(data)
        self.incoming_df=None
        self.incoming_counter=0
    
    
    def incoming_new(self,measure,value_list):
        
        '''
        When new data comes,the data instance expands to include the incoming data.
        self.incoming_df holds the new data only    
        
        Note : We can deal with time indexed dataframe that contains alot of measures.
        '''
        a={}
        for column_name in self.data.columns.values:
            if column_name !=measure:
                a[column_name]=[None for _ in range(len(value_list))]
            else:
                a[column_name]=value_list
        
        self.incoming_df=pd.DataFrame( data=
                        a,
                         index=pd.date_range(self.nexttimestamp,periods=len(value_list),freq='D')
                        )
        self.incoming_df.index.names = ['date']

        
        self.data=self.data.append(self.incoming_df) 
        
#         self.train=None
#         self.test=None
        
        
        return self.data,self.incoming_df

    def reset_incoming(self):
        self.incoming_all=None
        self.data=self.initial_data



class model_build:   
    
    '''
    Important attributes include 
    self.train , self.test , self.forecast and self.predicted_test  start/end timestamps for each.
    When a model instance is fit again , All of the above will change 
    
    Note: Every data variable is a DF which may have multiple measures
    ''' 
    
    def __init__(self,model_directory=None):
        self.model_name=model_name
        if model_directory is not None :
            m=load_latest_model()
        self.model=m
        set_model_attributes
        
        
    
    def _set_start_end_timestamps(self,data):
        
        if data is not None :
            start,end=data.index[0],data.index[-1]
        else:
            start,end=None,None
            
        return start,end
    
    def fit(self,train_data,measure,test_data=None,**kwargs):
        '''
        Can fit using train only or train and test . 
        In case of test the model will predict for test as a preparation for evaluate function
        '''
        self.train_data=train_data[measure]
        self.prophet_train_data=train_data.reset_index()[['date',measure]].rename(columns={'date':'ds',measure:'y'}) 
        m1=Prophet(**kwargs)
        m1.fit(self.prophet_train_data)
        
        self.model=m1
        self.model_id=uuid.uuid4()
        
        self.modeltrain_startime,self.modeltrain_endtime=self._set_start_end_timestamps(self.train_data)
        
        ##
        train_ds=self.model.make_future_dataframe(periods=0,include_history=True)
        self.predicted_train=self.model.predict(train_ds)[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')
        
    


        if test_data is not None :            
            self.test_data=test_data[measure]
            test_ds=self.model.make_future_dataframe(len(self.test_data),include_history=False)
            self.predicted_test=self.model.predict(test_ds)[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds') 
            
            
        else:
            self.test_data=None
            self.predicted_test=None

        self.modeltest_starttime,self.modeltest_endtime=self._set_start_end_timestamps(self.test_data)

        
        self.forecasts=None  #The model has not forecasted out of sample yet


        print(f"Train data is between {self.modeltrain_startime.strftime('%d-%m-%Y')} and {self.modeltrain_endtime.strftime('%d-%m-%Y')}")
        
        try:
            print(f"Test data is between {self.modeltest_starttime.strftime('%d-%m-%Y')} and {self.modeltest_endtime.strftime('%d-%m-%Y')}")
        except:
            pass   
        
    def evaluate(self):
        '''Evaluate the model in case of test set
           Returns model output 
        '''
        
        self.train_MAE=metrics.mean_absolute_error(self.train_data, self.predicted_train['yhat'])        
        print('Train Mean Absolute Error:',self.train_MAE)  

          
        if self.test_data is not None :           
            
            self.test_MAE=metrics.mean_absolute_error(self.test_data, self.predicted_test['yhat'])        
            print('Test Mean Absolute Error:',self.test_MAE)  
            

        
        
        model_output_data={'model_id':[self.model_id] ,
                           'model_train_starttime':[self.modeltrain_startime],
                           'model_train_endtime':[self.modeltrain_endtime],
                          'model_test_starttime':[self.modeltest_starttime],
                           'model_test_endtime':[self.modeltest_endtime],
                          'training_type':['fit'],
                          'model_training_day':[datetime.now().strftime('%d-%m-%Y')],
                           'model_train_MAE':self.train_MAE,
                           'model_test_MAE':self.test_MAE
                          }
        return pd.DataFrame(model_output_data)
    
    def forecast(self,start_date=None,forecast_steps=1):
        '''Use the model to forecast a number of steps 
        If start date is not given, predict from the end of the test ,if no test set given before in fit .
        Then predict from the end of train
        
        Returns Forecast output 
        '''
        if start_date is None :
            if self.test_data is None :
                start_date =self.modeltrain_endtime + timedelta(days = 1)
            else :
                start_date =self.modeltest_endtime + timedelta(days = 1)
                
        forecast_timeframe=generate_prophet_time_frame(start_date=start_date,forecast_steps=forecast_steps)
        self.forecasts=self.model.predict(forecast_timeframe)[['ds','yhat','yhat_lower','yhat_upper']].set_index('ds')


        self.modelforecast_startime,self.modelforecast_endtime=self._set_start_end_timestamps(self.forecasts)

    
        print(f"Forecasts between {self.modelforecast_startime.strftime('%d-%m-%Y')} and {self.modelforecast_endtime.strftime('%d-%m-%Y')} \n")
        return self.forecasts.reset_index().join(pd.DataFrame({'model_id':forecast_steps * [self.model_id]}))
            
    
    def save_model(self,model_directory): #model_path=C:\modeldb\sample        
        with open(os.path.join(model_directory,f'{self.model_id}.json'), 'w') as fout:
            json.dump(model_to_json(self.model), fout)  # Save model
    
    def load_latest_model(self,model_directory):
        model_files=os.listdir(model_directory)
        paths = [os.path.join(model_directory, basename) for basename in model_files]
        latest_model_path=max(paths, key=os.path.getctime)
        print(latest_model_path)
        with open(latest_model_path, 'r') as fin:
            m = model_from_json(json.load(fin))  # Load model
        return m

    '''def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)'''
    
    def plot_fit(self):
        plt.figure(figsize=(12,5))

        
        if self.test_data is None:
#             m1.plot(results.reset_index())
#             plt.show()        
            self.model.plot(self.predicted_train.reset_index())
            
        else :
            
            all_ds=self.model.make_future_dataframe(len(self.test_data),include_history=True)
            results=self.model.predict(all_ds)[['ds','yhat','yhat_lower','yhat_upper']]
            self.model.plot(results)
            
        plt.legend()
        plt.show()


            
        
            
    def plot(self,incoming_df=None,measure=None):
        ''' The model will have the latest train and test sets . 
            Also the  forecast 
        
        '''
        plt.figure(figsize=(12,5))
        plt.plot(self.train_data,label="Train")
        if self.test_data is not None :
            plt.plot(self.test_data,label="Test")
            plt.plot(self.predicted_test['yhat'],label='Test Predictions')
            plt.fill_between(self.test_data.index,self.predicted_test['yhat_lower'],self.predicted_test['yhat_upper'],color='k',alpha=0.15)

        if self.forecasts is not None :
            plt.plot(self.forecasts['yhat'],label='Forecasts')
            plt.fill_between(self.forecasts.index,self.forecasts['yhat_lower'],self.forecasts['yhat_upper'],color='k',alpha=0.15)
        
        if incoming_df is not None :
            plt.plot(incoming_df[measure],label="Incoming")

        
            
        plt.legend()
        plt.show()
    