# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 08:37:10 2022

@author: Emelie Chandni
"""
# Import libraries
import pandas as pd
from datetime import datetime

# Import own modules
from model import Model
from azure_connect import connect_to_azure

def run_AI_ogt():
    
    LINE_list, dataset = line_iteration()
    file_name = 'result_line.csv'      
    time_delta = []
    
    time_delta.append(datetime.now())
    for i in range(7, 8):
        print('------ Prepare line ' + str(LINE_list[i]) + ' -------')
        line = LINE_list[i]
        filter_line = dataset['Linje'] == line
        dataset_line = dataset.loc[filter_line]       
        dataset_journeyGid = dataset_line.groupby('JourneyGID').sum()
        GID_list = dataset_journeyGid.index.values.tolist() 
        for i in range(0, len(GID_list)):           
            print('------------ PREPARING DATA IN PROGRESS ------------')
            print('Iteration: ' + str(i + 1))
            print('Current journeyGID: ' + str(GID_list[i]) + ' of ' + str(len(GID_list)) + ' GID:s')
            model = Model(dataset_line, GID_list[i])
            print('------------ PREPARATION PHASE DONE ------------')
            print('------------ TRAINING IN PROGRESS ------------')
            predict_split_forcast(model, file_name)
            print('------------ END OF CURRENT INTERATION ------------')
            print()
        print('------------ COMPLETION OF PROCESS ------------')
        print('Pushing data do azure account -->')
        connect_to_azure(file_name)
        time_delta.append(datetime.now())
        print('The calculation started at: ', time_delta[0])
        print()
        print('The calculation ended at: ', time_delta[1])
    

def line_iteration():    
    dataset = pd.read_csv('../../../indata/ATR_214_215_216_217_218_220_221_222.csv')
    #dataset = pd.read_csv('../../../indata/ATR_206_202300609.csv')
    dataset_lines = dataset.groupby('Linje').sum()
    LINE_list = dataset_lines.index.values.tolist()  
    print('These lines are presented in the dataset:')
    print()
    print(LINE_list)
    return LINE_list, dataset

'''
    for i in range(0, len(LINE_list)):
        print('Prepare line ' + str(LINE_list[i]))
        line = LINE_list[i]
        filter_line = dataset['Linje'] == line
        dataset_line = dataset.loc[filter_line]
        print('Done ' + str(LINE_list[i]))
    return file_name, dataset_line
'''


# ---- Split length of forcast ----
# IMPORTANT: verification data IS NOT avaliable
# User defines the size of the forcast set by defining len_pred 
def predict_split_forcast(model, file_name):
    
    # Project specific structures
    avgtid_list = model.dataset_preparation()
    
    # General structures
    print('------------------------- ITERATION AVG. TID -------------------------')
    for i in range(0, len(avgtid_list)):
        print('Current avgtid: ', avgtid_list[i])
        model.cleaning_data(i)
        model.scaling_data_forcast()
        len_pred = model.series_to_supervised()    
        if len_pred > 50:
            model.prepare_test_data_forcast()
            regressor = model.create_neural_network()
            model.train_predict_forcast(regressor, i, file_name)