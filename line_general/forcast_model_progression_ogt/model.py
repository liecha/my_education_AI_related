# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 07:03:27 2022

@author: Emelie Chandni
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 04:55:10 2021

@author: Emelie Chandni
"""
# Import libraries
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

class Model:
    
    def __init__(self, dataset, journeyGID):
        self.dataset = dataset
        self.len_pred = 0
        self.prediction_window = 1
        self.epochs = 50
        self.batch_size = 32
        self.scaled_dataset = []
        self.train_start = []
        self.train_stop = []
        self.test_start = []
        self.test_stop = []
        self.df_split_train_test = []
        self.scaler = []
        self.X_train = np.array
        self.y_train = np.array
        self.X_test = np.array
        self.y_test = np.array
        self.y_pred = np.array
        self.y_pred_inv = np.array
        
        # Project specific variabels
        self.journeyGID = journeyGID
        self.headings = ['SequenceInJourney', 'JourneyGID', 'Linje', 'Turnr', 'Avg.tid', 'VehicleGID', 'Hållplatsnamn', 'Datum', 'Tidpunkt', 'Ombord' ]
        self.param = 'Ombord'       
        self.journeyAVGtid_list = []
        self.prepared_dataset = []
        self.max_sequence = 0
        self.min_sequence = 0
        self.filtered_data = []
        self.stop_names = []
        self.current_line = 0
        self.seq_list =[]
        self.stop_list =[]
        self.set_of_days = []
        self.difference_tour_days = 1
         
    def dataset_preparation(self):  
        # Filtrera bort de parametrar som inte har relevans 
        # Centrala parametrar: tur, linje, sträcka (hållplatser)
        dataset_ATR_selection = self.dataset[self.headings]
        dataset_ATR_selection = dataset_ATR_selection.dropna()
        
        self.current_line = dataset_ATR_selection.iloc[0]['Linje']
        
        # Filtrera på en JourneyGID
        filter_journeyGID = dataset_ATR_selection['JourneyGID'] == self.journeyGID
        data_filter_journeyGID = dataset_ATR_selection.loc[filter_journeyGID]
        
        dataset_journeyAVGtid = data_filter_journeyGID.groupby('Avg.tid').sum()
        self.journeyAVGtid_list = dataset_journeyAVGtid.index.values.tolist()
        
        print('---------- AVGTID CURRENT JOURNEYGID -----------')
        print(self.journeyAVGtid_list)
        
        # Filtrera på en avgångstid        
        for i in range(0, len(self.journeyAVGtid_list)):
            filter_AVGtid = data_filter_journeyGID['Avg.tid'] == self.journeyAVGtid_list[i]
            data_filter_AVGtid = data_filter_journeyGID.loc[filter_AVGtid].sort_values(by=['Tidpunkt'])
            self.filtered_data.append(data_filter_AVGtid)
            
        return self.journeyAVGtid_list
    
    
    def day_detector(self, data_filter_journeyAVGtid):        
        self.seq_list = data_filter_journeyAVGtid['SequenceInJourney'].tolist()
        self.stop_list = data_filter_journeyAVGtid['Hållplatsnamn'].tolist()
        
        list_dates = data_filter_journeyAVGtid.groupby('Datum').mean().index.values.tolist()
        day_array = []
        for i in range(0, len(list_dates)):
            datetime_object = datetime.strptime(list_dates[i], '%Y-%m-%d')
            day_array.append(datetime_object.weekday())
        last_day = max(day_array)
        first_position = 0
        for i in range(0, len(day_array)):
            if day_array[i] == 0:
                first_position = i
                break
        self.set_of_days = []
        for i in range(first_position, len(day_array)):
            self.set_of_days.append(day_array[i])
            if day_array[i] == last_day:
                break
        
        if len(self.set_of_days) > self.set_of_days[-1]:
            self.set_of_days = self.set_of_days[-self.set_of_days[-1]-1:]
        
        diff_array = []
        for i in range(0, len(self.set_of_days)-1):
            diff = self.set_of_days[i+1] - self.set_of_days[i]
            diff_array.append(diff)
        
            if len(diff_array) != 0:
                self.difference_tour_days = round(sum(diff_array)/len(diff_array))
        
        print('Days that this tour drives: ', self.set_of_days)
        print('Diff between days: ', self.difference_tour_days)

    
    def cleaning_data(self, i):  
        data = self.filtered_data[i]
        self.day_detector(data)
        print(data)
        
        # Cleaning out double SequenceInJourneys
        sequenceJourneys = data.groupby('Hållplatsnamn').mean().sort_values(by=['SequenceInJourney'])['SequenceInJourney']    
        
        # Hitta de index med minimal och maximal sekvens
        sequenceJourneys = data.groupby('SequenceInJourney').mean().index.values.tolist()
        
        self.stop_names = []
        for i in range(len(sequenceJourneys)):
            for j in range(len(self.seq_list)):
                if sequenceJourneys[i] == self.seq_list[j]:
                    self.stop_names.append(self.stop_list[j])
                    break
        
        self.min_sequence = sequenceJourneys[0]
        self.max_sequence = sequenceJourneys[-1]
        
        if len(self.stop_names) > self.max_sequence:
            for i in range(0, len(sequenceJourneys)-1):
                if sequenceJourneys[i+1] - sequenceJourneys[i] == 0:
                    self.stop_names.pop(i+1)
        
        saved_rows = []
        counter = 0
        for i in range(0, len(data)):
            this_value = data.iloc[i]['SequenceInJourney']
            for j in range(counter, len(self.stop_names)):
                if this_value == sequenceJourneys[j]:
                    saved_rows.append(data.iloc[i])
                    counter = counter + 1
                    if counter == self.max_sequence:
                        counter = 0
                        j = 0
                    break
                else:
                    new_row = data.iloc[i].copy()
                    new_row['SequenceInJourney'] = sequenceJourneys[j]
                    new_row['Hållplatsnamn'] = self.stop_names[counter]
                    saved_rows.append(new_row)
                    counter = counter + 1
                    if counter == self.max_sequence:
                        counter = 0
                    j = 0   
        
        df = pd.DataFrame(saved_rows).reset_index(drop=True)
                      
        # Adjustment - ta bort rader med värde Ombord < 0 som icke är vid endstationer
        for i, j in df.iterrows():
            if (j.SequenceInJourney != self.max_sequence) and (j.SequenceInJourney != self.min_sequence):
                if j.Ombord < 0:
                    df._set_value(i,'Ombord', 0.0)
        
        self.prepared_dataset = []
        self.prepared_dataset.append(df)

       
    def scaling_data_forcast(self):
        # Scaling the training_set (use standardisation or normalization)
        # Since our output signal is continous (sigmoid function)
        # --> NORMALIZATION is recommended
        # Implement a scaler with value (range) between 0 and 1
        scaler = MinMaxScaler()
        self.scaler.append(scaler)
        self.scaled_dataset = []
        data = self.prepared_dataset[0][self.param].values.reshape(-1, 1)       
        data_scaled = scaler.fit_transform(data)
        self.scaled_dataset.append(data_scaled)
        

    def series_to_supervised(self):         
        dataset_scaled = self.scaled_dataset[0]
        self.len_pred = int(len(dataset_scaled)*0.2)
        print('len_pred (20% of dataset length): ', self.len_pred)
        print('Total dataset lenght ', len(self.prepared_dataset[0]))
        print('Lower Limit SHOULD BE prediction_window == ', self.prediction_window)
        print('Upper Limit SHOULD BE len(dataset)-len_pred == ', len(dataset_scaled)-self.len_pred)
        X_train = []
        y_train = []
        for i in range(self.len_pred, len(dataset_scaled)-self.len_pred):
            X_train.append(dataset_scaled[i-self.prediction_window:i,0])
            y_train.append(dataset_scaled[i,0])
         
        # Convert the lists X_train and y_train to np.arrays
        if self.len_pred > 50:
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            self.X_train = X_train
            self.y_train = y_train
            print('SHOULD BE ', len(dataset_scaled)-self.prediction_window-self.len_pred)
            print('series_to_supervised ', self.X_train.shape, self.y_train.shape) 
        else:
            print('Neglect dataset since batch selection is to small.')
        
        return self.len_pred
        
        
    def prepare_test_data_forcast(self):
        data = self.prepared_dataset[0] 
        inputs = data[self.param][len(data) - self.len_pred - self.prediction_window:].values
        inputs = inputs.reshape(-1,1)
        scaler = self.scaler[0]
        inputs_scaled = scaler.fit_transform(inputs)
        print('Lower Limit SHOULD BE prediction_window == ', self.prediction_window)
        print('Upper Limit SHOULD BE prediction_window + len_pred == ', self.prediction_window + self.len_pred)
        X_test = []
        X_date = []
        for i in range(self.prediction_window, self.prediction_window + self.len_pred):
            X_date.append(data[i-self.prediction_window:i])
            X_test.append(inputs_scaled[i-self.prediction_window:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        self.X_test = X_test
        print('prepare_test_data_forcast ', self.X_test.shape)  
        print('Should be len_pred == ', self.len_pred)
        print('X_date == ', X_date[-1])
   
    
    def create_neural_network(self):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout

        neurons = 512
        # Initializing the RNN 
        # The output is continous value --> use regression
        regressor = Sequential()
    
        # Add the first layer to the neural network
        regressor.add(LSTM(units = neurons, return_sequences = True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
    
        # Dropout regulation (use to be 20%)
        # --> this is the number of neurons to be ignored
        regressor.add(Dropout(0.2))
    
        # Add more layers to the neural network
        # Second layer
        regressor.add(LSTM(units = neurons, return_sequences = True))
        regressor.add(Dropout(0.2))
    
        # Third layer
        regressor.add(LSTM(units = neurons, return_sequences = True))
        regressor.add(Dropout(0.2))
    
        # Forth layer - last layer before output  layer
        regressor.add(LSTM(units = neurons))
        regressor.add(Dropout(0.2))
    
        # Output layer
        regressor.add(Dense(units = 1))
    
        # Compiling the RNN
        # NOTE: Optimizer - RMSprops is recommended for RNN but Adam was detected to 
        # be a better choice for this problem
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #mean_squared_error
        return regressor


    def train_predict_forcast(self, regressor, i, file_name):
        regressor.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=False)
        scaler = self.scaler[0]
        self.y_pred = regressor.predict(self.X_test)
        self.y_pred = self.y_pred.reshape(-1,1)
        self.y_pred_inv = scaler.inverse_transform(self.y_pred).round()
        print('len X_test ', len(self.X_test))
        
        df_result_forcats = self.prepare_datetime_forcast(i)
        print('len df_result_forcats ', len(df_result_forcats))
        df_result_forcats = df_result_forcats[0:self.len_pred].assign(prediction=self.y_pred_inv)
        #self.plot_ogt_result_verification(df_result_forcats)
        self.save_ogt_result_verification(df_result_forcats, file_name)
    
    
    def prepare_datetime_forcast(self, avgtid_i):
        from datetime import timedelta

        df_train_set = self.prepared_dataset[0]
    
        last_datetime = df_train_set.iloc[-1]['Tidpunkt']
        pd_last_datetime = pd.to_datetime(last_datetime)
        last_day = pd_last_datetime.date()
        
        next_day = 0
        for i in range(0, len(self.set_of_days)):
            if self.set_of_days[i] == last_day.weekday():
                next_day = (i+1) % (self.set_of_days[-1] + 1)
        
        for i in range(0, 6):
            td = timedelta(days=i)
            next_date = last_day + td
            if next_date.weekday() == next_day:
                break
            
        first_one = df_train_set.index[df_train_set['SequenceInJourney'] == 1].tolist()[0]
        time_serie_one_tour = df_train_set[first_one:first_one+self.max_sequence]['Tidpunkt']
        
        date_time = pd.to_datetime(time_serie_one_tour)
        date_time_forcast = []
        for i in range(0, len(date_time)):
            date_time_forcast.append(pd.to_datetime(last_datetime[0:10] + ' ' + str(date_time.iloc[i].time())))

        date_forcast = []
        stops_forcast = []
        sequence = []
        avgtid = []
        journeyGID = []
        line = []
        for i in range(1, self.len_pred):   
            for j in range(0, len(time_serie_one_tour)):
                td = timedelta(days=self.difference_tour_days)
                test = pd.to_datetime(str(next_date) + ' ' + str(date_time_forcast[j].time()))
                if next_date.weekday() <= self.set_of_days[-1]:
                    date_forcast.append(test)
                    stops_forcast.append(self.stop_names[j])
                    sequence.append(j+1)
                    journeyGID.append(self.journeyGID)
                    avgtid.append(self.journeyAVGtid_list[avgtid_i])
                    line.append(self.current_line)
                if len(date_forcast) == self.len_pred:
                    break
            next_date = next_date + td
            last_day = next_date
            
        df_result_forcats = pd.DataFrame(list(zip(sequence, line, journeyGID, avgtid, date_forcast, stops_forcast)),
               columns =['SequenceInJourney', 'Linje', 'JourneyGID', 'Avg.tid', 'Tidpunkt', 'Hållplatsnamn'])
        print(df_result_forcats)
        return df_result_forcats
    
    
    def plot_result_forcast(self, df_result_forcats):  
        stopname = df_result_forcats['Hållplatsnamn']
        X = np.arange(self.len_pred)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        fig.autofmt_xdate(rotation=90)
        ax.bar(X + 0.10, df_result_forcats['prediction'].values, color = 'g', width = 0.1)
    
        xftm = mdates.DateFormatter('%y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xftm)
        title = 'Forcast plot start' #'Avgång ' + str(self.prepared_dataset[0].iloc[0]['JourneyGID']) + ', Avg. tid ' + df_result_forcats[0]['Tidpunkt']
        plt.xlabel('Time')
        plt.ylabel('Antal passagerare')
        plt.title(title)
        plt.xticks(X, stopname)   
        plt.legend(['Prediktion passagerare'])
    
           
    def save_ogt_result_verification(self, pd_test_selection, file_name):
       
        # Calculate number of rows to delete
        index_adjustment = 0
        row = pd_test_selection.iloc[0]
        if row['SequenceInJourney'] != 1:
            index_adjustment = self.max_sequence - row['SequenceInJourney'] + 1
    
        pd_test_selection = pd_test_selection.drop(pd_test_selection.index[0:index_adjustment])
        
        # Adjustment 2 - inga påstigande vid endstation
        for i, j in pd_test_selection.iterrows():
            if j.SequenceInJourney == self.max_sequence:
                pd_test_selection._set_value(i,'prediction', 0.0)
        
        current_file = pd.read_csv(file_name)
        concat_file = pd.concat([current_file, pd_test_selection], axis=0)
        concat_file.to_csv(file_name, index=False)
     
        
    def plot_ogt_result_verification(self, pd_test_selection):
        
        # Calculate number of rows to delete
        index_adjustment = 0
        row = pd_test_selection.iloc[0]
        if row['SequenceInJourney'] != 1:
            index_adjustment = self.max_sequence - row['SequenceInJourney'] + 1
    
        pd_test_selection = pd_test_selection.drop(pd_test_selection.index[0:index_adjustment])
        
        # Adjustment 2 - inga påstigande vid endstation
        for i, j in pd_test_selection.iterrows():
            if j.SequenceInJourney == self.max_sequence:
                pd_test_selection._set_value(i,'prediction', 0.0)
                 
        min_index = pd_test_selection.index[pd_test_selection['SequenceInJourney'] == self.min_sequence].tolist()
        max_index = pd_test_selection.index[pd_test_selection['SequenceInJourney'] == self.max_sequence].tolist()
        
        start_index = 0
        stop_index = 0
        saved_tours = []
        
        for i in range(1, len(max_index)):
            tour = pd_test_selection.loc[min_index[start_index]:max_index[stop_index]]
            if len(tour) == 0:
                if max_index[stop_index + 1] > len(max_index):
                    break
                tour = pd_test_selection.loc[min_index[start_index]:max_index[stop_index + 1]]              
                if len(tour) == 0:
                    if max_index[stop_index + 2] > len(max_index):
                        break
                    tour = pd_test_selection.loc[min_index[start_index]:max_index[stop_index + 2]]
                    if len(tour) == 0:
                        if max_index[stop_index + 3] > len(max_index):
                            break
                        tour = pd_test_selection.loc[min_index[start_index]:max_index[stop_index + 3]]
              
            saved_tours.append(tour)
            start_index = start_index + 1
            stop_index = stop_index + 1
                   
         # Select all tours with a predicted overload of passenger
        tours_overload = []
        load_limit = 50 #(66 + 43)/2
        for i in range(0, len(saved_tours)):
            current_tour = saved_tours[i]['prediction'].values
            for j in range(0, len(current_tour)):
                if current_tour[j] >= load_limit:
                    tours_overload.append(saved_tours[i])
                    break
                     
        if len(tours_overload) == 0:
            print('---- NO OVERLOADS DETECTED ----')
            print('NOTE: journeyGID ' + str(pd_test_selection['JourneyGID'].iloc[0]) + ' avg. tid ' + str(pd_test_selection['Avg.tid'].iloc[0]) + ' has no overloaded tours')
        else:     
            for i in range(0, len(tours_overload)):
                stopname = tours_overload[i]['Hållplatsnamn']
                X = np.arange(len(tours_overload[i]))
                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                fig.autofmt_xdate(rotation=90)
                ax.bar(X + 0.10, tours_overload[i]['prediction'].values, color = 'g', width = 0.1)
            
                xftm = mdates.DateFormatter('%y-%m-%d %H:%M:%S')
                ax.xaxis.set_major_formatter(xftm)
                title = 'journeyGID ' + str(tours_overload[i]['JourneyGID'].iloc[0]) + ', Avg.tid ' + str(tours_overload[i]['Avg.tid'].iloc[0]) + ', Tidpunkt ' + str(tours_overload[i]['Tidpunkt'].iloc[0])[0:10]    
                plt.xlabel('Time')
                plt.ylabel('Antal passagerare')
                plt.title(title)
                plt.xticks(X, stopname)   
                plt.legend(['Prediktion passagerare'])
            
            