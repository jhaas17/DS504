import pandas as pd
import numpy as np
import datetime
import math
import torch
from geopy.distance import geodesic

from model import NeuralNet

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def process_data(data):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
            [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.
    
    """

    data_subs = pd.DataFrame()

    trajectory = []
    empty_trajectories = []
    empty_time = 0
    empty_distance = 0
    num_empty=0
    occupied_time = 0
    occupied_trajectories = []
    occupied_distance = 0
    num_trips = 0
    s = data[0][3]
    starting_time = datetime.datetime.strptime(data[0][2], '%Y-%m-%d %H:%M:%S').time()

    for index, row in enumerate(data):
        longitude = row[0]
        latitude = row[1]
        date_time = datetime.datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
        time = date_time.time()
        date = date_time.date()
        current_status = row[3]

        if current_status != s or index == len(data)-1:
            if current_status == 0:
                empty_trajectories.extend(trajectory)
                num_empty+=1
            elif current_status==1:
                occupied_trajectories.extend(trajectory)
                num_trips+=1
            s = current_status
            trajectory = []
        if current_status==0 and index!=len(data)-1:
            empty_time += (datetime.datetime.combine(date,datetime.datetime.strptime(data[index+1][2], '%Y-%m-%d %H:%M:%S').time()) - datetime.datetime.combine(date,time)).total_seconds()/60
            empty_trajectories.append(np.array([longitude, latitude, (datetime.datetime.combine(date,time)- datetime.datetime.combine(date,starting_time)).total_seconds()], dtype=np.double))
            empty_distance += geodesic((latitude,longitude),(data[index+1][1], data[index+1][0])).km
        elif current_status==1 and index!=len(data)-1:
            occupied_time += (datetime.datetime.combine(date,datetime.datetime.strptime(data[index+1][2], '%Y-%m-%d %H:%M:%S').time()) - datetime.datetime.combine(date,time)).total_seconds()/60
            occupied_trajectories.append(np.array([longitude, latitude, (datetime.datetime.combine(date,time)- datetime.datetime.combine(date,starting_time)).total_seconds()], dtype=np.double))
            occupied_distance += geodesic((latitude,longitude),(data[index+1][1], data[index+1][0])).km

    start_time = -(datetime.datetime.combine(date,datetime.time(hour=0, minute=0, second=0)) - datetime.datetime.combine(date,starting_time)).total_seconds()/60
    end_time = -(datetime.datetime.combine(date,datetime.time(hour=0, minute=0, second=0)) - datetime.datetime.combine(date,time)).total_seconds()/60

    if num_empty == 0:
        empty_avg_time = 0
        empty_avg_dist = 0
    else:
        empty_avg_time = empty_time/num_empty
        empty_avg_dist = empty_distance/num_empty
    if num_trips == 0:
        occupied_avg_time = 0
        occupied_avg_dist = 0
    else:
        occupied_avg_time = occupied_time/num_trips
        occupied_avg_dist = occupied_distance/num_trips
    data_subs = data_subs.append({'empty_time':empty_avg_time,'occupied_time':occupied_avg_time,'empty_avg_dist':empty_avg_dist, 'occupied_avg_dist':occupied_avg_dist,
                                  'start_time':start_time,'end_time':end_time,'num_trips':num_trips}, ignore_index=True)

    return data_subs.to_numpy()[0]

def run(data, model):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    
    """
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(data).to(device)
        out = model(features)
        y = torch.argmax(out).item()
    
    return y


