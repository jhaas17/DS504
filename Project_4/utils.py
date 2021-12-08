import pandas as pd
import numpy as np
import datetime
import random

def make_pairs(data):
    traj_pairs = []
    labels = []
    #iterate through each key
    for key in data.keys():
        #iterate through each trajectory
        for i, traj in enumerate(data[key]):
            #drop plates
            traj = [point[1:] for point in traj]

            #pick a random SAME class trajectory
            same_idx = np.random.choice([x for x in range(len(data[key])) if x!=i])
            same_traj = [point[1:] for point in data[key][same_idx]]
            traj_pairs.append([traj, same_traj])
            labels.append(1)

            #pick a random key of DIFFERENT class and random trajectory
            neg_key = np.random.choice([x for x in data.keys() if x!=key])
            neg_idx = np.random.choice(len(data[neg_key]))

            neg_traj = [point[1:] for point in data[neg_key][neg_idx]]

            traj_pairs.append([traj, neg_traj])
            labels.append(0)

    return traj_pairs, labels

def process_data_train(data, labels, subtraj_len):

    subtraj_pairs = []
    subtraj_labels = []

    #iterate through each pair of trajectories
    for i, pair in enumerate(data):
        max_traj_len = min(len(pair[0]),len(pair[1]))
        
        max_sub_traj = np.floor(max_traj_len/subtraj_len)
        traj1 = pair[0][:max_traj_len]
        traj2 = pair[1][:max_traj_len]

        
        subtraj1 = []
        subtraj2 = []
        count = 0
        for j in range(max_traj_len):
            if len(subtraj1)<subtraj_len :
                subtraj1.append(np.array(standardize(traj1[j][0],traj1[j][1],traj1[j][2],traj1[j][3],traj1[j][4])))
                subtraj2.append(np.array(standardize(traj2[j][0],traj2[j][1],traj2[j][2],traj2[j][3],traj2[j][4])))
            else:
                subtraj1 = np.array(subtraj1)
                subtraj2 = np.array(subtraj2)
                subtraj_pairs.append([np.stack(subtraj1), np.stack(subtraj2)])
                subtraj_labels.append(labels[i])
                subtraj1=[]
                subtraj2=[]
                count+=1
            if count == max_sub_traj: 
                break

    return np.array(subtraj_pairs), np.array(subtraj_labels)

def standardize(long, lat, sec, stat, time):
    mean_long = 114.04668551547071
    std_long = 0.10119329191141248
    mean_lat = 22.59044789012173
    std_lat = 0.06865295391399516

    long = (long - mean_long)/std_long 
    lat = (lat-mean_lat)/std_lat 

    sec = sec/ (24*60*60)

    # time = extract_time(time)
    return [long,lat,sec,stat]

def get_maxmin_long_lat(data):
    longs = []
    lats = []
    traj_lens = []
    for key in data.keys():
        for i, traj in enumerate(data[key]):
            traj_lens.append(len(traj))
            print(np.array(traj).shape)
            for point in traj:
                longs.append(point[1])
                lats.append(point[2])
    print("Mean Long: " + str(np.mean(longs)))
    print("Std Long: " + str(np.std(longs)))
    print("Mean Lat: " + str(np.mean(lats)))
    print("Std Lat: " + str(np.std(lats)))
    print("Max trajectory: " + str(max(traj_lens)))

    