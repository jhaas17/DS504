"""
  Input:
      Traj: a list of list, contains one trajectory for one driver 
      example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
         [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
  Output:
      Data: any format that can be consumed by your model.
  
  """
from utils import *
import numpy as np 
import tensorflow as tf
def process_data(traj_1, traj_2, subtraj_len=64, use_sample=True):
    subtraj_pairs = []

    max_traj_len = min(len(traj_1),len(traj_2))

    max_sub_traj = np.floor(max_traj_len/subtraj_len)
    traj1 = traj_1[:max_traj_len]
    traj2 = traj_2[:max_traj_len]


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
            subtraj1=[]
            subtraj2=[]
            count+=1
        if count == max_sub_traj: 
            break
    if use_sample:
        idx = np.random.choice(range(len(subtraj_pairs)),10)
        final = np.array(subtraj_pairs)[idx]
    else: 
        final = np.array(subtraj_pairs)
    return final

def run(data, model,sess):
    """
    
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    
    """
    pred = np.zeros(2)
    for pair in data: 
        yhat = model.predict([np.array([pair[0],]),np.array([pair[1],])],verbose=0)[0]
        y = 1 if yhat >= 0.5 else 0 
        pred[y] += 1

    prediction = np.argmax(pred)

    return int(prediction)

