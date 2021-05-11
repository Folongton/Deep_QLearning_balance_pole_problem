import gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import namedtuple
from itertools import count
from PIL import Image

import tensorflow as tf

# Hyperparameters
batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100_000
lr = 0.001
num_episodes = 1000

def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.show(block=True)
        
def get_moving_average(period, values): 
    values = pd.Series(values)
    ma = values.rolling(period).mean()
    ma = ma.fillna(0)
    return ma

plot(np.random.rand(300),100)