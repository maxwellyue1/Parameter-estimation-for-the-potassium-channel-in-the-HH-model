import csv
import os
import random
import numpy as np
import pandas as pd
from traces_dataset import Traces_Dataset
import torch 
from GA_test import ga_algorithm

dataset = Traces_Dataset('dataset_test.csv')

target_time_traces_set = dataset.time_traces.numpy()
target_current_traces_set = dataset.current_traces.numpy()
target_params_set = dataset.params.numpy()

print(target_time_traces_set.shape, target_current_traces_set.shape, target_params_set.shape)

for i in range(target_params_set.shape[0]): 
    ga_algorithm(target_time_traces_set[i], target_current_traces_set[i], target_params_set[i])


