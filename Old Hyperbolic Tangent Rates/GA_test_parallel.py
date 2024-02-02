import csv
import os
import random
import numpy as np
import pandas as pd
from traces_dataset import Traces_Dataset
import torch
from GA_function import ga_algorithm
from multiprocessing import Pool
import time

def run_ga(index):
    start_time = time.time()
    ga_algorithm(target_time_traces_set[index],
                target_params_set[index])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"GA estimation for index {index} took {elapsed_time} seconds")

if __name__ == "__main__":
    dataset = Traces_Dataset('dataset_test.csv')

    target_time_traces_set = dataset.time_traces.numpy()
    target_params_set = dataset.params.numpy()

    print(target_time_traces_set.shape, target_params_set.shape)

    # Number of cores to use
    num_cores = os.cpu_count()

    # Create a pool of processes
    with Pool(processes=num_cores) as pool:
        # Record start time for total time calculation
        total_start_time = time.time()
        
        # Map the run_ga function to each index in parallel
        pool.map(run_ga, range(target_params_set.shape[0]))

        # Record end time for total time calculation
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time
        print(f"Total time for estimating the whole dataset: {total_elapsed_time} seconds")
