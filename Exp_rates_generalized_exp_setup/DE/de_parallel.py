import numpy as np
import csv
import os
import sys  # Import the sys module
import time
from functools import partial
from multiprocessing import Pool
from scipy.optimize import differential_evolution

sys.path.append('..')
from dataset_reader import Traces_Dataset
from DE_obj_model import de_obj_model   
from exp_hh_model import HH_model_exp

dataset = Traces_Dataset('../dataset_test.csv')

params = dataset.params.numpy()
current_traces = dataset.current_traces.numpy()
time_traces = dataset.time_traces.numpy()

prestep_V_vec = dataset.prestep_V.numpy()
step_Vs_vec = dataset.step_Vs.numpy()

def obj(x, *args): 
    '''
    x: a 1-D array of the variables for the obj function (the parameters we are estimating)
    *args=(sim_setup_2d, target_current_trances)
    '''
    trail_model = de_obj_model(x, args[0])
    trail_traces = trail_model.simulation()
    # print(trail_traces[1])
    target_model = de_obj_model(args[1], args[0])
    target_traces = target_model.simulation()
    # print(target_traces[1]) 

    fit = np.sum(np.square(trail_traces - target_traces))
    # relative_error = fit/np.sum(np.square(target_traces))
    
    return fit


# these bounds are from the distribution of the params in the dataset used for NN training
params_searching_bounds = {
    'p': (1, 4),
    'g_max': (100, 140), 
    'E_rev': (-100, -60), 
    'a_m': (0, 100), 
    'b_m': (0, 100), 
    'delta_m': (0, 1), 
    's_m': (-100, 0)
}
bounds = [params_searching_bounds['p'], params_searching_bounds['g_max'], params_searching_bounds['E_rev'], params_searching_bounds['a_m'], params_searching_bounds['b_m'], params_searching_bounds['delta_m'], params_searching_bounds['s_m']]

hyperparameters_grid = {
    'strategy': ['best1bin', 'best1exp', 'rand1exp', 'rand1exp', 
                'rand2bin', 'rand2exp', 'best2bin', 'best2exp',
                'randtobest1bin', 'randtobest1exp',
                'currenttobest1bin', 'currenttobest1exp'],
    'popsize': [21], #[14,28,42,56,70],  # Example popsize hyperparameter
    'mutation': [(0.1, 0.9)],  # Example mutation hyperparameter
    'recombination': [0.9],  # Example recombination hyperparameter
    'init': ['latinhypercube'],  # Example init hyperparameter
}


csv_filename = "de_experiment_results_parrallell_try_1000_allmetrics.csv"
# Define the headers for the CSV file
csv_headers = ['Strategy', 'Popsize', 'MSE Overall Avg', 'MSE Overall Std', 'RMSE Overall Avg', 'RMSE Overall Std', 'MAE Overall Avg', 'MAE Overall Std', 'MAPE Overall Avg', 'MAPE Overall Std', 'Elapsed Time Avg',  'Elapsed Time Std']

# Check if the CSV file exists; if not, create and write the headers
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)


def process_sample(sample, strategy, popsize, mutation, recombination, init):
    prestep_V_2d = prestep_V_vec[sample].reshape(-1,1)
    step_Vs_2d = step_Vs_vec[sample].reshape(-1,1)
    t = time_traces[sample]
    # target_traces = current_traces[sample]
    target_params = params[sample]

    # sim setup for obj evaluation model
    sim_setup_2d = {'prestep_V': prestep_V_2d, 'step_Vs': step_Vs_2d, 't': t}   

    start_time = time.time()
    result = differential_evolution(obj, bounds, args=(sim_setup_2d, target_params), strategy=strategy, popsize=popsize, mutation=mutation, recombination=recombination, init=init, seed=42, maxiter=300, tol=-1)
    end_time = time.time()
    
    mse = (target_params - result.x) ** 2
    rmse = np.sqrt(mse)
    mae = np.abs(target_params - result.x)
    mape = np.abs((target_params - result.x) / result.x)

    elapsed_time = end_time - start_time
    return sample, mse, rmse, mae, mape, elapsed_time

if __name__ == '__main__':
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        pool = Pool()  # Creates a pool of processes
        
        for strategy in hyperparameters_grid['strategy']:
            for popsize in hyperparameters_grid['popsize']:
                for mutation in hyperparameters_grid['mutation']:
                    for recombination in hyperparameters_grid['recombination']:
                        for init in hyperparameters_grid['init']:
                            # Use partial to fix hyperparameters for the current loop iteration
                            process_func = partial(process_sample, strategy=strategy, popsize=popsize, mutation=mutation, recombination=recombination, init=init)
                            
                            # Map the process function to the sample range using the multiprocessing pool
                            results = pool.map(process_func, range(1000))

                            mse_list = []
                            rmse_list = []
                            mae_list = []
                            mape_list = []
                            time_list = []
                            for result in results:
                                sample, mse, rmse, mae, mape, elapsed_time = result
                                mse_list.append(mse)
                                rmse_list.append(rmse)
                                mae_list.append(mae)
                                mape_list.append(mape)
                                time_list.append(elapsed_time)

                            mse_mat = np.vstack(mse_list)
                            rmse_mat = np.vstack(rmse_list)
                            mae_mat = np.vstack(mae_list)
                            mape_mat = np.vstack(mape_list)
                            time_mat = np.array(time_list).reshape(-1, 1)

                            mse_overall_avg = np.mean(mse_mat)
                            mse_overall_std = np.std(np.mean(mse_mat, axis=1))

                            rmse_overall_avg = np.mean(rmse_mat)
                            rmse_overall_std = np.std(np.mean(rmse_mat, axis=1))

                            mae_overall_avg = np.mean(mae_mat)
                            mae_overall_std = np.std(np.mean(mae_mat, axis=1))

                            mape_overall_avg = np.mean(mape_mat)
                            mape_overall_std = np.std(np.mean(mape_mat, axis=1))

                            time_overall_avg = np.mean(time_mat)
                            time_overall_std = np.std(time_mat)
                            
                            writer.writerow([strategy, popsize, mse_overall_avg, mse_overall_std, rmse_overall_avg, rmse_overall_std, mae_overall_avg, mae_overall_std, mape_overall_avg, mape_overall_std, time_overall_avg, time_overall_std])
        
        pool.close()  # Close the pool
        pool.join()   # Wait for all processes to finish before exiting

