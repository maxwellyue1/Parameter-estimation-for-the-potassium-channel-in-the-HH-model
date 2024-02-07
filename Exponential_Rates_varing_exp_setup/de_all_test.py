from dataset_reader import Traces_Dataset
from DE_obj_model import de_obj_model   
from exp_hh_model import HH_model_exp
import numpy as np
from scipy.optimize import differential_evolution
# import matplotlib.pyplot as plt

dataset = Traces_Dataset('dataset_exp_new.csv')

params = dataset.params.numpy()
current_traces = dataset.current_traces.numpy()
time_traces = dataset.time_traces.numpy()

def obj(x, *args): 
    '''
    x: a 1-D array of the variables for the obj function (the parameters we are estimating)
    args: a tupleo f additional fixed parameters (prestep_V, step_V0, time_traces)
    *args=(sim_setup_2d, target_params)
    '''
    trail_model = de_obj_model(x, args[0])
    trail_traces = trail_model.simulation()
    # print(trail_traces[1])
    target_model = de_obj_model(args[1], args[0])
    target_traces = target_model.simulation()
    # print(target_traces[1]) 
    
    return np.sum(np.square(trail_traces - target_traces))

# these bounds are from the distribution of the params in the dataset used for NN training
params_searching_bounds = {
    'p': (1, 5),
    'g_max': (100, 140), 
    'E_rev': (-100, -60), 
    'a_m': (0, 13), 
    'b_m': (0, 100), 
    'delta_m': (0, 1), 
    's_m': (-17, -10)
}
bounds = [params_searching_bounds['p'], params_searching_bounds['g_max'], params_searching_bounds['E_rev'], params_searching_bounds['a_m'], params_searching_bounds['b_m'], params_searching_bounds['delta_m'], params_searching_bounds['s_m']]

mse_list = []

for sample in range(dataset.__len__()): 
    
    prestep_V_2d = dataset.prestep_V[sample].numpy().reshape(-1,1)
    step_Vs_2d = (np.arange(dataset.step_V1[sample].numpy(), dataset.step_V1[sample].numpy() + dataset.num_traces*10, 10)).reshape(-1,1)
    t = time_traces[sample]
    # target_traces = current_traces[sample]
    target_params = params[sample]

    # sim setup for obj evaluation model
    sim_setup_2d = {'prestep_V': prestep_V_2d, 'step_Vs': step_Vs_2d, 't': t}   

    result = differential_evolution(obj, bounds, args=(sim_setup_2d, target_params), maxiter=100)
    mse = (target_params - result.x) ** 2
    mse_list.append(mse)

mse_mat = np.vstack(mse_list)
np.save('de_mse_params_all_test.npy', mse_mat)