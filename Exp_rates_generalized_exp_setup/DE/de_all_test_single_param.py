from dataset_reader import Traces_Dataset
from DE_obj_model import de_obj_model   
from exp_hh_model import HH_model_exp
import numpy as np
from scipy.optimize import differential_evolution

dataset = Traces_Dataset('dataset_exp_new.csv')

params = dataset.params.numpy()
current_traces = dataset.current_traces.numpy()
time_traces = dataset.time_traces.numpy()


def obj_i_param(x, *args): 
    '''
    x: a single float value of the i-th param
    args: a tupleo f additional fixed parameters (prestep_V, step_V0, time_traces)
    *args=(i, sim_setup_2d, target_params)
    '''
    params = args[2].copy()
    params[args[0]] = x
    trail_model = de_obj_model(params, args[1])
    trail_traces = trail_model.simulation()
    # print(trail_traces[1])
    target_model = de_obj_model(args[2], args[1])
    target_traces = target_model.simulation()
    # print(target_traces[1]) 
    
    return np.sum(np.square(trail_traces - target_traces))

params_searching_bounds = {
    'p': (1, 5),
    'g_max': (100, 140), 
    'E_rev': (-100, -60), 
    'a_m': (0, 13), 
    'b_m': (0, 100), 
    'delta_m': (0, 1), 
    's_m': (-17, -10)
}

mses_ith_samples_list=[]
nit_ith_samples_list = []

for i in range(dataset.num_params):
    i = 3
    key_at_index = list(params_searching_bounds.keys())[i]
    bounds = [params_searching_bounds[key_at_index]]

    mses_ith_samples = []
    nit_ith_samples = []
    for sample in range(dataset.__len__()): 

        prestep_V_2d = dataset.prestep_V[sample].numpy().reshape(-1,1)
        step_Vs_2d = (np.arange(dataset.step_V1[sample].numpy(), dataset.step_V1[sample].numpy() + dataset.num_traces*10, 10)).reshape(-1,1)
        t = time_traces[sample]
        # target_traces = current_traces[sample]
        target_params = params[sample]

        # sim setup for obj evaluation model
        sim_setup_2d = {'prestep_V': prestep_V_2d, 'step_Vs': step_Vs_2d, 't': t}   

        result = differential_evolution(obj_i_param, bounds, args=(i, sim_setup_2d, target_params), maxiter=5000)
        mse_ith = (target_params[i] - result.x) ** 2
        mses_ith_samples.append(mse_ith)
        nit_ith = result.nit
        nit_ith_samples.append(nit_ith)

    mses_ith_samples = np.array(mses_ith_samples)
    nit_ith_samples = np.array(nit_ith_samples).reshape(-1, 1)
    print(mses_ith_samples.shape, nit_ith_samples.shape)
    mses_ith_samples_list.append(mses_ith_samples)
    nit_ith_samples_list.append(nit_ith_samples)
    print(len(mses_ith_samples_list), len(nit_ith_samples_list))

mses_ith_samples_mat = np.hstack(mses_ith_samples_list)
nit_ith_samples_mat = np.hstack(nit_ith_samples_list)
print(mses_ith_samples_mat.shape, nit_ith_samples_mat.shape)
np.save('de_mses_on_ith_params.npy', mses_ith_samples_mat)
np.save('de_nit_on_ith_params.npy', nit_ith_samples_mat)