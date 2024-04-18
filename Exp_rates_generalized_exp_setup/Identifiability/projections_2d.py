import matplotlib.pyplot as plt

import itertools
import numpy as np

import sys
sys.path.append('..')
from dataset_reader import Traces_Dataset

from exp_generate_data import exp_formalism_data_generator
from exp_hh_model import HH_model_exp

dataset = Traces_Dataset('../dataset_test.csv')

target_num = 9
target_prestep_V = int(dataset.prestep_V[target_num])
target_step_V1 = int(dataset.step_Vs[target_num].numpy()[0])
target_dV = dataset.step_Vs[target_num].numpy()[1] - dataset.step_Vs[target_num].numpy()[0]

t = np.arange(0.0, 6.0, 0.01)
target_params = dataset.params[target_num].numpy()

sim_setup = {'prestep_V': target_prestep_V, 'step_Vs': dataset.step_Vs[target_num].numpy(), 't':t}
params = {'p': target_params[0], 
          'g_max': target_params[1], 
          'E_rev': target_params[2], 
                 'a_m': target_params[3], 
                 'b_m': target_params[4], 
                 'delta_m': target_params[5], 
                 's_m': target_params[6]}
print(params)

model = HH_model_exp(params, sim_setup)
model.simulation()
if model.check_current_ss() and model.check_steady_state_curve(): 
    target_max_ind_arr = model.max_index_array
    target_current_traces = model.current_traces

# p inclusive on both ends
params_bounds = {'p': (1, 4), 
                 'g_max': (100, 140), 
                 'E_rev': (-100, -60), 
                 'a_m': (0, 100), 
                 'b_m': (0, 100), 
                 'delta_m': (0, 1), 
                 's_m': (-100, 0)}

# prestep_V bounds -> only take values from lb to up with increment of 10
# step_Vs bounds -> the lower bounds on step_Vs, generate an array of step_Vs using increment of 10 with 11 elements. 
# dV_bd inclusive on both ends

exp_setupNbounds = {'prestep_V_bounds': (target_prestep_V, target_prestep_V), 'step_Vs_lb': (target_step_V1, target_step_V1), 'dV_bd': (target_dV, target_dV), 'n_traces': 8, 't': t}

# sim_setup = {'prestep_V': -100, 'step_Vs': np.linspace(-20, 100, 13), 't': np.arange(0.0, 6.0, 0.01)}

data_generator = exp_formalism_data_generator(params_bounds, exp_setupNbounds)
n_samples = 60000
data_generator.generate_data_dotty_plots(n_samples)

max_ind_arrs = data_generator.dataset[:, :8]
current_traces = data_generator.dataset[:, 8:-7].reshape(-1, 8, 600)
params = data_generator.dataset[:, -7:]

def obj_trace(sample, trace):
    '''
    find the area difference in the rising phase instead of the whole simulation. 
    This can be done by defining a window using the earlier threshold between two sample traces. 
    the fct is comparing two traces from the same experiment in two different samples. 
    '''
    # find the earlier threshold index + 1. 
    stop_ind = int(min(max_ind_arrs[sample, trace], target_max_ind_arr[trace]))
    #print(sample1, sample2, trace)
    #print(np.where(selected_t[sample1, trace] == -1)[0][0], np.where(selected_t[sample2, trace] == -1)[0][0])
    # we calculate the (avg over stop_ind points) 

    # diff_area_one_trace = np.trapz(np.abs(selected_current_traces_3d[sample1, trace, :stop_ind] - selected_current_traces_3d[sample2, trace, :stop_ind]), dataset_generator.t[:stop_ind]) / np.trapz(np.maximum(selected_current_traces_3d[sample1, trace, :stop_ind], selected_current_traces_3d[sample2, trace, :stop_ind]), dataset_generator.t[:stop_ind])
    diff_area_one_trace = np.trapz(np.abs(current_traces[sample, trace, :stop_ind] - target_current_traces[trace, :stop_ind]), t[:stop_ind]) / np.trapz(np.maximum(current_traces[sample, trace, :stop_ind], target_current_traces[trace, :stop_ind]), t[:stop_ind])
    
    
    return diff_area_one_trace

def obj(sample): 
    diff_area = np.mean([obj_trace(sample, trace) for trace in range(target_current_traces.shape[0])])
    # diff_params_square = np.sum(np.square(selected_params[sample1] - selected_params[sample2]))
    return diff_area #, diff_params_square

fitness_list = [obj(sample) for sample in range(n_samples)]
param_lists = []
for i in range(params.shape[1]):
    param_list = [params[sample, i] for sample in range(n_samples)]
    param_lists.append(param_list)



arr = np.array([0,1,2,3,4,5,6])

# Using itertools.combinations
combinations = list(itertools.combinations(arr, 2))

print(combinations)

# This is just an example, you should modify this according to your specific intervals
intervals = np.linspace(np.min(fitness_list), np.max(fitness_list), 5)  # Define your intervals here
colors = ['red', 'green', 'blue', 'yellow', 'black']  # Add more colors if needed

def assign_color(value):
    for i in range(len(intervals)-1):
        if intervals[i] <= value < intervals[i+1]:
            return colors[i]
    return colors[-1]  # Default color if value doesn't fall into any interval

# Create a subplot grid with 7 rows and 3 columns
fig, axs = plt.subplots(7, 3, figsize=(15, 30))

for i, ax_row in enumerate(axs):
    for j, ax in enumerate(ax_row):
        idx = i * len(ax_row) + j  # Calculate the index of the subplot
        if idx < len(combinations):
            x_data = np.array(param_lists[combinations[idx][0]])
            y_data = np.array(param_lists[combinations[idx][1]])
        
            point_colors = [assign_color(value) for value in fitness_list]
        
            ax.scatter(x_data, y_data, c=point_colors, alpha=0.9, s=1)
            # ax.set_title(f'Combination {combinations[idx][0]},{combinations[idx][1]}')
            ax.set_xlabel(f'{list(params_bounds.keys())[combinations[idx][0]]}')  # Add x-axis label
            ax.set_ylabel(f'{list(params_bounds.keys())[combinations[idx][1]]}')  # Add y-axis label
        else:
            ax.axis('off')  # Hide empty subplots

plt.tight_layout()

plt.savefig(f'2dprojection_on{target_num}_{n_samples}samples.png')
plt.show()

np.save(f'2d_param_lists_on_{target_num}_{n_samples}samples.npy', np.array(param_lists))
np.save(f'2d_fitness_list_on_{target_num}_{n_samples}samples.npy', np.array(fitness_list))