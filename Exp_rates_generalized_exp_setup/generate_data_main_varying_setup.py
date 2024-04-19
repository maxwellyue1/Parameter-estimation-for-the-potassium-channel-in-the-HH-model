import numpy as np
import os
from exp_generate_data import exp_formalism_data_generator
from exp_hh_model import HH_model_exp

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
exp_setupNbounds = {'prestep_V_bounds': (-140, -60), 'step_Vs_lb': (-50, 10), 'dV_bd': (8, 12), 'n_traces': 8, 't': np.arange(0.0, 6.0, 0.00001)}

# sim_setup = {'prestep_V': -100, 'step_Vs': np.linspace(-20, 100, 13), 't': np.arange(0.0, 6.0, 0.01)}

data_generator = exp_formalism_data_generator(params_bounds, exp_setupNbounds)
data_generator.generate_data(1000)

script_dir = os.getcwd()
file_name = "dataset_00001_test.csv"
file_path = os.path.join(script_dir, file_name)
if os.path.isfile(file_path):
    # file exists
    data_generator.save_tubular_data(file_path)
else:
    data_generator.create_empty_csv(file_path)