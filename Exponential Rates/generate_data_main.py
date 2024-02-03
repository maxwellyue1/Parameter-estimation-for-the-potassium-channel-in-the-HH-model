import numpy as np
import os
from exp_generate_data import exp_formalism_data_generator
from exp_hh_model import HH_model_exp

params_bounds = {'p': (1, 5), 
                 'g_max': (100, 140), 
                 'E_rev': (-100, -60), 
                 'a_m': (0, 100), 
                 'b_m': (0, 100), 
                 'delta_m': (0, 1), 
                 's_m': (-100, 0)}
sim_setup = {'prestep_V': -100, 'step_Vs': np.linspace(-20, 100, 13), 't': np.arange(0.0, 6.0, 0.01)}

data_generator = exp_formalism_data_generator(params_bounds, sim_setup)
data_generator.generate_data(1000)

file_name = "dataset_exp_test.csv"
if os.path.isfile(file_name):
    # file exists
    data_generator.save_tubular_data(file_name)
else:
    data_generator.create_empty_csv(file_name)