import numpy as np
from exp_hh_model import HH_model_exp
import csv

class exp_formalism_data_generator:
    '''
    This  class should be initiated by a set of parameters, and simulation setups
    To generate a number of sample by check the model conditions iteratively
    able to select a number of points using linear interpolation. 
    '''
    def __init__(self, params_bounds, exp_setupNbounds, n_points=20):
        self.params_bounds = params_bounds

        self.prestep_V_bounds = exp_setupNbounds['prestep_V_bounds']
        self.step_Vs_lb = exp_setupNbounds['step_Vs_lb']
        self.n_traces = exp_setupNbounds['n_traces']
        self.t = exp_setupNbounds['t']
        self.dV_bd = exp_setupNbounds['dV_bd']

        self.n_points = n_points

        # make sure the prestep_V < V_0
        assert self.prestep_V_bounds[1] < self.step_Vs_lb[0]
        # make sure the |V_n - V_0| >= 50
        assert self.dV_bd[0] * (self.n_traces - 1) >= 50
    

    def generate_data(self, n_sample):
        count = 0
        self.dataset = np.empty((n_sample, (1 + self.n_traces) + (self.n_points * self.n_traces * 2) + len(self.params_bounds)))

        while count < n_sample: 
            sim_setup = {'t':self.t}
            params = {}

            # get the sim_setup
            #     are taken to be a multiple of 2 within the bounds defined.
            # prestep_V, dV and V_0  can be any int within the predefined range. 

            # prestep_V = np.random.randint(self.prestep_V_bounds[0]//2, self.prestep_V_bounds[1]//2 + 1) * 2
            prestep_V = np.random.randint(self.prestep_V_bounds[0], self.prestep_V_bounds[1] + 1)
            step_V1 = np.random.randint(self.step_Vs_lb[0], self.step_Vs_lb[1] + 1)
            dV = np.random.randint(self.dV_bd[0], self.dV_bd[1] + 1)
            step_Vs = np.arange(step_V1, step_V1 + self.n_traces*dV, dV)

            sim_setup['prestep_V'] = prestep_V
            sim_setup['step_Vs'] = step_Vs
            self.dataset[count, :(1 + self.n_traces)] = np.insert(step_Vs, 0, prestep_V)

            # get the params
            for p_name in list(self.params_bounds.keys()): 
                if p_name == 'p': 
                    params[p_name] = np.random.randint(self.params_bounds[p_name][0], self.params_bounds[p_name][1] + 1) # handle the numpy take open upper bd
                else:
                    params[p_name] = np.random.uniform(self.params_bounds[p_name][0], self.params_bounds[p_name][1])

            model = HH_model_exp(params, sim_setup)
            model.simulation()

            if model.check_current_ss() and model.check_steady_state_curve(): 
                collected_t, collected_current_traces = model.collect_points(self.n_points)
                # print(collected_t.shape, collected_current_traces.shape, np.array(list(params.values())).shape)
                data = np.concatenate((collected_t.flatten(), collected_current_traces.flatten(), np.array(list(params.values()))))
                self.dataset[count, (1 + self.n_traces):] = data

                count += 1

        return self.dataset.shape

   
    def create_empty_csv(self, file_name = 'dataset_exp.csv'):
        names = []
        names.append('prestep_V')
        for step in range(self.n_traces): 
            names.append(f'step_V{step}')
        for i in range(self.n_traces):
            for point in range(self.n_points): 
                t_name = f't^{int(i)}_{int(point)}'
                names.append(t_name)
        for i in range(self.n_traces):
            for pt in range(self.n_points): 
                names.append(f'I^{int(i)}_{int(pt)}')

        names += list(self.params_bounds.keys())

        # Open the CSV file in write mode to create an empty file
        # with open(csv_file_path, mode="w", newline="") as file:
        #     pass  # This just creates an empty file

        with open(file_name, mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            # Write each row of the array to the CSV file
            csv_writer.writerow(names)
        
    
    def save_tubular_data(self, file_name = 'dataset_exp.csv'):
        # Open the CSV file in write mode
        with open(file_name, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            # Write each row of the array to the CSV file
            for row in self.dataset:
                csv_writer.writerow(row)

    
    def generate_data_dotty_plots(self, n_sample):
        '''
        generate data for dotty plots. 
        (n_samples, n_traces + n_traces * len(t) + len(params_bounds))
                    max_index_array, current_traces, params
        '''
        count = 0
        self.dataset = np.empty((n_sample, self.n_traces + self.n_traces * len(self.t) + len(self.params_bounds)))
        # print(self.dataset.shape)
        while count < n_sample: 
            sim_setup = {'t':self.t}
            params = {}

            # get the sim_setup
            #     are taken to be a multiple of 2 within the bounds defined.
            # prestep_V, dV and V_0  can be any int within the predefined range. 

            # prestep_V = np.random.randint(self.prestep_V_bounds[0]//2, self.prestep_V_bounds[1]//2 + 1) * 2
            prestep_V = np.random.randint(self.prestep_V_bounds[0], self.prestep_V_bounds[1] + 1)
            step_V1 = np.random.randint(self.step_Vs_lb[0], self.step_Vs_lb[1] + 1)
            dV = np.random.randint(self.dV_bd[0], self.dV_bd[1] + 1)
            step_Vs = np.arange(step_V1, step_V1 + self.n_traces*dV, dV)

            sim_setup['prestep_V'] = prestep_V
            sim_setup['step_Vs'] = step_Vs
            self.dataset[count, :(1 + self.n_traces)] = np.insert(step_Vs, 0, prestep_V)

            # get the params
            for p_name in list(self.params_bounds.keys()): 
                if p_name == 'p': 
                    params[p_name] = np.random.randint(self.params_bounds[p_name][0], self.params_bounds[p_name][1] + 1) # handle the numpy take open upper bd
                else:
                    params[p_name] = np.random.uniform(self.params_bounds[p_name][0], self.params_bounds[p_name][1])

            model = HH_model_exp(params, sim_setup)
            model.simulation()

            if model.check_current_ss() and model.check_steady_state_curve(): 
                # collected_t, collected_current_traces = model.collect_points(self.n_points)
                # print(collected_t.shape, collected_current_traces.shape, np.array(list(params.values())).shape)
                data = np.concatenate((model.max_index_array, model.current_traces.flatten(), np.array(list(params.values()))))
                self.dataset[count] = data
                count += 1

        return self.dataset.shape