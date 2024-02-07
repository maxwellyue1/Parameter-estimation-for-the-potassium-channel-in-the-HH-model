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
        self.n_points = n_points

    
    # def collect_points(self, model): 
    #     '''
    #     this takes point for the current_traces of 1 sample 
    #     pass in model of class HH_model_exp
    #     '''
    #     # clean_data_evenly_vertically_spaced
    #     n_traces = model.current_traces.shape[0]
    #     t = self.t # simulation time
    #     collected_t = np.empty((n_traces, self.n_points))  # (n_traces, n_points)
    #     collected_current_traces = np.empty((n_traces, self.n_points))

    #     for i in range(n_traces):    # over the number of traces
    #         index_array = []  # the index array we are taking for the time points and values for current traces
    #         # it is specific for each sample and current trace
    #         min_val_inatrace = model.current_traces[i, 0]  # min value for a specific trace
    #         max_val_inatrace = model.current_traces[i, model.max_index_array[i]]  # max value for a specific trace
    #         target_values = np.linspace(min_val_inatrace, max_val_inatrace, self.n_points)  # target values array

    #         collected_t[i, 0] = t[0]
    #         collected_t[i, -1] = t[model.max_index_array[i]]
    #         collected_current_traces[i] = target_values


    #         arr = model.current_traces[i, 0 : (model.max_index_array[i]+2)]  # the specific cropped current trace searching in
    #         # we added 2, 1 for the python indexing, 1 to handle exception when finding index
    #         for pt in range(1, self.n_points - 1):  # iterate over the number of points 
    #             num = target_values[pt]  # the target current value at a specific point
    #             index = np.argmin((num - arr) >= 0)-1  # the index in arr with the current value on the left of the num
    #             # apply linear approximation to get selected time point
    #             collected_t[i, pt] = t[index] + (t[index+1] - t[index]) * (num - model.current_traces[i, index]) / (model.current_traces[i, index+1] - model.current_traces[i, index])
    #     return collected_t, collected_current_traces
    

    def generate_data(self, n_sample):
        count = 0
        self.dataset = np.empty((n_sample, 2 + self.n_points * self.n_traces * 2 + len(self.params_bounds)))

        while count < n_sample: 
            sim_setup = {'t':self.t}
            params = {}

            # get the sim_setup
            # n_traces of step_Vs are selected from V1(chosen by lower bounds defined), and increase by 10 mV
            prestep_V = np.random.randint(self.prestep_V_bounds[0]/10, self.prestep_V_bounds[1]/10 + 1) * 10
            step_V1 = np.random.randint(self.step_Vs_lb[0]/10, self.step_Vs_lb[1]/10 + 1) * 10
            step_Vs = np.arange(step_V1, step_V1 + self.n_traces*10, 10)
            sim_setup['prestep_V'] = prestep_V
            sim_setup['step_Vs'] = step_Vs
            self.dataset[count, :2] = [prestep_V, step_V1]

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
                self.dataset[count, 2:] = data

                count += 1

        return self.dataset.shape

   
    def create_empty_csv(self, file_name = 'dataset_exp.csv'):
        names = []
        names.append('prestep_V')
        names.append('step_V1')
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

    
