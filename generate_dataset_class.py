import numpy as np
from hh_model import HH_model
import csv

class potassium_channel_dataset_genaerator:
    # we could set the random_seed if we want to regenerate the data
    # No random_seed when collecting data into file
    def __init__(self, inputs):#, random_seed = 42):
        #self.random_seed = random_seed

        self.p = inputs['p']
        self.q = inputs['q']
        self.prestep_V = inputs['prestep_V']
        self.step_Vs = inputs['step_Vs']

        self.prestep_Vs = inputs['prestep_Vs']
        self.step_V = inputs['step_V']

        self.t = np.arange(0.0, inputs['end_time'], inputs['time_step']) # m, the length of time in when step voltages are applied
        
        self.X_h = inputs['X_h']
        self.param_bounds_wo_h = inputs['param_bounds_wo_h']

        self.maximum_current = self.param_bounds_wo_h['g_max'][1] * np.max(np.append(self.step_Vs, self.step_V))


    def generate_data(self, n_sample):
        #np.random.seed(self.random_seed)

        bounds = list(self.param_bounds_wo_h.values())
        self.params = np.empty((n_sample, len(bounds)))
        for i in range(len(bounds)):
            self.params[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], n_sample)

        self.current_traces_3d = np.empty((n_sample, len(self.step_Vs)+len(self.prestep_Vs), len(self.t)))
        for n in range(n_sample):

            model = HH_model(self.p, self.q, self.params[n, 0], self.params[n, 1], self.params[n, 2:], self.X_h, self.prestep_V, self.step_Vs, self.prestep_Vs, self.step_V, self.t)
            # step_ms = np.array([model.find_x_fixedV(model.openclose_rates_fixedV(v, model.X_m)[0], model.openclose_rates_fixedV(v, model.X_m)[1], model.find_steady_state(model.prestep_V, model.X_m)) for v in model.step_Vs])
            # step_hs = np.array([model.find_x_fixedV(model.openclose_rates_fixedV(v, model.X_h)[0], model.openclose_rates_fixedV(v, model.X_h)[1], model.find_steady_state(model.prestep_V, model.X_h)) for v in model.step_Vs])
            # step_Is = np.array([model.find_I(model.step_Vs[i], step_ms[i], step_hs[i]) for i in range(len(model.step_Vs))])

            # prestep_ms = np.array([model.find_x_fixedV(model.openclose_rates_fixedV(model.step_V, model.X_m)[0], model.openclose_rates_fixedV(model.step_V, model.X_m)[1], model.find_steady_state(pv, model.X_m)) for pv in model.prestep_Vs])
            # prestep_hs = np.array([model.find_x_fixedV(model.openclose_rates_fixedV(model.step_V, model.X_h)[0], model.openclose_rates_fixedV(model.step_V, model.X_h)[1], model.find_steady_state(pv, model.X_h)) for pv in model.prestep_Vs])
            # prestep_Is = np.array([model.find_I(model.step_V, prestep_ms[j], prestep_hs[j]) for j in range(len(model.prestep_Vs))])

            # current_traces_3d[n, len(model.step_Vs):] = prestep_Is
            # current_traces_3d[n, 0:len(model.step_Vs)] = step_Is
            self.current_traces_3d[n] = model.generate_current_trace()

        # dataset_flattened_current_traces = current_traces_3d.reshape(current_traces_3d.shape[0], -1)
        # dataset = np.hstack((dataset_flattened_current_traces, params))

        # # Define the CSV file path to store the dataset
        # csv_file_path = "raw_dataset.csv"

        # # Open the CSV file in write mode
        # with open(csv_file_path, mode="a", newline="") as file:
        #     csv_writer = csv.writer(file)
        #     # Write each row of the array to the CSV file
        #     for row in dataset:
        #         csv_writer.writerow(row)

        # # Open the CSV file and count the number of rows
        # num_rows = 0
        # with open(csv_file_path, mode="r") as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         num_rows += 1

        # return self.current_traces_3d, self.params
    

    def find_illed_samples(self, threshold = 1e-5): 
        self.threshold = threshold
        n_traces = self.current_traces_3d.shape[1]

        # initialize 
        self.max_index_array = np.full((self.current_traces_3d.shape[0], n_traces), -1)
                
            # list containing the illed sample(current traves that barely varies)
        self.illed_sample = []

        # Ignore the divide by zero warning
        np.seterr(divide='ignore')

        for n in range(self.current_traces_3d.shape[0]): 
            for i in range(n_traces):

                #diff_array = np.abs(current_traces_3d[n, i, :][1:] - current_traces_3d[n, i, :][:-1])

                # normalized difference array
                diff_array = np.abs((self.current_traces_3d[n, i, :][1:] - self.current_traces_3d[n, i, :][:-1]) / (self.current_traces_3d[n, i, :][:-1] - self.current_traces_3d[n, i, :][0])) 
                # dividing by ~0 sometimes

                # Checking for ill-constructed samples
                if np.where(diff_array < threshold)[0].size == 0: ######## > -> <
                    print('A current trace at {}th sample varies below the threshold at all time points!'.format(n))
                    self.illed_sample.append(n)
                    break
                else:
                    # return last false value, despite if theres true value ahead of it, i.e. last valid index - 1
                    self.max_index_array[n, i] = np.where(diff_array > threshold)[0][-1] + 1
                    ##max_index_array[n, i] = np.argmax(diff_array <= threshold) 

        print(f'There are {len(self.illed_sample)} samples with undefined threshold.')

        # return self.illed_sample, self.max_index_array
    

    def find_small_current_samples(self, scale = 1/100):

        # this is the minimum current for all traces in a sample
        min_current = scale * self.maximum_current
        self.scale = scale
        self.small_samples = []
        for n in range(self.current_traces_3d.shape[0]): 
            small = True
            i = 0
            while small and i<self.current_traces_3d.shape[1]: 
                if np.max(self.current_traces_3d[n,i]) > min_current: 
                    small = False
                i += 1
            if small: 
                print(f'All trace from {n}th sample are below the scaled current!')
                self.small_samples.append(n)

        print(f'There are {len(self.small_samples)} small samples.')
        # return self.small_samples
    
    def delete_illed_small_samples(self): 
        n_traces = self.current_traces_3d.shape[1]

        # initialize matrices for selected_t and selected_current_traces_3d,
        # with -1.0, but we will only fill the entries before the threshold
        self.selected_t = np.full((self.current_traces_3d.shape[0], n_traces, len(self.t)), -1.0)  # 3d array, (n_sample, n_stepVs, n_points)
        self.selected_current_traces_3d = np.full((self.current_traces_3d.shape[0], n_traces, len(self.t)), -1.0)

        for n in range(self.current_traces_3d.shape[0]): 
            for i in range(n_traces): 
                #print(dataset_generator.t[: max_index_array[n, i]])
                self.selected_t[n, i, :self.max_index_array[n, i]+1] = self.t[: self.max_index_array[n, i]+1]
                self.selected_current_traces_3d[n, i, :self.max_index_array[n, i]+1] = self.current_traces_3d[n, i, :self.max_index_array[n, i]+1]

        self.samples_delete = list(set(self.illed_sample) | set(self.small_samples))
        # Delete  illed samples and small samples in the selected t, current traces, and params.T
        self.selected_t = np.delete(self.selected_t, self.samples_delete, axis=0)
        self.selected_current_traces_3d = np.delete(self.selected_current_traces_3d, self.samples_delete, axis=0)
        self.selected_params = np.delete(self.params, self.samples_delete, axis=0)
        self.selected_max_index_array = np.delete(self.max_index_array, self.samples_delete, axis=0)

        # return self.samples_delete, self.selected_t, self.selected_current_traces_3d, self.selected_params, self.selected_max_index_array


    def collect_points(self, n_points = 20): 
        # clean_data_evenly_vertically_spaced
        self.n_points = n_points
        self.collected_t = np.empty((self.selected_current_traces_3d.shape[0], self.selected_current_traces_3d.shape[1], n_points))  # 3d array, (n_sample, n_stepVs, n_points)
        self.collected_current_traces_3d = np.empty((self.selected_current_traces_3d.shape[0], self.selected_current_traces_3d.shape[1], n_points))

        for n in range(self.selected_current_traces_3d.shape[0]): 
            for i in range(self.selected_current_traces_3d.shape[1]):
                index_array = []  # the index array we are taking for the time points and values for current traces
                # it is specific for each sample and current trace
                min_val_inatrace = self.selected_current_traces_3d[n, i, 0]  # min value for a specific trace
                max_val_inatrace = self.selected_current_traces_3d[n, i, self.selected_max_index_array[n, i]]  # max value for a specific trace
                target_values = np.linspace(min_val_inatrace, max_val_inatrace, n_points)  # target values array

                self.collected_t[n, i, 0] = self.selected_t[n, i, 0]
                self.collected_t[n, i, -1] = self.selected_t[n, i, self.selected_max_index_array[n, i]]
                self.collected_current_traces_3d[n, i] = target_values


                arr = self.selected_current_traces_3d[n, i, 0:self.selected_max_index_array[n, i]+2]  # the specific cropped current trace searching in
                # we added 2, 1 for the python indexing, 1 to handle exception when finding index
                for pt in range(1, n_points-1):  # iterate over the number of points 
                    num = target_values[pt]  # the target current value at a specific point
                    index = np.argmin((num - arr) >= 0)-1  # the index in arr with the current value on the left of the num

                    # apply linear approximation to get selected time point
                    self.collected_t[n, i, pt] = self.selected_t[n, i, index] + (self.selected_t[n, i, index+1] - self.selected_t[n, i, index]) * (num - self.selected_current_traces_3d[n, i, index]) / (self.selected_current_traces_3d[n, i, index+1] - self.selected_current_traces_3d[n, i, index])



    def clean_data_evenly_vertically_spaced(self, raw_current_traces, params, n_points=20, threshold=1e-5):
        self.n_points = n_points
        self.threshold = threshold

        n_traces = raw_current_traces.shape[1]

        max_index_array = np.full((raw_current_traces.shape[0], n_traces), -1)
        
            # list containing the illed sample(current traves that barely varies)
        illed_sample = []

        # Ignore the divide by zero warning
        np.seterr(divide='ignore')

        for n in range(raw_current_traces.shape[0]): 
            for i in range(n_traces):

                #diff_array = np.abs(current_traces_3d[n, i, :][1:] - current_traces_3d[n, i, :][:-1])

                # normalized difference array
                diff_array = np.abs((raw_current_traces[n, i, :][1:] - raw_current_traces[n, i, :][:-1]) / (raw_current_traces[n, i, :][:-1] - raw_current_traces[n, i, :][0])) 
                # dividing by ~0 sometimes

                # Checking for ill-constructed samples
                if np.where(diff_array < threshold)[0].size == 0:
                    print('A current trace at {}th sample varies below the threshold at all time points! Delete this sample!'.format(n))
                    illed_sample.append(n)
                    break
                else:
                    # return last false value, despite if theres true value ahead of it, i.e. last valid index - 1
                    max_index_array[n, i] = np.where(diff_array > threshold)[0][-1] + 1
                    ##max_index_array[n, i] = np.argmax(diff_array <= threshold) 

            #print(max_index_array)

        selected_t = np.empty((raw_current_traces.shape[0], n_traces, n_points))  # 3d array, (n_sample, n_stepVs, n_points)
        selected_current_traces_3d = np.empty((raw_current_traces.shape[0], n_traces, n_points))
        for n in range(raw_current_traces.shape[0]): 
            for i in range(n_traces):
                index_array = []  # the index array we are taking for the time points and values for current traces
                # it is specific for each sample and current trace
                min_val_inatrace = raw_current_traces[n, i, 0]  # min value for a specific trace
                max_val_inatrace = raw_current_traces[n, i, max_index_array[n, i]]  # max value for a specific trace
                target_values = np.linspace(min_val_inatrace, max_val_inatrace, n_points)  # target values array

                selected_t[n, i, 0] = self.t[0]
                selected_t[n, i, -1] = self.t[max_index_array[n, i]]
                selected_current_traces_3d[n, i] = target_values


                arr = raw_current_traces[n, i, 0:max_index_array[n, i]+2]  # the specific cropped current trace searching in
                # we added 2, 1 for the python indexing, 1 to handle exception when finding index
                for pt in range(1, n_points-1):  # iterate over the number of points 
                    num = target_values[pt]  # the target current value at a specific point
                    index = np.argmin((num - arr) >= 0)-1  # the index in arr with the current value on the left of the num

                    # apply linear approximation to get selected time point
                    selected_t[n, i, pt] = self.t[index] + (self.t[index+1] - self.t[index]) * (num - raw_current_traces[n, i, index]) / (raw_current_traces[n, i, index+1] - raw_current_traces[n, i, index])

        # delete the illed sample stored in selected_t and selected_current_traces_3d
        selected_t = np.delete(selected_t, illed_sample, axis=0)
        selected_current_traces_3d = np.delete(selected_current_traces_3d, illed_sample, axis=0)
        selected_params = np.delete(params, illed_sample, axis=0)
        
        # cleaned_dataset = (selected_t, selected_current_traces_3d, parameters)
        # with open(f'dataset_{n_sample}_{n_points}_{threshold}.pickle', 'wb') as handle:
        #     pickle.dump(cleaned_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return selected_t, selected_current_traces_3d, selected_params
    
   
   
    def create_empty_csv(self, file_path = 'dataset.csv'):
        names = []
        for i in range(self.step_Vs.shape[0]+self.prestep_Vs.shape[0]):
            for point in range(self.n_points): 
                t_name = 'time point {} for {}th trace'.format(int(point), int(i))
                names.append(t_name)
        for i in range(self.step_Vs.shape[0]+self.prestep_Vs.shape[0]): 
            for pt in range(self.n_points): 
                names.append('Current trace {} at point {}'.format(int(i), int(pt)))

        names += list(self.param_bounds_wo_h.keys())

        # Open the CSV file in write mode to create an empty file
        # with open(csv_file_path, mode="w", newline="") as file:
        #     pass  # This just creates an empty file

        with open(file_path, mode="w", newline="") as file:
            csv_writer = csv.writer(file)
            # Write each row of the array to the CSV file
            csv_writer.writerow(names)
        
    
    def save_tubular_data(self, file_path = 'dataset.csv'):
        flattened_time_traces = self.collected_t.reshape(self.collected_t.shape[0], -1)
        flattened_current_traces = self.collected_current_traces_3d.reshape(self.collected_current_traces_3d.shape[0], -1)
        dataset = np.hstack((flattened_time_traces, flattened_current_traces, self.selected_params))

        # # Define the CSV file path to store the dataset
        # csv_file_path = "dataset.csv"

        # Open the CSV file in write mode
        with open(file_path, mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            # Write each row of the array to the CSV file
            for row in dataset:
                csv_writer.writerow(row)

    
