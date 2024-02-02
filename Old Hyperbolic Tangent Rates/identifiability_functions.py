import numpy as np

def find_illed_samples(current_traces_3d, params, threshold = 1e-5): 
    '''
    t is the simulation time
    Here, we take current traces of (# of samples, # of traces, # of simulation points)
    and params of (# of samples, 8), we want to delete sample with trace that has undefined 
    threshold within the simulation time, namely illed_sample.
    '''

    n_traces = current_traces_3d.shape[1]

    # initialize 
    max_index_array = np.full((current_traces_3d.shape[0], n_traces), -1)
            
        # list containing the illed sample(current traves that barely varies)
    illed_sample = []

    # Ignore the divide by zero warning
    np.seterr(divide='ignore')

    for n in range(current_traces_3d.shape[0]): 
        for i in range(n_traces):

            #diff_array = np.abs(current_traces_3d[n, i, :][1:] - current_traces_3d[n, i, :][:-1])

            # normalized difference array
            diff_array = np.abs((current_traces_3d[n, i, :][1:] - current_traces_3d[n, i, :][:-1]) / (current_traces_3d[n, i, :][:-1] - current_traces_3d[n, i, :][0])) 
            # dividing by ~0 sometimes

            # Checking for ill-constructed samples
            if np.where(diff_array < threshold)[0].size == 0: ######## > -> <
                print('A current trace at {}th sample varies below the threshold at all time points!'.format(n))
                illed_sample.append(n)
                break
            else:
                # return last false value, despite if theres true value ahead of it, i.e. last valid index - 1
                max_index_array[n, i] = np.where(diff_array > threshold)[0][-1] + 1
                ##max_index_array[n, i] = np.argmax(diff_array <= threshold) 

    print(f'There are {len(illed_sample)} samples with undefined threshold.')

    return illed_sample, max_index_array
    

def find_small_current_samples(current_traces_3d, params, max_current, scale = 1/100):
    '''
    a small current sample is defined as a sample with max current values from all traces
    below min_current
    '''
    # this is the minimum current for all traces in a sample
    min_current = scale * max_current
    small_samples = []
    for n in range(current_traces_3d.shape[0]): 
        small = True
        i = 0
        while small and i<current_traces_3d.shape[1]: 
            if np.max(current_traces_3d[n,i]) > min_current: 
                small = False
            i += 1
        if small: 
            print(f'All trace from {n}th sample are below the scaled current!')
            small_samples.append(n)

    print(f'There are {len(small_samples)} small samples.')
    return small_samples


def delete_illed_small_samples(t, current_traces_3d, params, max_index_array, illed_samples, small_samples): 
    n_traces = current_traces_3d.shape[1]

    # initialize matrices for selected_t and selected_current_traces_3d,
    # with -1.0, but we will only fill the entries before the threshold
    selected_t = np.full((current_traces_3d.shape[0], n_traces, len(t)), -1.0)  # 3d array, (n_sample, n_stepVs, n_points)
    selected_current_traces_3d = np.full((current_traces_3d.shape[0], n_traces, len(t)), -1.0)

    for n in range(current_traces_3d.shape[0]): 
        for i in range(n_traces): 
            #print(dataset_generator.t[: max_index_array[n, i]])
            selected_t[n, i, :max_index_array[n, i]+1] = t[: max_index_array[n, i]+1]
            selected_current_traces_3d[n, i, :max_index_array[n, i]+1] = current_traces_3d[n, i, :max_index_array[n, i]+1]

    samples_delete = list(set(illed_samples) | set(small_samples))
    # Delete  illed samples and small samples in the selected t, current traces, and params.T
    selected_t = np.delete(selected_t, samples_delete, axis=0)
    selected_current_traces_3d = np.delete(selected_current_traces_3d, samples_delete, axis=0)
    selected_params = np.delete(params, samples_delete, axis=0)
    selected_max_index_array = np.delete(max_index_array, samples_delete, axis=0)

    return samples_delete, selected_t, selected_current_traces_3d, selected_params, selected_max_index_array


