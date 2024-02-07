# Description: Hodgkin-Huxley model with exp formalism
import numpy as np

class HH_model_exp:
    '''
    In this model, we only consider the Potassium channel with activation (m) gates. 
    The opening and closing rates are modelled using an exponential form described in HH_model.pdf
    This class can generate current traces based on params and simulation setups.
    It can also check various conditions: threshold on current traces, m_infty conditions. 
    '''
    def __init__(self, params, sim_setup):
        # params and sim_setup are both dict
        self.p = params['p']
        self.g_max = params['g_max']
        self.E_rev = params['E_rev']
        self.a_m = params['a_m']
        self.b_m = params['b_m']
        self.delta_m = params['delta_m']
        self.s_m = params['s_m']

        self.V_2m = - self.s_m * np.log(self.b_m / self.a_m)

        self.prestep_V = sim_setup['prestep_V']
        self.step_Vs = sim_setup['step_Vs']
        self.t = sim_setup['t']
    
    def m_infty(self, V): 
    # Find the steady-state curve at a fixed V, x goes to steady_state as t increses
    # V: membrane voltage
        # V_2m is only defined for steady state and time constant
        return 1 / (1 + np.exp((V - self.V_2m) / self.s_m))
    
    
    def tau_m(self, V): 
        '''
        Find the time constant curve at a fixed V, it governs the rate at which x approaches to the steady state at a fixed V.
        V: membrane voltage
        '''
        tau_0m = (1 / self.a_m) * np.exp((self.delta_m * self.V_2m) / self.s_m)
        return (tau_0m * np.exp(self.delta_m * ((V - self.V_2m) / self.s_m))) / (1 + np.exp((V - self.V_2m) / self.s_m))

    # Construct the model
    def alpha(self, V): 
        '''
        Find the rates of opening and closing of the activation(m)/inactivation(h) gates using hyperbolic tangent functions
        V is the applied memebrane voltage
        '''
        return self.a_m * np.exp(- self.delta_m * V / self.s_m)
    
    def beta(self, V): 
        '''
        find the closing rate of a channel 
        '''
        return self.b_m * np.exp((1 - self.delta_m) * V / self.s_m)

    def m(self, V): 
        '''
        Find the activation/inactivation variable analytically with I.C. x(0) = initial_cond
        This only stands with fixed prestep potential simulation setup!
        '''
        return self.m_infty(V) + (self.m_infty(self.prestep_V) - self.m_infty(V)) * np.exp(- self.t / self.tau_m(V))

    def find_I(self, V):
        return self.g_max * np.power(self.m(V), self.p) * (V - self.E_rev) 
    
    def simulation(self): 
        '''
        simulation only contains fixed prestep potential and a number of step potentials, covered in the sim_setup dict
        generate current traces according to the setup
        '''

        self.current_traces = np.array([self.find_I(v) for v in self.step_Vs])

        return self.current_traces
    
    def check_current_ss(self, threshold = 1e-5, thres_min_ind = 20): 
        '''
        check for current steady state threshold condition: if any one of the traces varies over the threshold through all simulation time, then fails
        record the threshold index for data pts collection 
        '''
        check = True
        self.max_index_array = np.full((len(self.step_Vs)), -1)
        # iterating over the number of traces
        for i in range(self.current_traces.shape[0]): 
            diff_arr = np.abs((self.current_traces[i, :][1:] - self.current_traces[i, :][:-1]) / (self.current_traces[i, :][:-1] - self.current_traces[i, :][0]))
            if np.where(diff_arr < threshold)[0].size == 0:
                check = False
                # print(f'{self.p, self.g_max, self.E_rev, self.a_m, self.b_m, self.delta_m, self.s_m} generates currents with undefined threshold!')
                break
            else: 
                self.max_index_array[i] = np.where(diff_arr > threshold)[0][-1] + 1

        if np.any(self.max_index_array < thres_min_ind): 
            check = False
        return check
    
    def m_infty_dev(self, V): 
        return - np.exp((V - self.V_2m) / self.s_m) / (self.s_m * (1 + np.exp((V - self.V_2m) / self.s_m)) ** 2)

    def check_steady_state_curve(self, ends_threshold=0.05, mid_pt_sensitivity_ub=1/40):
        '''
        We check the if there are (num_pts) observable pts on the steady state curve. 
        the observable pts on the cuve are from the current traces steady states
        ends_threshold: the upper bound on the m_inf(V_1), and the lower bound on m_inf(V_n)
        mid_pt_sensitivity_ub: the upper bound on dm_inf(V_2m)/dt
        Need to check if V_2m in [self.step_Vs[0], self.step_Vs[-1]]
        '''
        V_1 = self.step_Vs[0]
        V_n = self.step_Vs[-1]

        if (self.m_infty(V_1) > ends_threshold) or (self.m_infty(V_n) < (1-ends_threshold)) or (self.m_infty_dev(self.V_2m) > mid_pt_sensitivity_ub) or (self.V_2m < self.step_Vs[0]) or (self.V_2m > self.step_Vs[-1]):
            check = False
        else: 
            check = True

        return check

    
    def collect_points(self, n_points): 
        '''
        this takes point for the current_traces of 1 sample 
        pass in model of class HH_model_exp
        '''
        # clean_data_evenly_vertically_spaced
        n_traces = self.current_traces.shape[0]
        t = self.t # simulation time
        collected_t = np.empty((n_traces, n_points))  # (n_traces, n_points)
        collected_current_traces = np.empty((n_traces, n_points))

        for i in range(n_traces):    # over the number of traces
            index_array = []  # the index array we are taking for the time points and values for current traces
            # it is specific for each sample and current trace
            min_val_inatrace = self.current_traces[i, 0]  # min value for a specific trace
            max_val_inatrace = self.current_traces[i, self.max_index_array[i]]  # max value for a specific trace
            target_values = np.linspace(min_val_inatrace, max_val_inatrace, n_points)  # target values array

            collected_t[i, 0] = t[0]
            collected_t[i, -1] = t[self.max_index_array[i]]
            collected_current_traces[i] = target_values


            arr = self.current_traces[i, 0 : (self.max_index_array[i]+2)]  # the specific cropped current trace searching in
            # we added 2, 1 for the python indexing, 1 to handle exception when finding index
            for pt in range(1, n_points - 1):  # iterate over the number of points 
                num = target_values[pt]  # the target current value at a specific point
                index = np.argmin((num - arr) >= 0)-1  # the index in arr with the current value on the left of the num
                # apply linear approximation to get selected time point
                collected_t[i, pt] = t[index] + (t[index+1] - t[index]) * (num - self.current_traces[i, index]) / (self.current_traces[i, index+1] - self.current_traces[i, index])
        return collected_t, collected_current_traces