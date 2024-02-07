# Description: Hodgkin-Huxley model with exp formalism
import numpy as np

class de_obj_model:
    '''
    The goal of this class is to build a model that takes in the target data (time traces of shape (n_traces, n_pts))
    , along with the prestep_V (1,1), step_Vs (n_traces, 1). 
    Given the a set of params (1d array), returns the simulated current traces (11, 20)

    This method is implemented in the objective fct of DE
    , and uses element matrix operations to speed up the searching in Differential Evolution. 
    '''
    def __init__(self, params, sim_setup):
        # params and sim_setup are both dict
        self.p = params[0]
        self.g_max = params[1]
        self.E_rev = params[2]
        self.a_m = params[3]
        self.b_m = params[4]
        self.delta_m = params[5]
        self.s_m = params[6]

        self.V_2m = - self.s_m * np.log(self.b_m / self.a_m)

        self.prestep_V = sim_setup['prestep_V']
        self.step_Vs = sim_setup['step_Vs']
        self.t = sim_setup['t']
    
    def m_infty(self, V): 
    # Find the steady-state curve at a fixed V, x goes to steady_state as t increses
    # V: membrane voltage
        # V_2m is only defined for steady state and time constant
        return (1 / (1 + np.exp((V - self.V_2m) / self.s_m))).reshape((-1,1))
    
    
    def tau_m(self, V): 
        '''
        Find the time constant curve at a fixed V, it governs the rate at which x approaches to the steady state at a fixed V.
        V: membrane voltage
        '''
        tau_0m = (1 / self.a_m) * np.exp((self.delta_m * self.V_2m) / self.s_m)
        return ((tau_0m * np.exp(self.delta_m * ((V - self.V_2m) / self.s_m))) / (1 + np.exp((V - self.V_2m) / self.s_m))).reshape((-1,1))

    # Construct the model
    def alpha(self, V): 
        '''
        Find the rates of opening and closing of the activation(m)/inactivation(h) gates using hyperbolic tangent functions
        V is the applied memebrane voltage
        '''
        return (self.a_m * np.exp(- self.delta_m * V / self.s_m)).reshape((-1,1))
    
    def beta(self, V): 
        '''
        find the closing rate of a channel 
        '''
        return (self.b_m * np.exp((1 - self.delta_m) * V / self.s_m)).reshape((-1,1))

    def m(self, V): 
        '''
        Find the activation/inactivation variable analytically with I.C. x(0) = initial_cond
        This only stands with fixed prestep potential simulation setup!
        '''
        return self.m_infty(V) + (self.m_infty(self.prestep_V) - self.m_infty(V)) * np.exp(- self.t / self.tau_m(V))

    def find_I(self, V):
        return np.full(self.m(self.step_Vs).shape, self.g_max) * np.power(self.m(V), np.full(self.m(self.step_Vs).shape, self.p)) * (V - np.full(self.m(self.step_Vs).shape, self.E_rev)) 
    
    def simulation(self): 
        '''
        simulation only contains fixed prestep potential and a number of step potentials, covered in the sim_setup dict
        generate current traces according to the setup
        '''

        # self.current_traces = np.array([self.find_I(v) for v in self.step_Vs])
        self.current_traces = self.find_I(self.step_Vs)

        return self.current_traces
    