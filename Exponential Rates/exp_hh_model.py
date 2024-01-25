# Description: Hodgkin-Huxley model with exp formalism
import numpy as np

class HH_model_exp:
    '''
    In this model, we only consider the Potassium channel with activation (m) gates. 
    The opening and closing rates are modelled using an exponential form described in HH_model.pdf
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

        self.prestep_V = sim_setup['prestep_V']
        self.step_Vs = sim_setup['step_Vs']
        self.t = sim_setup['t']
    
    def m_infty(self, V): 
    # Find the steady-state curve at a fixed V, x goes to steady_state as t increses
    # V: membrane voltage
        V_2m = - self.s_m * np.log(self.b_m / self.a_m)
        # V_2m is only defined for steady state and time constant
        return 1 / (1 + np.exp((V - V_2m) / self.s_m))
    
    
    def tau_m(self, V): 
        '''
        Find the time constant curve at a fixed V, it governs the rate at which x approaches to the steady state at a fixed V.
        V: membrane voltage
        '''
        V_2m = - self.s_m * np.log(self.b_m / self.a_m)
        tau_0m = (1 / self.a_m) * np.exp((self.delta_m * V_2m) / self.s_m)
        return (tau_0m * np.exp(self.delta_m * ((V - V_2m) / self.s_m))) / (1 + np.exp((V - V_2m) / self.s_m))

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

        current_traces = np.array([self.find_I(v) for v in self.step_Vs])

        return current_traces