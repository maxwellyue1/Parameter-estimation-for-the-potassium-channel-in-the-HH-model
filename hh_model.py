# Description: Hodgkin-Huxley model
import numpy as np

class HH_model:
    def __init__(self, p, q, g_max, E_rev, X_m, X_h, prestep_V, step_Vs, prestep_Vs, step_V, t):
        self.p = p
        self.q = q
        self.g_max = g_max
        self.E_rev = E_rev
        self.X_m = X_m
        self.X_h = X_h
        self.prestep_V = prestep_V
        self.step_Vs = step_Vs
        self.prestep_Vs = prestep_Vs
        self.step_V = step_V
        self.t = t
    
    def find_steady_state(self, V, X): 
    # Find the steady-state curve at a fixed V, x goes to steady_state as t increses
    # V: membrane voltage
    # X: the activation or inactivation parameter set
        return 1 / (1 + X[1]/X[0] * (1+np.exp((V-X[2])/X[4]))/(1+np.exp((V-X[3])/X[5])))
    
    
    def find_time_constant(self, V, X): 
        '''
        Find the time constant curve at a fixed V, it governs the rate at which x approaches to the steady state at a fixed V.
        V: membrane voltage
        X: the activation or inactivation parameter set
        '''
        return 1 / (X[0]/(1+np.exp((V-X[2])/X[4])) + X[1]/(1+np.exp((V-X[3])/X[5])))

    # Construct the model
    def openclose_rates_fixedV(self, V, X): 
        '''
        Find the rates of opening and closing of the activation(m)/inactivation(h) gates using hyperbolic tangent functions
        V is the applied memebrane voltage
        X is the parameter set associated with activation/inactivation term 
        '''
        a = (X[0]) / (1 + np.exp((V - X[2]) / X[4]))
        b = (X[1]) / (1 + np.exp((V - X[3]) / X[5]))
        return a, b

    def find_x_fixedV(self, a, b, init_cond): 
        '''
        Find the activation/inactivation variable analytically with I.C. x(0) = initial_cond
        a, b: the opening and closing rates of of the activation/inactivation gates
        t_steps
        '''
        return (a - (a - (a + b) * init_cond) * np.exp(-self.t * (a + b))) / (a + b)

    def find_I(self, V, m, h):
        return self.g_max * np.power(m, self.p) * np.power(h, self.q) * (V - self.E_rev) 
    
    def generate_current_trace(self): 
        current_trace = np.empty((len(self.step_Vs)+len(self.prestep_Vs), len(self.t)))

        step_ms = np.array([self.find_x_fixedV(self.openclose_rates_fixedV(v, self.X_m)[0], self.openclose_rates_fixedV(v, self.X_m)[1], self.find_steady_state(self.prestep_V, self.X_m)) for v in self.step_Vs])
        step_hs = np.array([self.find_x_fixedV(self.openclose_rates_fixedV(v, self.X_h)[0], self.openclose_rates_fixedV(v, self.X_h)[1], self.find_steady_state(self.prestep_V, self.X_h)) for v in self.step_Vs])
        step_Is = np.array([self.find_I(self.step_Vs[i], step_ms[i], step_hs[i]) for i in range(len(self.step_Vs))])

        prestep_ms = np.array([self.find_x_fixedV(self.openclose_rates_fixedV(self.step_V, self.X_m)[0], self.openclose_rates_fixedV(self.step_V, self.X_m)[1], self.find_steady_state(pv, self.X_m)) for pv in self.prestep_Vs])
        prestep_hs = np.array([self.find_x_fixedV(self.openclose_rates_fixedV(self.step_V, self.X_h)[0], self.openclose_rates_fixedV(self.step_V, self.X_h)[1], self.find_steady_state(pv, self.X_h)) for pv in self.prestep_Vs])
        prestep_Is = np.array([self.find_I(self.step_V, prestep_ms[j], prestep_hs[j]) for j in range(len(self.prestep_Vs))])
            
        current_trace[len(self.step_Vs):, :] = prestep_Is
        current_trace[0:len(self.step_Vs), :] = step_Is

        return current_trace