import sympy as sym
import numpy as np
import pandas as pd
import multiprocessing

# define symbols
t = sym.Symbol('t')

I = sym.Symbol('I')
m = sym.Symbol('m')
m_inf = sym.Symbol('m_inf')

p = sym.Symbol('p')

prestep_V = sym.Symbol('prestep_V')
step_V = sym.Symbol('step_V')

a = sym.Symbol('a')
b = sym.Symbol('b')

g_max = sym.Symbol('g_max')
E_rev = sym.Symbol('E_rev')
M_ma = sym.Symbol('M_ma')
M_mb = sym.Symbol('M_mb')
V_2ma = sym.Symbol('V_2ma')
V_2mb = sym.Symbol('V_2mb')
s_ma = sym.Symbol('s_ma')
s_mb = sym.Symbol('s_mb')


# define model - for potassium channel
m_inf = 1 / (1 + M_mb/M_ma * ((1+sym.exp((prestep_V-V_2ma)/s_ma)) / (1+sym.exp((prestep_V-V_2mb)/s_mb))))

a = M_ma / (1 + sym.exp((step_V - V_2ma) / s_ma))
b = M_mb / (1 + sym.exp((step_V - V_2mb) / s_mb))

m = (a - (a-(a+b)*m_inf)*sym.exp(-t*(a+b))) / (a + b)
I = g_max * m ** 4 * (step_V - E_rev)


# define substitutions
t_points = np.arange(0.0, 120, 0.01) 
sim_p = 4

sim_prestep_V = -100
prestep_Vs = [-80, -50, -20]

step_Vs = [0.00, 10.00, 20.00, 30.00, 40.00, 50.00, 60.00, 70.00, 80.00]
sim_step_V = 80

traces_Vs = [(sim_prestep_V, v) for v in step_Vs]
traces_Vs.extend([(v, sim_step_V) for v in prestep_Vs])

# read dataset_test
params_test = pd.read_csv('dataset_test.csv').iloc[:, -8:]
num_matrices = params_test.shape[0]
num_params = 8
sensitivity_coef_matrices = np.zeros((num_matrices, len(traces_Vs) * len(t_points), num_params))


# Function to calculate sensitivity coefficients for a subset of matrices
def calculate_sensitivity_coefficients(start_idx, end_idx, t_points, traces_Vs, params_test, sensitivity_coef_matrices, I):
    for i in range(start_idx, end_idx):
        col = 0
        params_sub = {g_max: 0, E_rev: 0, M_ma: 0, M_mb: 0, V_2ma: 0, V_2mb: 0, s_ma: 0, s_mb: 0}
        
        for p in list(params_sub.keys()):
            params_sub[p] = params_test[str(p)][i]

        for param in list(params_sub.keys()):
            row = 0
            for point in t_points:
                params_sub[t] = point
                for trace in traces_Vs:
                    params_sub[prestep_V] = trace[0]
                    params_sub[step_V] = trace[1]
                    sensitivity_coef_matrices[i, row, col] = sym.diff(I, param).subs(params_sub).evalf()
                    row += 1

            col += 1

# Number of processes (adjust as needed)
num_processes = 50

# Divide the work among processes
indices_per_process = len(params_test) // num_processes
processes = []

# Create and start processes
for i in range(num_processes):
    start_idx = i * indices_per_process
    end_idx = (i + 1) * indices_per_process if i < num_processes - 1 else len(params_test)
    
    process = multiprocessing.Process(target=calculate_sensitivity_coefficients, args=(start_idx, end_idx, t_points, traces_Vs, params_test, sensitivity_coef_matrices, I))
    processes.append(process)
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()



np.save('sensitivity_coeff_matrix_parallel.npy', sensitivity_coef_matrices)