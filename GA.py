from generate_dataset_class import potassium_channel_dataset_genaerator
from hh_model import HH_model
import numpy as np
import random, numpy as np
import matplotlib.pyplot as plt
import uuid
import csv
import os
from deap import base
from deap import creator
from deap import tools

def main():
    # initialize a dictionary of training history to store in a csv file
    history_dict = {}

    unique_id = str(uuid.uuid4())[:8]
    history_dict['unique_id'] = unique_id
    print(f'Experiment ID: {unique_id}')

    M_ha = 2
    M_hb = 5
    V_2ha = -40
    V_2hb = -35
    s_ha = 6
    s_hb = -10
    X_h = [M_ha, M_hb, V_2ha, V_2hb, s_ha, s_hb]

    param_bounds_wo_h_dict = {'g_max': [100.0, 140.0],
                    'E_rev': [-100, -80],

                    'M_ma': [0, 50],
                    'M_mb': [0, 50],
                    'V_2ma': [-60, 60],
                    #'V_2ma': [10, 100],
                    'V_2mb': [-60, 60],
                    's_ma': [-50, -0.5],  # excluding small abs values to avoid operating error in exp
                    's_mb': [0.5, 50]}

    input = {'p': 4,
            'q': 0,
            'step_Vs': np.array([0.00, 10.00, 20.00, 30.00, 40.00, 50.00, 60.00, 70.00, 80.00]), # mV
            'prestep_Vs': np.array([-80, -50, -20]),
            'step_V': 80,
            'prestep_V': -100,  #mV
            'end_time': 6,
            'time_step': 0.01,
            'X_h': X_h,
            'param_bounds_wo_h': param_bounds_wo_h_dict}

    dataset_generator = potassium_channel_dataset_genaerator(input)

    dataset_generator.generate_data(5)
    dataset_generator.find_illed_samples()
    dataset_generator.find_small_current_samples()
    dataset_generator.delete_illed_small_samples()

    selected_t = dataset_generator.selected_t
    selected_current_traces_3d = dataset_generator.selected_current_traces_3d
    selected_params = dataset_generator.selected_params
    selected_max_index_array = dataset_generator.selected_max_index_array

    dataset_generator.collect_points()

    t_traces_set = dataset_generator.collected_t
    current_traces_set = dataset_generator.collected_current_traces_3d
    params_set = dataset_generator.params

    target_t_traces = t_traces_set[-1]
    target_current_traces = current_traces_set[-1]
    target_params = params_set[-1]


    gens_collect = [1, 10, 50, 100, 150, 200, 250, 300] # the generations to collect parameters

    gens_best_sol = [] # the best idividuals at each generation in gens_collect

    no_of_generations = 300 # decide iterations

    # decide, population size or no of individuals or solutions being considered in each generation
    population_size = 1000#300

    # chromosome (also called individual) in DEAP
    # length of the individual or chrosome should be divisible by no. of variables
    # is a series of 0s and 1s in Genetic Algorithm

    # here, one individual may be
    # [1,0,1,1,1,0,......,0,1,1,0,0,0,0] of length 400, 50 for each variable
    # each element is called a gene or allele or simply a bit
    # Individual in bit form is called a Genotype and is decoded to attain the Phenotype i.e. the
    size_of_individual = 400

    # above, higher the better but uses higher resources

    # we are using bit flip as mutation,
    # probability that a gene or allele will mutate or flip,
    # generally kept low, high means more random jumps or deviation from parents, which is generally not desired
    probability_of_mutation = 0.1#0.05

    # no. of participants in Tournament selection
    # to implement strategy to select parents which will mate to produce offspring
    tournSel_k = 10

    # no, of variables which will vary,here we have variables(parameters) in X_k for potassium channel
    # this is so because each variables is of same length and are represented by one individual
    # here first 50 bits/genes represent x and the rest 50 represnt y.
    no_of_variables = 8

    # the search space of parameters, in the form as a list of 2 element tuples
    # first element is lower bound, second is higher bound
    param_bounds_wo_h = [(100.0, 140.0), #g_max
                        (-100, -80), #E_rev

                        (0, 50), #M_ma
                        (0, 50), #M_mb
                        (-60, 60), #V_2ma
                        (-60, 60), #V_2mb
                        (-50, -0.5),  # excluding small abs values to avoid operating error in exp, s_ma
                        (0.5, 50)] #s_mb

    # CXPB  is the probability with which two individuals
    #       are crossed or mated
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.2

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # (1.0,) +ve weight for maximization or (-1.0,) -ve weight for minimization
    # since we only have one objective function, we only have one elment in the weights tuple.

    # an Individual is a list with one more attribute called fitness
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # instance of Toolbox class
    toolbox = base.Toolbox()

    # Attribute generator, generation function
    # toolbox.attr_bool(), when called, will draw a random integer between 0 and 1
    # it is equivalent to random.randint(0,1)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # here give the no. of bits in an individual i.e. size_of_individual, here 250
    # depends upon decoding strategy, which uses precision
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, size_of_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    n_points=20


    # Additional functions might be of interest
    def find_steady_state(V, X):
        '''
        Find the steady-state curve at a fixed V, x goes to steady_state as t increses
        V: membrane voltage
        X: the activation or inactivation parameter set
        '''
        return 1 / (1 + X[1]/X[0] * (1+np.exp((V-X[2])/X[4]))/(1+np.exp((V-X[3])/X[5])))


    def find_time_constant(V, X):
        '''
        Find the time constant curve at a fixed V, it governs the rate at which x approaches to the steady state at a fixed V.
        V: membrane voltage
        X: the activation or inactivation parameter set
        '''
        return 1 / (X[0]/(1+np.exp((V-X[2])/X[4])) + X[1]/(1+np.exp((V-X[3])/X[5])))

    # Construct the model
    def openclose_rates_fixedV(V, X):
        '''
        Find the rates of opening and closing of the activation(m)/inactivation(h) gates using hyperbolic tangent functions
        V is the applied memebrane voltage
        X is the parameter set associated with activation/inactivation term
        '''
        a = (X[0]) / (1 + np.exp((V - X[2]) / X[4]))
        b = (X[1]) / (1 + np.exp((V - X[3]) / X[5]))
        return a, b

    def find_x_fixedV(a, b, t_steps, init_cond):
        '''
        Find the activation/inactivation variable analytically with I.C. x(0) = initial_cond
        a, b: the opening and closing rates of of the activation/inactivation gates
        t_steps
        '''
        return (a - (a - (a + b) * init_cond) * np.exp(-t_steps * (a + b))) / (a + b)

    def find_I(p, q, V, m, h, g_max, E_rev):
        return g_max * np.power(m, p) * np.power(h, q) * (V - E_rev)


    def get_current_trace(time_traces, parameters):
        '''
        time_points: np array of n_points time points
        parameters: np array of len(params)

        return a 1d np array

        '''
        current_traces_3d = np.empty((len(input['step_Vs'])+len(input['prestep_Vs']), n_points))

        g_max = parameters[0]
        E_rev = parameters[1]
        X_m = parameters[2:]

        for i in range(time_traces.shape[0]):
            if i < len(input['step_Vs']):
                step_ms = np.array([find_x_fixedV(openclose_rates_fixedV(input['step_Vs'][i], X_m)[0], openclose_rates_fixedV(input['step_Vs'][i], X_m)[1], time_traces[i], find_steady_state(input['prestep_V'], X_m))])
                step_hs = np.array([find_x_fixedV(openclose_rates_fixedV(input['step_Vs'][i], X_h)[0], openclose_rates_fixedV(input['step_Vs'][i], X_h)[1], time_traces[i], find_steady_state(input['prestep_V'], X_h))])
                step_Is = find_I(input['p'], input['q'], input['step_Vs'][i], step_ms, step_hs, g_max, E_rev)
                current_traces_3d[i, :] = step_Is

            else:
                j = i - len(input['step_Vs'])
                prestep_ms = np.array([find_x_fixedV(openclose_rates_fixedV(input['step_V'], X_m)[0], openclose_rates_fixedV(input['step_V'], X_m)[1], time_traces[i], find_steady_state(input['prestep_Vs'][j], X_m))])
                prestep_hs = np.array([find_x_fixedV(openclose_rates_fixedV(input['step_V'], X_h)[0], openclose_rates_fixedV(input['step_V'], X_h)[1], time_traces[i], find_steady_state(input['prestep_Vs'][j], X_h))])
                prestep_Is = find_I(input['p'], input['q'], input['step_V'], prestep_ms, prestep_hs, g_max, E_rev)
                current_traces_3d[i, :] = prestep_Is
        #print(step_Is.shape, prestep_Is.shape, current_traces_3d.shape)

        return current_traces_3d

    def decode_all_x(individual,no_of_variables,bounds):
        '''
        returns list of decoded x in same order as we have in binary format in chromosome
        bounds should have upper and lower limit for each variable in same order as we have in binary format in chromosome
        '''

        len_chromosome = len(individual)
        len_chromosome_one_var = int(len_chromosome/no_of_variables)
        bound_index = 0
        x = []

        # one loop each for x(first 50 bits of individual) and y(next 50 of individual)
        for i in range(0,len_chromosome,len_chromosome_one_var):
            # converts binary to decimial using 2**place_value
            chromosome_string = ''.join((str(xi) for xi in  individual[i:i+len_chromosome_one_var]))
            binary_to_decimal = int(chromosome_string,2)

            # this formula for decoding gives us two benefits
            # we are able to implement lower and upper bounds for each variable
            # gives us flexibility to choose chromosome of any length,
            #      more the no. of bits for a variable, more precise the decoded value can be
            lb = bounds[bound_index][0]
            ub = bounds[bound_index][1]
            precision = (ub-lb)/((2**len_chromosome_one_var)-1)
            decoded = (binary_to_decimal*precision)+lb
            x.append(decoded)
            bound_index +=1

        # returns a list of solutions in phenotype o, here [x,y]
        return x

    def objective_fxn(individual):
    # decoding chromosome to get decoded x in a list
    X = decode_all_x(individual, no_of_variables, param_bounds_wo_h)

    Ik_stim = get_current_trace(target_t_traces, X)

    obj_function_value = 0
    obj_function_value = np.sum(np.square(Ik_stim - target_current_traces))  # fitness of ith voltage trace. i.e. sum_t[(Ik_stim-Ik)^2]
    return [obj_function_value]

    # registering objetive function with constraint
    toolbox.register("evaluate", objective_fxn) # privide the objective function here
    #toolbox.decorate("evaluate", tools.DeltaPenalty(check_feasiblity, 1000, penalty_fxn)) # constraint on our objective function

    # registering basic processes using bulit in functions in DEAP
    toolbox.register("mate", tools.cxTwoPoint) # strategy for crossover, this classic two point crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=probability_of_mutation) # mutation strategy with probability of mutation
    toolbox.register("select", tools.selTournament, tournsize=tournSel_k) # selection startegy

    hall_of_fame = tools.HallOfFame(1)

    stats = tools.Statistics()

    # registering the functions to which we will pass the list of fitness's of a gneration's offspring
    # to ge the results
    stats.register('Min', np.min)
    stats.register('Max', np.max)
    stats.register('Avg', np.mean)
    stats.register('Std', np.std)

    logbook = tools.Logbook()


    # create poppulation as coded in population class
    # no. of individuals can be given as input
    pop = toolbox.population(n=population_size)

    # The next thing to do is to evaluate our brand new population.

    # use map() from python to give each individual to evaluate and create a list of the result
    fitnesses = list(map(toolbox.evaluate, pop))

    # ind has individual and fit has fitness score
    # individual class in deap has fitness.values attribute which is used to store fitness value
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # evolve our population until we reach the number of generations

    # Variable keeping track of the number of generations
    g = 0
    # clearing hall_of_fame object as precaution before every run
    hall_of_fame.clear() #?????

    # Begin the evolution
    while g < no_of_generations:
    # A new generation
    g = g + 1

    #The evolution itself will be performed by selecting, mating, and mutating the individuals in our population.

    # the first step is to select the next generation.
    # Select the next generation individuals using select defined in toolbox here tournament selection
    # the fitness of populations is decided from the individual.fitness.values[0] attribute
    #      which we assigned earlier to each individual
    # these are best individuals selected from population after selection strategy
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals, this needs to be done to create copy and avoid problem of inplace operations
    # This is of utter importance since the genetic operators in toolbox will modify the provided objects in-place.
    offspring = list(map(toolbox.clone, offspring))

    # Next, we will perform both the crossover (mating) and the mutation of the produced children with
    #        a certain probability of CXPB and MUTPB.
    # The del statement will invalidate the fitness of the modified offspring as they are no more valid
    #       as after crossover and mutation, the individual changes

    # Apply crossover and mutation on the offspring
    # note, that since we are not cloning, the changes in child1, child2 and mutant are happening inplace in offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #random.seed(42)
        if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring:
        #random.seed(42)
        if random.random() < MUTPB:
        toolbox.mutate(mutant)
        del mutant.fitness.values


    # Evaluate the individuals with an invalid fitness (after we use del to make them invalid)
    # again note, that since we did not use clone, each change happening is happening inplace in offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    # To check the performance of the evolution, we will calculate and print the
    # minimal, maximal, and mean values of the fitnesses of all individuals in our population
    # as well as their standard deviations.
    # Gather all the fitnesses in one list and print the stats
    #this_gen_fitness = [ind.fitness.values[0] for ind in offspring]
    this_gen_fitness = [] # this list will have fitness value of all the offspring
    for ind in offspring:
        this_gen_fitness.append(ind.fitness.values[0])


    #### SHORT METHOD

    # will update the HallOfFame object with the best individual according to fitness value and weight (while creating base.Fitness class)
    hall_of_fame.update(offspring)


    ####################################### collect parameters for each selected gen
    if g in gens_collect:
        for g_best_indi in hall_of_fame:
        # using values to return the value and
        # not a deap.creator.FitnessMin object

        #g_best_obj_val_overall = g_best_indi.fitness.values[0]
        #print('Minimum value for function: ',g_best_obj_val_overall)

        print(f'{g}th Optimum Solution: ',decode_all_x(g_best_indi, no_of_variables, param_bounds_wo_h))
        gens_best_sol.append(decode_all_x(g_best_indi, no_of_variables, param_bounds_wo_h))
    ########################################

    # pass a list of fitnesses
    # (basically an object on which we want to perform registered functions)
    # will return a dictionary with key = name of registered function and value is return of the registered function
    stats_of_this_gen = stats.compile(this_gen_fitness)

    # creating a key with generation number
    stats_of_this_gen['Generation'] = g

    # printing for each generation
    print(stats_of_this_gen)

    # recording everything in a logbook object
    # logbook is essentially a list of dictionaries
    logbook.append(stats_of_this_gen)


    # now one generation is over and we have offspring from that generation
    # these offspring wills serve as population for the next generation
    # this is not happening inplace because this is simple python list and not a deap framework syntax
    pop[:] = offspring

    # print the best solution using HallOfFame object
    for best_indi in hall_of_fame:
    # using values to return the value and
    # not a deap.creator.FitnessMin object
    best_obj_val_overall = best_indi.fitness.values[0]
    print('Minimum value for function: ',best_obj_val_overall)
    print('Optimum Solution: ',decode_all_x(best_indi,no_of_variables,param_bounds_wo_h))


    # finding the fitness value of the fittest individual of the last generation or
    # the solution at which the algorithm finally converges
    # we find this from logbook

    # select method will return value of all 'Min' keys in the order they were logged,
    # the last element will be the required fitness value since the last generation was logged last
    best_obj_val_convergence = logbook.select('Min')[-1]

    # plotting Generations vs Min to see convergence for each generation
    os.makedirs('GA_plots', exist_ok=True)
    plt.figure(figsize=(20, 10))

    # using select method in logbook object to extract the argument/key as list
    plt.plot(logbook.select('Generation'), logbook.select('Min'))

    plt.title("Minimum values of F_fit Reached Through Generations",fontsize=20,fontweight='bold')
    plt.xlabel("Generations",fontsize=18,fontweight='bold')
    plt.ylabel("Value of Fitness Function",fontsize=18,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')


    # the red line at lowest value of f(x,y) in the last generation or the solution at which algorithm converged
    plt.axhline(y=best_obj_val_convergence,color='r',linestyle='--')

    # the red line at lowest value of f(x,y)
    plt.axhline(y=best_obj_val_overall,color='r',linestyle='--')


    #
    # location of both text in terms of x and y coordinate
    # k is used to create height distance on y axis between the two texts for better readability
    if best_obj_val_convergence > 2:
        k = 0.8
    elif best_obj_val_convergence > 1:
        k = 0.5
    elif best_obj_val_convergence > 0.5:
        k = 0.3
    elif best_obj_val_convergence > 0.3:
        k = 0.2
    else:
        k = 0.1

    # for best_obj_val_convergence
    xyz1 = (no_of_generations/2.4,best_obj_val_convergence)
    xyzz1 = (no_of_generations/2.2,best_obj_val_convergence+(k*3))

    plt.annotate("At Convergence: %0.5f" % best_obj_val_convergence,xy=xyz1,xytext=xyzz1,
                arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
                fontsize=18,fontweight='bold')

    # for best_obj_val_overall
    #xyz2 = (no_of_generations/6,best_obj_val_overall)
    #yzz2 = (no_of_generations/5.4,best_obj_val_overall+(k/0.1))

    #plt.annotate("Minimum Overall: %0.5f" % best_obj_val_overall,xy=xyz2,xytext=xyzz2,
    #             arrowprops=dict(facecolor='black',shrink=1,width=1,headwidth=5),
    #             fontsize=18,fontweight='bold')

    plt.savefig(f'GA_plots/fig {unique_id}')

    for i in range(target_params.shape[0]):
    history_dict[list(param_bounds_wo_h_dict.keys())[i]+'_mse'] = np.mean(np.square(gens_best_sol[-1][i] - target_params[i]))

    history_dict['overall_mse'] = np.mean(np.square(target_params - gens_best_sol[-1]))

    history_dict['target'] = target_params
    history_dict['estimated'] = gens_best_sol[-1]

    def experiment_records(row_data, file_path = 'GA records.csv'): 
        '''
        row_data is a dictionary of the row_data we want to store
        '''
        # Check if the file exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a' if file_exists else 'w', newline='') as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            # If the file is newly created, write the header row
            if not file_exists:
                header_row = row_data.keys() if isinstance(row_data, dict) else row_data
                csv_writer.writerow(header_row)

            # Write the data row
            if isinstance(row_data, dict):
                csv_writer.writerow(row_data.values())
            else:
                csv_writer.writerow(row_data)

    experiment_records(history_dict)

if __name__ == "__main__":
    main()