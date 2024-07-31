import numpy as np
from GAN import *

def differential_evolution(func, bounds, pop_size=20, mutation_factor=1.9, crossover_prob=0.9, max_generations=100):
    """
    Differential Evolution Algorithm
    :param func: The objective function to minimize
    :param bounds: List of tuples specifying the (min, max) bounds for each variable
    :param pop_size: Size of the population
    :param mutation_factor: Differential weight (0 < mutation_factor < 2) 0.8
    :param crossover_prob: Crossover probability (0 < crossover_prob < 1) 0.7
    :param max_generations: Maximum number of generations
    :return: Best individual and its fitness value
    """
    dimensions = len(bounds)
    
    # Initialize population
    population = np.random.rand(pop_size, dimensions)
    for i in range(dimensions):
        population[:, i] = bounds[i][0] + population[:, i] * (bounds[i][1] - bounds[i][0])
    
    # Evaluate initial population
    fitness = np.asarray([func(*ind) for ind in population])
    
    for generation in range(max_generations):
        new_population = np.copy(population)
        
        for i in range(pop_size):
            # Mutation
            indices = [index for index in range(pop_size) if index != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), 0, 1)
            
            # Crossover
            crossover = np.random.rand(dimensions) < crossover_prob
            if not np.any(crossover):
                crossover[np.random.randint(0, dimensions)] = True
            trial = np.where(crossover, mutant, population[i])
            
            # Ensure trial vector is within bounds
            for j in range(dimensions):
                trial[j] = bounds[j][0] + trial[j] * (bounds[j][1] - bounds[j][0])
            
            # Selection
            trial_fitness = func(*trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                fitness[i] = trial_fitness
        
        population = new_population
        
        # Print best fitness and individual
        best_index = np.argmin(fitness)
        print(f"Generation {generation+1}: Best Fitness = {fitness[best_index]}, Best Individual = {population[best_index]}")

        file = open('Results.txt', 'a')
        content = (f"Generation {generation+1}: Best Fitness = {fitness[best_index]}, Best Individual = {population[best_index]}")
        file.writelines(content)
        file.write("\n")
        file.close()
    return population[np.argmin(fitness)], np.min(fitness)


# Example usage
def example_function(x, y):
    return ANN(x,y)

# Define bounds for multiple variables
bounds = [(10,100), (1,2)]  

best_individual, best_fitness = differential_evolution(example_function, bounds)
print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")