import random
import numpy as np
import csv

# IMPORT ANN MODEL
# file = 'C:\Pro\ANN\scikit.py'
from GAN import *

# Genetic Algorithm parameters
population_size = 100
num_variables = 2
# variable_range = [(0,1), (-1,2), (-2,1)]
variable_range = [(10, 100), (1, 2)]
mutation_rate = 2
crossover_rate = 0.8
num_generations = 100

# Define the problem
def fitness_function(variables):
    # Example fitness function (minimization problem)
    # x = variables[0]
    # y = variables[1]
    # z = variables[2]
    
    X = variables[0]
    Y = variables[1]
    # Z = variables[2]
    # U = variables[3]
    fitness = (ANN(X, Y))
    # fitness = x**2 + y**2 + z**2
    # fitness = (1.5 - x + x*y)**2 + (2.5 - x + x*(y**2))**2 + (2.625 - x + x*(y**3))**2
    return fitness

# Generate initial population
def initialize_population(population_size, num_variables):
    population = []
    for _ in range(population_size):
        individual = [random.uniform(var_range[0], var_range[1]) for var_range in variable_range]
        population.append(individual)
    return population

# Evaluate the fitness of each individual in the population
def evaluate_population(population):
    fitness_values = []
    for individual in population:
        try:
            fitness = fitness_function(individual)
        except:
            fitness = 9999
            fitness_values.append(fitness)
            continue
        # fitness = fitness_function(individual)
        fitness_values.append(fitness)
    return fitness_values

# Select parents for crossover using tournament selection
def tournament_selection(population, fitness_values, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament_size = 5
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        selected_parents.append(population[winner_index])
    return selected_parents

# Perform crossover between parents to generate offspring
def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, len(parents)-1, 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        child1 = []
        child2 = []
        for j in range(num_variables):
            min_val = min(parent1[j], parent2[j])
            max_val = max(parent1[j], parent2[j])
            child1.append(random.uniform(min_val, max_val))
            child2.append(random.uniform(min_val, max_val))
        offspring.extend([child1, child2])
        # if random.random() < crossover_rate:
        #     cutting_point = random.randint(1, num_variables-1)
        #     child1 = parent1[:cutting_point] + parent2[cutting_point:]
        #     child2 = parent2[:cutting_point] + parent1[cutting_point:]
        #     offspring.extend([child1, child2])
        # else:
        #     offspring.extend([parent1, parent2])
    return offspring

# Perform mutation on the offspring
def mutation(offspring, mutation_rate):
    mutated_offspring = []
    for individual in offspring:
        mutated_individual = individual[:]
        for i in range(num_variables):
            if random.random() < mutation_rate:
                lower_bound, upper_bound = variable_range[i]
                mutated_individual[i] += random.uniform(lower_bound, upper_bound)
        mutated_offspring.append(mutated_individual)
    return mutated_offspring

# Main genetic algorithm loop
population = initialize_population(population_size, num_variables)
for generation in range(num_generations):
    # Evaluate population
    fitness_values = evaluate_population(population)

    # Select parents for crossover
    parents = tournament_selection(population, fitness_values, population_size)

    # Perform crossover
    offspring = crossover(parents, crossover_rate)

    # Perform mutation
    mutated_offspring = mutation(offspring, mutation_rate)

    # Create new population by combining parents and mutated offspring
    population = parents + mutated_offspring

    # Print best individual in the current generation
    best_fitness = min(fitness_values)
    # best_fitness_his = []
    # best_fitness_his.append(best_fitness)

    best_individual = population[fitness_values.index(best_fitness)]
    # best_individual_his = []
    # best_individual_his.append(best_individual)
    print("Generation:", generation+1, "Best Fitness:", (best_fitness), "Best Individual:", best_individual)

    # Extract results to a new file
    file = open('Results.txt', 'a')
    content = ("Generation: " + repr(generation+1) + " | Best Fitness: " + repr((best_fitness)) + " | Best Individual: " + repr(best_individual) + '\n')
    file.writelines(content)
    file.close()
    
