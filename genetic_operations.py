import numpy as np
import random
import pandas as pd
import itertools
from iteration_utilities import random_combination


random.seed(123)
np.random.seed(123)

def generate_population(n_chromosomes,n_genes):
    population = np.random.randint(0,2,(n_chromosomes,n_genes))
    return population

def crossover(population1,population2,method="onepoint"):
    n_chromosomes,n_genes = population1.shape

    if method=="onepoint":
        # crossover of chromosome 1 and chromosome 1
        chromosomes_crossed = np.column_stack((population1[:,:n_genes//2],population2[:,n_genes//2:]))
    elif method=="multipoint":
        chromosomes_crossed = np.empty((n_chromosomes,n_genes),dtype=population1.dtype)
        chromosomes_crossed[:,0::2] = population1[:,0::2]
        chromosomes_crossed[:,1::2] = population2[:,1::2]

    return chromosomes_crossed

def crossover_population(population,size=None,method="onepoint"):
    n_chromosomes,n_genes = population.shape

    assert size is not None, "Cannot have no size for population!!!"

    pop1idx = np.random.randint(0,n_chromosomes,size=size)
    pop2idx = np.random.randint(0,n_chromosomes,size=size)

    population1 = population[pop1idx,:]
    population2 = population[pop2idx,:]
    population_crossed = crossover(population1,population2,method=method)

    return population_crossed


def mutation(population,mutation_rate):
    n_chromosomes,n_genes = population.shape

    mutation_mask = np.random.rand(n_chromosomes,n_genes) < mutation_rate
    
    chromosome_mutated = np.logical_xor(population,mutation_mask).astype(int)

    return chromosome_mutated


