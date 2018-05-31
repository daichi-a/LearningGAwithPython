# -*- coding: utf-8 -*-
from GABaseLibCUDA import PopulationCUDA

if __name__ == '__main__':
    population_size = 1024
    chromosome_length = 50
    random_seed = 1

    population = \
        PopulationCUDA(population_size, chromosome_length, random_seed)
    population.initialize()

    generation_limit = 1000
    mutation_rate = 0.05
    tournament_unit_size = 4
    for i in range(generation_limit):
        population.evaluate()
        print(population.get_best_individual())

        population.onepoint_crossover(tournament_unit_size)
        population.mutation(mutation_rate)

        if population.get_best_score() >= chromosome_length:
            break

    
