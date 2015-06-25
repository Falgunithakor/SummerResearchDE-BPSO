from src.Population import Population
from src.Velocity import Velocity

__author__ = 'FalguniT'
import numpy as np

class DEBPSO(object):
    def __init__(self, population_i):
        self.population_i = population_i
        self.selected_descriptors = None

        self.create_first_velocity_population()

    def create_first_velocity_population(self):
        velocity = Velocity()
        velocity_matrix = velocity.create_first_velocity()

        population = Population(velocity_matrix=velocity_matrix)
        population.create_first_population()

    def fit(self, X, y):
        #self.population_i = X
        self.selected_descriptors = self.OnlySelectTheOnesColumns()

    def transform(self, X):
        return X.T[self.selected_descriptors].T

    def OnlySelectTheOnesColumns(self):
        numOfFea = self.population_i.shape[0]
        xi = np.zeros(numOfFea)
        for j in range(numOfFea):
           xi[j] = self.population_i[j]

        xi = xi.nonzero()[0]
        xi = xi.tolist()
        return xi