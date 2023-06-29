# seed the pseudorandom number generator
from random import random, sample, seed
import math
import numpy as np
import random
from tqdm import tqdm


def ZDT1(x):
    sum = 0
    for t in x:
        sum += t
    f1 = x[0]
    g = 1 + 9 * sum / (len(x) - 1)
    h = 1 - math.sqrt(f1 / g)
    f2 = g * h
    return [f1, f2]

def ZDT2(x):
    sum = 0
    for t in x:
        sum += t
    f1 = x[0]
    g = 1 + 9 * sum / (len(x) - 1)
    h = 1 - (f1 / g)**2
    f2 = g * h
    return [f1, f2]

OBJ_FUN = ZDT1
NUM_VAR = 30
NUM_FRONT = 100
MIN_VAL = 0
MAX_VAL = 1


def IGD(pareto_front, ref_point):
    sum = 0
    for i in pareto_front:
        min_dist = i.cal_dist(ref_point[0])
        for j in ref_point:
            min_dist = min(min_dist, i.cal_dist(j))
        sum += min_dist**2
    return math.sqrt(sum)/len(pareto_front)

def get_pareto_front(num_front):
    res = []
    x = np.zeros((NUM_VAR))
    for i in range(num_front):
        x[1] = i/num_front
        res.append(Solution(x))
    return res

class Solution:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.cost = np.array(OBJ_FUN(self.pos))
    
    def dominates(self, other_sol):
        return np.all(self.cost <= other_sol.cost) & np.any(self.cost < other_sol.cost)
    
    def cal_dist(self, other_sol):
        return np.linalg.norm(self.cost-other_sol.cost)

class SPEA2:
    def __init__(self):
        # setting:
        self.m = 2
        self.population_size = 100
        self.archive_size = 100
        self.max_iter = 100
        
        self.k = int(math.sqrt(self.population_size+self.archive_size))
        
        # mutation
        self.num_mutation = 100
        self.sigma = 0.2*(MAX_VAL-MIN_VAL)
        # crossover
        self.num_crossover = 100
        self.gamma = 0.1
    
    def init_pop(self):
        self.pop = []
        self.pop = [Solution(np.random.uniform(MIN_VAL, MAX_VAL, NUM_VAR)) \
                        for i in range(self.population_size)]
        self.archive = []

    def Crossover(self, x1, x2):
        gamma = self.gamma
        
        alpha = np.random.uniform(-gamma, 1+gamma, len(x1))
        
        y1 = alpha*x1+(1-alpha)*x2
        y2 = alpha*x2+(1-alpha)*x1
        
        y1 = np.clip(y1, MIN_VAL, MAX_VAL)
        y2 = np.clip(y2, MIN_VAL, MAX_VAL)
        
        return y1, y2
    
    def Mutate(self, x):
        y = x+self.sigma*np.random.normal(size=len(x))
        y = np.clip(y, MIN_VAL, MAX_VAL)
        return y

    def cal_fitness(self, pop, arch):
        total_pop = pop + arch
        size_total_pop = len(total_pop)
        
        # calculate s score
        for i in range(0, size_total_pop):
            total_pop[i].s = 0
            total_pop[i].dominated_by = []
        
        for i in range(0, size_total_pop):
            for j in range(i+1, size_total_pop):
                if total_pop[i].dominates(total_pop[j]):
                    total_pop[i].s += 1
                    total_pop[j].dominated_by.append(total_pop[i])
                elif total_pop[j].dominates(total_pop[i]):
                    total_pop[j].s += 1
                    total_pop[i].dominated_by.append(total_pop[j])
                
        for elem in total_pop:        
            elem.r = 0
            for elem1 in elem.dominated_by:
                elem.r += elem1.s
            
            distance = [elem.cal_dist(elem1) for elem1 in total_pop]
            distance = sorted(distance)
            elem.d = 1 / (distance[self.k] + 2)
            
            elem.f = elem.d + elem.r
        
        return total_pop
    
    def BinaryTournamentSelection(self, archive):
        sol = sample(archive, 2)
        return sol[0] if sol[0].f < sol[1].f else sol[1]

    def cal_archive(self, pop):
        archive = []
        for i in pop:
            if i.r == 0:
                archive.append(i)
        
        if len(archive) <= self.archive_size:
            pop = sorted(pop, key = lambda x: x.f)
            archive = pop[:self.archive_size]
        else:
            while len(archive) > self.archive_size:
                cur_size = len(archive)
                # calculate distance from one solution to another
                for i in range(cur_size):
                    archive[i].dist = []            
                for i in range(cur_size):
                    for j in range(i+1, cur_size):
                        dist = archive[i].cal_dist(archive[j])
                        archive[i].dist.append(dist)
                        archive[j].dist.append(dist)          
                for i in range(cur_size):
                    archive[i].dist=sorted(archive[i].dist)
                
                # find solution with min distance to other solution
                for i in range(0, cur_size-1):
                    min_v = archive[0].dist[i]
                    max_v = min_v
                    for j in range(1, cur_size):
                        min_v = min(min_v, archive[j].dist[i])
                        max_v = max(max_v, archive[j].dist[i])
                    if min_v < max_v:
                        break
                
                min_v = archive[0].dist[i]
                pos = 0
                for j in range(1, cur_size):
                    if  archive[j].dist[i] < min_v:
                        min_v =  archive[j].dist[i]
                        pos = j
                
                archive.pop(pos)
        
        return archive
    
    def run(self):
        self.init_pop()
        pareto_front = get_pareto_front(NUM_FRONT)
        for i in tqdm(range(self.max_iter)):
            # assign fitness score
            self.total_pop = self.cal_fitness(self.pop, self.archive)
            # update archive
            self.archive = self.cal_archive(self.total_pop)
            
            #TODO: code stop criteria
            # crossover
            crossover_population = []
            for i in range(0, self.num_crossover//2):
                sol1 = self.BinaryTournamentSelection(self.archive)
                sol2 = self.BinaryTournamentSelection(self.archive)
                pos1, pos2 = self.Crossover(sol1.pos, sol2.pos)
                crossover_population += [Solution(pos1), Solution(pos2)]
            # mutation
            mutation_population = []
            for i in range(self.num_mutation):
                p = self.BinaryTournamentSelection(self.archive)
                pos = self.Mutate(p.pos)
                mutation_population += [Solution(pos)]
            
            self.pop = crossover_population + mutation_population
        
        performance = IGD(self.archive, pareto_front)
        print(performance)

def main():
    spea = SPEA2()
    spea.run()

if __name__ == '__main__':
    main()