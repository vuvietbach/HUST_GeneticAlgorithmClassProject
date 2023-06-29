import argparse
from tqdm import tqdm
from random import sample
import numpy as np
import math
# region global variables
CAPMAX, MEMMAX, CPUMAX = 0, 0, 0
NUM_VNF = 0
MAX_VNF_PER_SERVER = 0
NUM_NODES = 0
NODE_INFO = {}
SERVER_NODE = []
SWITCH_NODE = []
NUM_EDGES = 0
EDGES = []
NUM_REQUESTS = 0
GRAPH = None
NUM_SERVERS = 0
VNF_USED = []
TOTAL_SERVER_COST = 0
TOTAL_SERVER_VNF_COST = 0
REQUEST_INFO = []
TOTAL_DELAY = 0
DISTANCE = None
DEFAULT_SOLUTION = None
# endregion

def parse_args():
    parser = argparse.ArgumentParser(description="parse command line arguments")
    parser.add_argument("--input", type=str, default="dataset/nsf_urban_4/input.txt")
    parser.add_argument("--request", type=str, default="dataset/nsf_urban_4/request10.txt")
    args = parser.parse_args()
    return args

def input(args):
    # open request file
    with open(args.request, 'r') as f:
        lines = f.readlines()
    global NUM_REQUESTS
    NUM_REQUESTS = int(lines[0].strip())
    new_lines = []
    for line in lines[1:]:
        line = line.strip().split(' ')
        line = [float(x) for x in line]
        new_lines.append(line)
    global REQUEST_INFO
    REQUEST_INFO = []
    global CAPMAX, MEMMAX, CPUMAX, VNF_USED
    for line in new_lines:
        request = {"bw": line[0], "mem": line[1], "cpu": line[2], "u": int(line[3]), "v": int(line[4]), "k": int(line[5]), "vnfs": [int(x) for x in line[6:]]}
        REQUEST_INFO.append(request)
        CAPMAX += request["bw"] * (request["k"] + 1)
        MEMMAX += request["mem"] * (request["k"] + 1)
        CPUMAX += request["cpu"] * (request["k"])    
        VNF_USED += request["vnfs"]

    VNF_USED = set(VNF_USED)
    
    # open input file
    with open(args.input, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(" ") for line in lines]
    
    global NUM_VNF, MAX_VNF_PER_SERVER
    NUM_VNF, MAX_VNF_PER_SERVER = [int(x) for x in lines[0]]

    global NUM_NODES, NODE_INFO, SERVER_NODE, SWITCH_NODE, NUM_EDGES, EDGES, GRAPH, NUM_SERVERS 
    NUM_NODES = int(lines[1][0])
    for i in range(2, 2+NUM_NODES):
        arr = [int(x) for x in lines[i]]
        NODE_INFO[arr[0]] = {'delay': arr[1], 'costServer': arr[2]}
        if arr[2] >= 0:
            NODE_INFO[arr[0]]["vnf_cost"] = arr[3:]
            SERVER_NODE.append(arr[0])
        else:
            SWITCH_NODE.append(arr[0])
    
    current_line = 2 + NUM_NODES
    NUM_EDGES = int(lines[current_line][0])
    GRAPH = {}
    for i in range(current_line + 1, current_line + 1 + NUM_EDGES):
        u, v, delay = [int(x) for x in lines[i]]
        EDGES.append([u, v, delay])
        if u not in GRAPH:
            GRAPH[u] = {}
        if v not in GRAPH:
            GRAPH[v] = {}
        GRAPH[u][v] = delay
        GRAPH[v][u] = delay
    
    NUM_SERVERS = len(SERVER_NODE)

def calculate_distance():
    global DISTANCE
    DISTANCE = np.ones((NUM_NODES, NUM_NODES)) * -1
    for i in range(NUM_NODES):
        DISTANCE[i][i] = 0
    # Initialize distances for direct edges
    for u in GRAPH:
        for v in GRAPH[u]:
            DISTANCE[u][v] = GRAPH[u][v]

    # Update distances using the Floyd-Warshall algorithm
    for k in range(NUM_NODES):
        for i in range(NUM_NODES):
            if DISTANCE[i][k] > 0:
                for j in range(NUM_NODES):
                    if DISTANCE[k][j] > 0:
                        if DISTANCE[i][j] == -1:
                            DISTANCE[i][j] = DISTANCE[i][k] + DISTANCE[k][j]
                        else:
                            DISTANCE[i][j] = min(DISTANCE[i][j], DISTANCE[i][k] + DISTANCE[k][j])



def initialize():
    global TOTAL_SERVER_COST, TOTAL_SERVER_VNF_COST, TOTAL_DELAY
    TOTAL_SERVER_COST = 0
    TOTAL_SERVER_VNF_COST = 0
    for i in range(NUM_SERVERS):
        sid = SERVER_NODE[i]
        TOTAL_SERVER_COST += NODE_INFO[sid]["costServer"]
        TOTAL_SERVER_VNF_COST += np.sum(NODE_INFO[sid]['vnf_cost'])
    
    TOTAL_DELAY = 0
    for node_id in NODE_INFO:
        TOTAL_DELAY += NODE_INFO[node_id]['delay']
    for edge in EDGES:
        TOTAL_DELAY += edge[2]
    TOTAL_DELAY *= NUM_REQUESTS

    global DISTANCE
    calculate_distance()

def main():
    args = parse_args()
    input(args)
    initialize()
    global DEFAULT_SOLUTION
    DEFAULT_SOLUTION = Solution.get_default_solution()
    
    spea = SPEA2(args)
    spea.run()



class Solution:
    @classmethod
    def get_default_solution(cls):
        default_num_vnf = np.ones((NUM_SERVERS)) * MAX_VNF_PER_SERVER
        default_num_vnf = default_num_vnf.astype(int)
        vnf_used_list = list(VNF_USED)
        num_rep = int((np.sum(default_num_vnf) // len(vnf_used_list)).item())
        rem = int((np.sum(default_num_vnf) % len(vnf_used_list)).item())

        default_vnfs = np.array(vnf_used_list * num_rep)
        default_vnfs = np.concatenate((default_vnfs, np.array(vnf_used_list[:rem])))
        default_vnfs = default_vnfs.astype(int)
        return Solution(default_num_vnf, default_vnfs)

    @classmethod
    def is_valid(cls, num_vnf, vnfs):
        if np.sum(num_vnf) < len(VNF_USED):
            return False
        if isinstance(vnfs, np.ndarray):
            vnfs = vnfs.tolist()
        if not VNF_USED.issubset(set(vnfs)):
            return False
        
        return True
    
    def __init__(self, num_vnf, vnfs):
        self.num_vnf = num_vnf
        self.vnfs = vnfs
        
        server_id = []
        for i in range(NUM_SERVERS):
            server_id += [SERVER_NODE[i]] * int(self.num_vnf[i])
        self.vnf_servers = {}
        for i in range(len(self.vnfs)):
            vid = self.vnfs[i]
            if vid not in self.vnf_servers:
                self.vnf_servers[vid] = []
            self.vnf_servers[vid].append(server_id[i])
                
        self.cost = Solution.evaluate_solution(self)
    
    MAX_PATHS = 20
    def compute_delay(self, request):
        current_paths = {request['u']:0}
        for i in range(request['k']):
            vid = request['vnfs'][i]

            options = self.vnf_servers[vid]

            next_paths = {option: float("inf") for option in options}
            for current_node in current_paths:
                for next_node in options:
                    if DISTANCE[current_node][next_node] > -1:
                        next_paths[next_node] = min(next_paths[next_node], \
                        current_paths[current_node]+DISTANCE[current_node][next_node]+NODE_INFO[next_node]['delay'])
            sorted_paths = sorted(next_paths.items(), key = lambda x: x[1])
            sorted_paths = sorted_paths[:min(Solution.MAX_PATHS, len(sorted_paths))]
            current_paths = {path[0]:path[1] for path in sorted_paths if path[1] < float("inf")}

        
        min_delay = float("inf")
        for current_node in current_paths:
            if DISTANCE[current_node][request['v']] > -1:
                min_delay = min(min_delay, current_paths[current_node] + DISTANCE[current_node][request['v']])
        return min_delay

    @classmethod
    def evaluate_solution(cls, solution):
        global TOTAL_SERVER_COST, TOTAL_SERVER_VNF_COST, TOTAL_DELAY
        # calculate placement cost
        CS =  0
        for i in range(NUM_SERVERS):
            if solution.num_vnf[i] > 0:
                CS += NODE_INFO[SERVER_NODE[i]]["costServer"]
        tmp = (TOTAL_SERVER_COST if TOTAL_SERVER_COST else 1)
        CS /= tmp
        # calculate installation cost
        server_id = []
        for i in range(NUM_SERVERS):
            server_id += [i] * solution.num_vnf[i]
        CV = 0
        NUM_VNF = len(solution.vnfs)
        for i in range(NUM_VNF):
            sid = SERVER_NODE[server_id[i]]
            vnf_id = solution.vnfs[i]
            CV += NODE_INFO[sid]["vnf_cost"][vnf_id]
        tmp = (TOTAL_SERVER_VNF_COST if TOTAL_SERVER_VNF_COST else 1)
        CV /= tmp
        # caluculate latency
        DL = 0
        for i in range(NUM_REQUESTS):
            DL += solution.compute_delay(REQUEST_INFO[i])
        tmp = (TOTAL_DELAY if TOTAL_DELAY else 1)
        DL /= tmp
        return np.array([CS, CV, DL])

    def dominates(self, other_solution):
        return np.all(self.cost <= other_solution.cost) & np.any(self.cost < other_solution.cost)
    
    def compute_distance(self, other_solution):
        return np.linalg.norm(self.cost-other_solution.cost)


    @classmethod 
    def generate_random_solution(cls):
        global VNF_USED, NUM_SERVERS, NUM_VNF, MAX_VNF_PER_SERVER
        
        max_trial = 40
        
        found = False
        for _ in range(max_trial):
            num_vnf = np.random.randint(MAX_VNF_PER_SERVER+1, size=(NUM_SERVERS))
            if np.sum(num_vnf) >= len(VNF_USED):
                found = True
                break
        if not found:
            return DEFAULT_SOLUTION
        
        found = False
        for _ in range(max_trial):
            vnfs = []
            for i in range(NUM_SERVERS):
                vnfs += sample(range(NUM_VNF), num_vnf[i])
            
            vnfs_set = set(vnfs)
            if VNF_USED.issubset(vnfs_set):
                found = True
                vnfs = np.array(vnfs)
                break
        if found:
            return Solution(num_vnf, vnfs)
        
        return DEFAULT_SOLUTION


    @classmethod
    def crossover(cls, solution1, solution2):
        if NUM_SERVERS > 1:
            threshold = np.random.randint(1, NUM_SERVERS-1)
            num_vnf1 = np.concatenate((solution1.num_vnf[:threshold], solution2.num_vnf[threshold:]))
            num_vnf2 = np.concatenate((solution2.num_vnf[:threshold], solution1.num_vnf[threshold:]))
            prefix1 = np.sum(solution1.num_vnf[:threshold])
            prefix2 = np.sum(solution2.num_vnf[:threshold])
            vnfs1 = np.concatenate((solution1.vnfs[:prefix1], solution2.vnfs[prefix2:]))
            vnfs2 = np.concatenate((solution2.vnfs[:prefix2], solution1.vnfs[prefix1:]))

            if not Solution.is_valid(num_vnf1, vnfs1):
                sol1 = Solution.generate_random_solution()
            else:
                sol1 = Solution(num_vnf1, vnfs1)
            if not Solution.is_valid(num_vnf2, vnfs2):
                sol2 = Solution.generate_random_solution()
            else:
                sol2 = Solution(num_vnf2, vnfs2)
            
            return sol1, sol2
    
    @classmethod
    def mutate(cls, solution):
        MAX_TRIAL = 20
        activated_servers = np.where(solution.num_vnf > 0)[0]
        for i in range(MAX_TRIAL):
            server = np.random.choice(activated_servers)
            vnf_start = np.sum(solution.num_vnf[:server])
            vnf_end = vnf_start + solution.num_vnf[server]
            
            vnf_used = set(np.concatenate((solution.vnfs[:vnf_start], solution.vnfs[vnf_end:])))
            vnf_lack = VNF_USED - vnf_used

            if len(vnf_lack) > MAX_VNF_PER_SERVER:
                continue

            server_vnf = np.concatenate((list(vnf_lack), np.random.choice(list(vnf_used), size=(MAX_VNF_PER_SERVER-len(vnf_lack)), replace=False)))
            new_num_vnf = solution.num_vnf.copy()
            new_num_vnf[server] = len(server_vnf)
            new_vnf = np.concatenate((solution.vnfs[:vnf_start], server_vnf, solution.vnfs[vnf_end:])).astype(np.int32)
            
            if Solution.is_valid(new_num_vnf, new_vnf):
                return Solution(new_num_vnf, new_vnf)
        
        return Solution.generate_random_solution()


class SPEA2():
    def __init__(self, args):
        self.args = args

        self.max_iter = 100
        self.population_size = 100
        self.archive_size = 100

        self.k = int(math.sqrt(self.population_size+self.archive_size))


        self.num_crossover = 100
        self.num_mutation = 100
    
    def initialize_population(self):
        population = [Solution.generate_random_solution() for _ in range(self.population_size)]
        archive = []
        return population, archive
    

    def compute_total_population(self, population, archive):
        total_population = population + archive
        total_size = len(total_population)
        
        # calculate s score
        for i in range(0, total_size):
            total_population[i].s = 0
            total_population[i].dominated_by = []
        
        for i in range(0, total_size):
            for j in range(i+1, total_size):
                if total_population[i].dominates(total_population[j]):
                    total_population[i].s += 1
                    total_population[j].dominated_by.append(total_population[i])
                elif total_population[j].dominates(total_population[i]):
                    total_population[j].s += 1
                    total_population[i].dominated_by.append(total_population[j])
                
        for element1 in total_population:        
            element1.r = 0
            for element2 in element1.dominated_by:
                element1.r += element2.s
            
            distance = [element1.compute_distance(element2) for element2 in total_population]
            distance = sorted(distance)
            element1.d = 1 / (distance[self.k] + 2)
            
            element1.f = element1.d + element1.r
        
        return total_population
    
    def compute_archive(self, total_population):
        archive = []
        for element in total_population:
            if element.r == 0:
                archive.append(element)
        
        if len(archive) <= self.archive_size:
            population = sorted(total_population, key = lambda x: x.f)
            archive = population[:self.archive_size]
        else:
            while len(archive) > self.archive_size:
                cur_size = len(archive)
                # calculate distance from one solution to another
                for i in range(cur_size):
                    archive[i].dist = []            
                for i in range(cur_size):
                    for j in range(i+1, cur_size):
                        dist = archive[i].compute_distance(archive[j])
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
    
    def binary_tournament_selection(self, archive):
        solutions = sample(archive, 2)
        return solutions[0] if solutions[0].f < solutions[1].f else solutions[1]
    
    @classmethod
    def mutate(solution):
        return Solution()
    
    def crossover(self, solution1, solution2):
        pass
    
    def run(self):
        population, archive = self.initialize_population()
        min_cost = float('inf')
        best_solution = None
        for i in tqdm(range(self.max_iter)):
            total_population = self.compute_total_population(population, archive)
            archive = self.compute_archive(total_population)
            # crossover
            crossover_population = []
            for i in range(0, self.num_crossover//2):
                solution_1 = self.binary_tournament_selection(archive)
                solution_2 = self.binary_tournament_selection(archive)
                solution_1, solution_2 = Solution.crossover(solution_1, solution_2)
                crossover_population += [solution_1, solution_2]
            # mutation
            mutation_population = []
            for i in range(self.num_mutation):
                solution = self.binary_tournament_selection(archive)
                solution = Solution.mutate(solution)
                mutation_population += [solution]

            population = crossover_population + mutation_population
            
            for element in population:
                tmp_cost = np.sum(element.cost)
                if(tmp_cost < min_cost):
                    min_cost = tmp_cost
                    best_solution = element
        
        # get best soluti
        cost = best_solution.cost
        CS, CV, DL = cost[0], cost[1], cost[2]
        # logging
        with open('result.txt', 'a') as f:
            # f.write(f"INPUT_FILE: {self.args.input}\n")
            # f.write(f"REQUEST_FILE: {self.args.request}\n")
            # f.write(f'TOTAL: {min_cost} COST_SERVER: {CS} COST_VNF: {CV} DELAY: {DL}\n')
            f.write(f'{self.args.input} {self.args.request} {min_cost} {CS} {CV} {DL}\n')

def debug():
    import pdb; pdb.set_trace()
    x = Solution.generate_random_solution()
    print(x.num_vnf)

if __name__ == '__main__':
    main()
    # debug()