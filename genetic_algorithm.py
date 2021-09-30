
import pandas as pd
import networkx
import numpy as np
import numpy.matlib
from numpy.random import randint
from numpy.random import rand
import random
from time import time
from fastga import FastMutationOperator

# ------------- High Quality Solutions -------------

def initSolution(G):
    Y = 1000000000
    S = [[i+1] for i in range(len(G.nodes()))]
    nodes_list = np.array(list(G.nodes()))
    A = networkx.to_numpy_array(G)
    
    
    
    while len(nodes_list) > 1:
        
        v1,v2,FCB = edgeContraction(A)
        if(FCB == -1):
            return Y,S
        G = networkx.contracted_nodes(G, nodes_list[v1], nodes_list[v2],copy=False)
        nodes_list = np.array(list(G.nodes()))
        
        A[v1][v1] = 1 + A[v1][v1] + A[v2][v2]
        
        A = np.delete(A,v2,0)
        A = np.delete(A,v2,1)
        
 
        
        if FCB < Y:
            Y = FCB
            S[v1].extend(S[v2])
            S.remove(S[v2])
            
        
def edgeContraction(A):
    
    triu_A = np.triu(A,0)
    [row_B,col_B] = np.nonzero(np.triu(A,1)>0)
    
    # Step 2
    B = triu_A[np.nonzero(np.triu(A,1)>0)]
    if(len(B)==0):
        return -1,-1,-1
    index_r = sub2ind(triu_A.shape,row_B+1,row_B+1)
    index_c = sub2ind(triu_A.shape,col_B+1,col_B+1)
    
    index_r = [i-1 for i in index_r]
    index_c = [i-1 for i in index_c]
    
    flag_row_B = triu_A[np.unravel_index(index_r,triu_A.shape,'F')] == 0
    flag_col_B = triu_A[np.unravel_index(index_c,triu_A.shape,'F')] == 0
    
    # Step 3
    de_row_B = np.sum(np.multiply(A[row_B,:],np.matlib.repmat(np.transpose(np.diag(A)==0),len(row_B),1)),axis = 1) - triu_A[np.unravel_index(index_r,triu_A.shape,'F')]
    de_col_B = np.sum(np.multiply(A[col_B,:], np.matlib.repmat(np.transpose(np.diag(A)==0),len(row_B),1)),axis=1) - triu_A[np.unravel_index(index_c,triu_A.shape,'F')]
    
    # Step 4
    
    verticesWithoutWeight = np.setdiff1d([i for i in range(triu_A.shape[1])], np.nonzero(np.diag(triu_A)>0))
    A1 = triu_A[verticesWithoutWeight,:]
    
    # Step 5
    
    co_B = np.sum(np.sum(A1)) - flag_row_B * de_row_B - flag_col_B * de_col_B + flag_row_B * flag_col_B * B
    
    # Step 6 
    
    co_M = B + triu_A[np.unravel_index(index_r,triu_A.shape,'F')] + triu_A[np.unravel_index(index_c,triu_A.shape,'F')]
    
    # Step 7
    
    TOTALWEIGHT = np.sum(triu_A)
    
    cut = TOTALWEIGHT - np.sum(np.diag(triu_A))- B - co_B
    
    # Step 8 
    
    max_co = np.maximum(np.maximum(co_M,co_B),max(np.diag(triu_A)))
    
    # Step 9
    
    FCB = (cut + max_co)/TOTALWEIGHT
    rcLoc = np.where(FCB == np.amin(FCB))
    
    if len(rcLoc[0])>1:
        e = random.randint(0,rcLoc[0].shape[0]-1)
        rcLoc1 = rcLoc[0][e]
        return row_B[rcLoc1],col_B[rcLoc1],np.amin(FCB)
    
    rcLoc1 = rcLoc[0][0]
    
    
    return row_B[rcLoc1],col_B[rcLoc1],np.amin(FCB)                                   
    
    
def sub2ind(sz, row , col):
    n_rows = sz[0] 
    return [n_rows * (c-1) + r for r,c in zip(row,col)] 


#-------------------- Algorithm ----------------


# Objective function
def fitnessFunction(chromo,G):
    
    A = networkx.to_numpy_array(G)
    modules = set(chromo)
    modules = list(modules)
    d = []
    
    for i in modules:
        
        mask = [i]*len(G.nodes)
        
        b = np.in1d(chromo,mask)
        
        c = np.outer(b , np.transpose(b))
        
        mul = np.multiply(A,c)
        
        d.append(np.sum(mul)/2)
        
    
    Coupling = np.sum(A) - np.sum(d)
    T = Coupling + np.sum(d)
    FCB = (Coupling + max(d))/T
    return FCB

# tournament selection
def selection(pop,scores,k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

# Laplace crossover two parents to create two children
def crossover(p1, p2, r_cross):
    
    
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        
        """
        
        for i in range(len(c1)):
            
            while True:
                # Laplace parameters
                a = 1
                b = 0.7
                u =  np.random.uniform(0,1)
                
                if u <= 0.5 :
                    beta = a - b*np.log(u)
                else:
                    beta = a + b*np.log(u)
                c1[i] = round(p1[i] + beta*np.absolute(p1[i]-p2[i]))
                c2[i] = round(p2[i] + beta*np.absolute(p1[i]-p2[i]))
                
                if (c1[i] >= 1 and c1[i] <= 491 and c2[i] >= 1 and c2[i] <= 491):
                    break
    
         """       
        
    return [c1, c2]

# mutation operator
def mutation(chromo,r_mut):
    
    for i in range(len(chromo)):
        # check for a mutation
        if rand() < r_mut:
            chromo[i] = randint(1,len(chromo))
    
    
            
# genetic algorithm
def genetic_algorithm(objective, n_genes, n_iter, n_pop,n_elit,r_cross,r_mut,G,hq):
    
    
    # random population seeded with 60 high quality solutions
    pop = [randint(1,len(G.nodes), n_genes).tolist() for _ in range(n_pop-len(hq))]
    pop.extend(hq)
    
    # keep track of best solution
    best, best_eval = 0, objective(pop[0],G)
    
    # enumerate generations
    for gen in range(n_iter):
        
        # evaluate all candidates in the population
        scores = [objective(c,G) for c in pop]
        
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best = %.3f" % (gen, scores[i]))
                
        # select parents & elitism
        selected = [selection(pop, scores) for _ in range(n_pop-n_elit)]
        
        sort_indices = np.argsort(scores)
        
        best_indices = sort_indices[-n_elit:]
        elit = [pop[i] for i in best_indices]
        selected.extend(elit)
        
        
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c,r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        
    return [best, best_eval]          
      

dataset = pd.read_csv("C:/Users/User/Desktop/Thesis/Dataset/log4j_1.0.4_features.csv",header=None)

#Convert to Series
data = dataset.iloc[:,:][0]

indexes = []

# Keep only function dependecies
for i in range(data.size):
    if data[i].count(')')==2:
        indexes.append(i)
        

data = data.loc[indexes]
data.index = range(data.size)

G = networkx.Graph()


for i in range(data.size):
    temp = data[i].split(";")
    G.add_node(temp[0])
    G.add_node(temp[1])
    G.add_edge(temp[0],temp[1])
    

#G = networkx.erdos_renyi_graph(200,0.8)
G1 = G.copy()    
GraphCopies = []
GraphCopies.append(G)

HQsolutions = [[0] for _ in range(60)]
hqpops = np.zeros((60,len(G1.nodes)),dtype=int)

start_time = time()
for i in range(60):
    
    GraphCopies.append(GraphCopies[i].copy())
    FCB, HQsolution = initSolution(GraphCopies[i])
    HQsolutions[i] = HQsolution

    # Transform High Quality Solutions to chromosomes
    hqpop = np.zeros(len(G1.nodes),dtype=int)
    
    module = 0
    for j in HQsolution:
        module = module + 1
        for k in j:
            hqpop[k-1] = module
            
    hqpops[i] = hqpop
    
end_time = time()
t = end_time - start_time
    



# define the total iterations
n_iter = 400
# genes
n_genes = len(G1.nodes)
# population size
n_pop = max(min(10*len(G1.nodes),200),40)
# elitism archive
n_elit = int(0.05*max(min(10*len(G1.nodes),200),40))
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_genes)
# perform the genetic algorithm search
best, score = genetic_algorithm(fitnessFunction, n_genes, n_iter, n_pop,n_elit,r_cross,r_mut,G1,hqpops.tolist())
print('Done!')
print('f(%s) = %f' % (best, score))

solution = [1, 58, 3, 3, 3, 3, 3, 4, 4, 4, 3, 3, 6, 6, 1, 1, 1, 1, 87, 8, 9, 9, 9, 9, 10, 9, 9, 9, 10, 11, 11, 12, 12, 13, 13, 13, 321, 4, 15, 15, 16, 16, 6, 6, 3, 3, 10, 17, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 4, 6, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 23, 23, 23, 4, 4, 3, 22, 24, 24, 22, 25, 26, 26, 3, 24, 22, 27, 28, 29, 30, 30, 30, 31, 31, 22, 22, 22, 24, 24, 24, 27, 22, 201, 24, 24, 4, 15, 16, 107, 22, 196, 390, 37, 38, 38, 38, 38, 39, 38, 1, 40, 40, 41, 1, 1, 42, 42, 43, 43, 43, 44, 45, 45, 45, 13, 46, 46, 456, 48, 40, 6, 48, 43, 43, 43, 43, 49, 49, 50, 50, 50, 50, 50, 51, 6, 52, 31, 52, 17, 31, 31, 31, 31, 31, 53, 54, 54, 472, 18, 18, 18, 18, 56, 56, 57, 18, 58, 59, 59, 59, 59, 59, 59, 59, 59, 4, 60, 60, 61, 60, 4, 62, 63, 63, 64, 65, 67, 63, 21, 63, 112, 63, 62, 386, 69, 70, 467, 72, 235, 44, 12, 61, 59, 1, 18, 40, 41, 436, 76, 77, 77, 77, 78, 3, 78, 79, 79, 43, 80, 81, 82, 9, 83, 49, 83, 463, 83, 9, 85, 86, 86, 87, 87, 88, 88, 89, 90, 88, 90, 88, 313, 395, 88, 89, 94, 94, 450, 88, 94, 94, 234, 235, 384, 254, 100, 83, 83, 83, 101, 101, 102, 102, 4, 149, 104, 4, 217, 102, 106, 103, 63, 10, 58, 10, 466, 112, 42, 56, 115, 116, 18, 116, 10, 10, 101, 117, 197, 119, 300, 101, 105, 175, 117, 9, 119, 123, 123, 124, 125, 125, 126, 126, 127, 43, 10, 60, 129, 130, 43, 43, 296, 1, 132, 132, 42, 44, 10, 240, 10, 13, 6, 3, 17, 135, 136, 136, 137, 138, 138, 138, 358, 140, 140, 140, 9, 438, 141, 43, 142, 141, 143, 30, 13, 43, 165, 235, 147, 148, 6, 363, 6, 25, 25, 25, 150, 151, 1, 152, 152, 10, 188, 152, 44, 44, 44, 152, 154, 154, 44, 13, 6, 155, 155, 25, 25, 59, 59, 155, 155, 156, 150, 156, 1, 393, 1, 1, 1, 127, 3, 3, 455, 156, 13, 159, 159, 159, 265, 43, 43, 43, 161, 83, 50, 404, 83, 165, 166, 167, 168, 168, 168, 10, 10, 169, 479, 171, 171, 171, 171, 171, 173, 171, 164, 168, 168, 176, 10, 177, 177, 1, 42, 255, 76, 78, 179, 180, 77, 181, 58, 1, 40, 456, 183, 184, 185, 185, 185, 19, 185, 186, 6, 10, 10, 112, 112, 187, 187, 188, 4, 197, 190, 50, 192, 10, 4, 112, 106, 430, 190, 190, 64, 190, 4, 21, 190, 190, 196, 197, 50, 6, 10]



    
    
      
        
        
        
        




