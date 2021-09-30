"""
Mojo Distance, Precision and Recall
"""
import numpy as np
from collections import Counter
import re
import math
from collections import defaultdict
from graph_partitions import *


# Maximum Bipartite Matching for Mojo-Distance
class GFG:
    def __init__(self,graph):
          
        # residual graph
        self.graph = graph 
        self.ppl = len(graph)
        self.jobs = len(graph[0])
  
    # A DFS based recursive function
    # that returns true if a matching 
    # for vertex u is possible
    def bpm(self, u, matchR, seen):
  
        for v in range(self.jobs):
  
            if self.graph[u][v] and seen[v] == False:
                  
                # Mark v as visited
                seen[v] = True 
  
                if matchR[v] == -1 or self.bpm(matchR[v], 
                                               matchR, seen):
                    matchR[v] = u
                    return True
        return False
  
    # Returns maximum number of matching 
    def maxBPM(self):
        matchR = [-1] * self.jobs
          
        
        result = 0 
        for i in range(self.ppl):
              
            seen = [False] * self.jobs
              
            if self.bpm(i, matchR, seen):
                result += 1
        return result


# Read partitions from graph_partitions.py
partitionB = ppr_partition(400)
partitionA = ground_truth_partition()

#----- Mojo Distance ---------
n = len(partitionB)
l = len(set(partitionA.values()))
m = len(set(partitionB.values()))

partitionAtags = defaultdict(list)
tags = []


for i in list(partitionA.keys()):
    current = partitionA[i]
    B = partitionB[i]
    tag = [int(s) for s in B.split() if s.isdigit()]
    tag = tag[0]
    
    if(current != partitionA[i]):
        tags = []
    
    partitionAtags[partitionA[i]].append(tag)
   
    
A = [[0]*(l+m) for _ in range(m+l)]
maximums = []

for i in list(partitionAtags.keys()):
    tagA = [int(s) for s in i.split() if s.isdigit()]
    tagA = tagA[0]
    c = Counter(partitionAtags[i])
    
    maximum = [x for x in c if c[x] == c.most_common(1)[0][1]]
    maximums.append(partitionAtags[i].count(maximum[0]))
    for j in maximum:
        A[tagA-1][l+j-1] = 1
       
    
        
    
g = GFG(A)
g = g.maxBPM()
M = n - np.sum(maximums)
MojoDistance = M + l - g
# Normalization
MJ = 1 - MojoDistance/n



#------- Recall & Precision -------------

comsA = set(partitionA.values())
comsB = set(partitionB.values())

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
   
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

comsA = list(set(partitionA.values()))
comsB = list(set(partitionB.values()))

comsA.sort(key=natural_keys)
comsB.sort(key=natural_keys)

Adict = {k: [] for k in comsA}
Bdict = {k: [] for k in comsB}

for i in list(partitionA.keys()):
    Adict[partitionA[i]].append(i)

for i in list(partitionB.keys()):
    Bdict[partitionB[i]].append(i)
    
    
Akeys = list(Adict.keys())
j = 0
mo = 0
for i in list(Bdict.keys()):
    corrects = len(set(Bdict[i]) & set(Adict[Akeys[j]]))
    mo = mo + corrects/len(set(Bdict[i]))
    j = j + 1

prec = mo/88

Akeys = list(Adict.keys())
j = 0
mo = 0
for i in list(Bdict.keys()):
    corrects = len(set(Bdict[i]) & set(Adict[Akeys[j]]))
    mo = mo + corrects/len(set(Adict[Akeys[j]]))
    j = j + 1   
    
rec = mo/88

    
"""
Pair-Precision & Pair-Recall 
"""
Bkeys = list(Bdict.keys())   
j = 0 
total_common = 0
for i in list(Adict.keys()):
    num_common = len(set(Adict[i]) & set(Bdict[Bkeys[j]]))
    total_common = num_common + total_common
    j = j + 1
    
total_common = math.floor(total_common/2)  
cardA = 0
for i in list(Adict.keys()):
    cardA = cardA + math.floor(len(Adict[i])/2)
    
cardB = 0
for i in list(Bdict.keys()):
    cardB = cardB + math.floor(len(Bdict[i])/2)
    
precision = total_common/cardA
recall = total_common/cardB
























