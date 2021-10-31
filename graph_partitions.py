import networkx
import pandas as pd
import numpy as np
import random
import community as community_louvain
import pygrank as pg
from pygrank.algorithms import HeatKernel
from pygrank.algorithms import AbsorbingWalks
from metrics import *


""" Change the communities and deteriorate the organization """
def perturb(node2community, graph, parameter):
    
    node2perturbed = node2community.copy()
    
    for i in range(parameter):
        
        while True:
            
            n1 = random.randint(min(node2perturbed.keys()), max(node2perturbed.keys()))
            n2 = random.randint(min(node2perturbed.keys()), max(node2perturbed.keys()))
            if n1 != n2:
                break
        
        temp = node2perturbed[n1]
        node2perturbed[n1] = node2perturbed[n2]
        node2perturbed[n2] = temp

    return node2perturbed
    
    


""" Page rank on the deteriorated community structure """
def remodularize(method, graph, node2perturbed):
    
        
    
    nodes = list(graph.nodes())
    perturbed_communities = list(set(node2perturbed.values()))
    personalization_array = []
    personalization = {k: 0 for k in nodes}
    absorption = {k: 1 for k in nodes}
    
    # Construct personalization array
    for i in perturbed_communities:
        personalization = {k: 0 for k in nodes}
        for j in nodes:
            if(i == node2perturbed[j]):
                personalization[j] = 1
                
        personalization_array.append(personalization)
    
    if(method == "pagerank"):
        
        ppr_communities = {k : [] for k in nodes}
        ranker = pg.PageRank(alpha=0.97, tol=1.E-9, max_iters=2000, normalization="symmetric")
        #ranker = AbsorbingWalks(0.99,tol=1.E-9,max_iters=1000)
        
        for i in range(len(personalization_array)):
            ranks = ranker(graph,personalization_array[i])
            for j in nodes:
                ppr_communities[j].append(ranks[j])
                
        
        node2remodularized = {k: 0 for k in nodes}
        coms = []
        for i in nodes:
            c = sorted(range(len(ppr_communities[i])), key=lambda k:ppr_communities[i][k])
            coms.append(c[-1])
            
        final_coms = list(set(coms))
                          
        for i in range(len(coms)):
            coms[i] = final_coms.index(coms[i]) + 1
    
        for i in range(len(coms)):
            node2remodularized[i+1] = coms[i]
                    
            
    
    return node2remodularized
        
        
                
            

    
                
    
    

    
    